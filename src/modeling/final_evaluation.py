"""Die einmalige Messung auf der Testmenge.

**Warum dieses Modul eine Sperre hat.** Die Testmenge ist die einzige Zahl im
Projekt, die eine Aussage über unbekannte Daten erlaubt — und das gilt nur, wenn
sie genau **einmal** benutzt wird. Wer nach dem Ergebnis noch etwas ändert,
Hyperparameter, Schwellenwert, Merkmale, hat die Testmenge zur zweiten
Validierungsmenge gemacht. Sie ist dann verbraucht, und die nächste Messung sagt
nur noch, wie gut man sie getroffen hat.

Deshalb verweigert dieses Modul einen zweiten Lauf, solange
``reports/final_test_evaluation.json`` vorliegt. Die Sperre ist kein technischer
Zwang — mit ``--force`` geht es weiter, und dann wird im Bericht festgehalten,
dass es nicht der erste Lauf war. Sie soll nur verhindern, dass es *aus
Versehen* passiert.

Gemessen wird mit dem Modell und dem Schwellenwert, die festliegen: aus
``models/model.joblib``, nicht neu trainiert. Nichts an dieser Messung darf noch
eine Entscheidung sein.

Aufruf::

    python -m src.modeling.final_evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import PATHS, load_params
from src.data.load import EXPECTED_SHA256, datei_hash
from src.data.split import lade_teilmengen, trenne_merkmale_und_ziel
from src.utils.logging_setup import hole_logger
from src.utils.tracking import git_commit

log = hole_logger(__name__)

BERICHT = "final_test_evaluation.json"

#: Wiederholungen für das Vertrauensintervall. 2000 reichen für zwei
#: Dezimalstellen und laufen in wenigen Sekunden.
ZIEHUNGEN = 2000


def berichtspfad():
    return PATHS.reports / BERICHT


def vertrauensintervall(
    y_wahr,
    wahrscheinlichkeit: np.ndarray,
    schwelle: float,
    ziehungen: int = ZIEHUNGEN,
    seed: int = 42,
) -> dict[str, Any]:
    """Vertrauensintervall für PR-AUC und Recall, durch Ziehen mit Zurücklegen.

    **Warum das hierher gehört:** Die Testmenge enthält 51 positive Fälle. Eine
    einzelne Kennzahl aus so wenigen Fällen schwankt stark — ohne Intervall liest
    man jeden Unterschied zur Validierungsmenge als echte Verbesserung oder
    Verschlechterung, obwohl er im Rauschen liegen kann.

    Gezogen wird aus derselben einen Messung. Es ist keine zweite Messung und
    keine Entscheidung, sondern die Angabe ihrer Genauigkeit.
    """
    zufall = np.random.default_rng(seed)
    y = np.asarray(y_wahr)
    pr_auc, recall = [], []

    for _ in range(ziehungen):
        auswahl = zufall.integers(0, len(y), len(y))
        y_zug, p_zug = y[auswahl], wahrscheinlichkeit[auswahl]
        if y_zug.sum() == 0:  # Ziehung ohne positive Fälle überspringen
            continue
        pr_auc.append(float(average_precision_score(y_zug, p_zug)))
        recall.append(float(recall_score(y_zug, (p_zug >= schwelle).astype(int), zero_division=0)))

    def intervall(werte: list[float]) -> list[float]:
        return [
            round(float(np.percentile(werte, 2.5)), 4),
            round(float(np.percentile(werte, 97.5)), 4),
        ]

    return {
        "verfahren": f"Ziehen mit Zurücklegen, {len(pr_auc)} Wiederholungen, 95 %",
        "pr_auc": intervall(pr_auc),
        "recall": intervall(recall),
    }


def bewerte_testmenge(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Misst das gespeicherte Modell auf der Testmenge.

    Returns:
        Kennzahlen, Verwechslungsmatrix, Kosten und Herkunftsangaben.
    """
    params = params or load_params()
    schwelle_aus_params = float(params["decision"]["threshold"])
    kosten = params["costs"]

    modell_pfad = PATHS.models / "model.joblib"
    if not modell_pfad.exists():
        raise FileNotFoundError(
            f"Kein Modell unter {modell_pfad}. "
            "Bitte zuerst 'python -m src.pipelines.run_pipeline' ausführen."
        )
    inhalt = joblib.load(modell_pfad)
    pipeline = inhalt["pipeline"]
    # Der Schwellenwert kommt aus der Modelldatei: Er gehört zum Modell wie
    # seine Gewichte. params.yaml wird nur zum Vergleich gelesen.
    schwelle = float(inhalt["schwelle"])

    _, _, test = lade_teilmengen()
    X_test, y_test = trenne_merkmale_und_ziel(test, params)

    wahrscheinlichkeit = pipeline.predict_proba(X_test)[:, 1]
    vorhersage = (wahrscheinlichkeit >= schwelle).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, vorhersage, labels=[0, 1]).ravel()

    erwartete_kosten = float(
        fn * kosten["false_negative_eur"]
        + fp * kosten["false_positive_eur"]
        + tp * kosten["true_positive_eur"]
    )
    nichts_tun = float(int(y_test.sum()) * kosten["false_negative_eur"])

    kennzahlen = {
        "pr_auc": float(average_precision_score(y_test, wahrscheinlichkeit)),
        "roc_auc": float(roc_auc_score(y_test, wahrscheinlichkeit)),
        "recall": float(recall_score(y_test, vorhersage, zero_division=0)),
        "precision": float(precision_score(y_test, vorhersage, zero_division=0)),
        "schwelle": schwelle,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "kosten_eur": erwartete_kosten,
        "kosten_nichts_tun_eur": nichts_tun,
        "ersparnis_eur": nichts_tun - erwartete_kosten,
    }

    # Die Kennzahlen der Validierungsmenge zum Vergleich daneben, damit man den
    # Abstand sieht: Ein deutlich schlechteres Testergebnis wäre ein Zeichen für
    # Überanpassung an die Validierungsmenge.
    val_pfad = PATHS.reports / "model_results.json"
    validierung = (
        json.loads(val_pfad.read_text(encoding="utf-8"))["kennzahlen"] if val_pfad.exists() else {}
    )

    return {
        "kennzahlen_test": kennzahlen,
        "vertrauensintervall_test": vertrauensintervall(
            y_test, wahrscheinlichkeit, schwelle, seed=params["seed"]
        ),
        "kennzahlen_validierung": validierung,
        "testmenge": {
            "zeilen": int(len(test)),
            "ausfaelle": int(y_test.sum()),
            "ausfallrate": round(float(y_test.mean()), 5),
        },
        "schwelle_aus_params": schwelle_aus_params,
        "schwelle_aus_modell": schwelle,
        "herkunft": {
            "model_sha256": datei_hash(modell_pfad),
            "raw_data_sha256": EXPECTED_SHA256,
            "git_commit": git_commit(),
            "seed": params["seed"],
        },
        "qualitaetsschranke": params["quality_gate"],
    }


def _euro(betrag: float) -> str:
    return f"{betrag:,.0f}".replace(",", ".") + " EUR"


def _main() -> None:
    zerleger = argparse.ArgumentParser(description="Einmalige Messung auf der Testmenge")
    zerleger.add_argument(
        "--force", action="store_true", help="auch messen, wenn es schon einen Bericht gibt"
    )
    zerleger.add_argument(
        "--grund",
        default=None,
        help="Begründung für eine Wiederholung; wird in den Bericht geschrieben",
    )
    argumente = zerleger.parse_args()

    # Eine Wiederholung ohne Begründung gibt es nicht. Die Sperre soll nicht
    # verhindern, dass man erneut messen kann - sie soll verhindern, dass es
    # unbemerkt und unbegründet geschieht.
    if argumente.force and not argumente.grund:
        print('--force verlangt --grund "...": Warum wird erneut gemessen?')
        print("Die Begründung wird in den Bericht geschrieben.")
        sys.exit(2)

    pfad = berichtspfad()
    wiederholung = pfad.exists()
    if wiederholung and not argumente.force:
        print(f"Es gibt bereits einen Bericht: {PATHS.relativ(pfad)}")
        vorher = json.loads(pfad.read_text(encoding="utf-8"))
        print(f"Gemessen am: {vorher['gemessen_am']}")
        print(f"PR-AUC auf der Testmenge: {vorher['kennzahlen_test']['pr_auc']:.4f}")
        print()
        print("Die Testmenge ist für genau eine Messung gedacht. Wird nach dem")
        print("Ergebnis noch etwas verändert, ist sie zur zweiten")
        print("Validierungsmenge geworden und sagt nichts mehr über unbekannte")
        print("Daten. Wenn es trotzdem sein muss: --force (wird im Bericht")
        print("festgehalten).")
        sys.exit(1)

    ergebnis = bewerte_testmenge()
    ergebnis["gemessen_am"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ergebnis["erster_lauf"] = not wiederholung
    if wiederholung:
        ergebnis["wiederholungsgrund"] = argumente.grund

    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8")

    t = ergebnis["kennzahlen_test"]
    v = ergebnis["kennzahlen_validierung"]
    menge = ergebnis["testmenge"]

    print("Einmalige Messung auf der Testmenge")
    print(f"  {menge['zeilen']} Zeilen, {menge['ausfaelle']} Ausfälle ({menge['ausfallrate']:.2%})")
    print(f"  Modell: {ergebnis['herkunft']['model_sha256'][:12]}…, Schwellenwert {t['schwelle']}")
    print()

    kopf = f"  {'Kennzahl':22s} {'Test':>12s} {'Validierung':>14s} {'Abstand':>10s}"
    print(kopf)
    print("  " + "─" * (len(kopf) - 2))
    for name, schluessel in (
        ("PR-AUC", "pr_auc"),
        ("ROC-AUC", "roc_auc"),
        ("Recall", "recall"),
        ("Precision", "precision"),
    ):
        wert = t[schluessel]
        vergleich = v.get(schluessel)
        abstand = f"{wert - vergleich:+.4f}" if vergleich is not None else "—"
        print(
            f"  {name:22s} {wert:>12.4f} "
            f"{vergleich if vergleich is None else f'{vergleich:.4f}':>14} "
            f"{abstand:>10s}"
        )

    intervall = ergebnis["vertrauensintervall_test"]
    print()
    print(f"  Vertrauensintervall ({intervall['verfahren']})")
    print(f"    PR-AUC  {intervall['pr_auc'][0]:.4f} bis {intervall['pr_auc'][1]:.4f}")
    print(f"    Recall  {intervall['recall'][0]:.4f} bis {intervall['recall'][1]:.4f}")
    print()
    print(f"  Verwechslungsmatrix   TP {t['tp']}  FP {t['fp']}  FN {t['fn']}  TN {t['tn']}")
    print(f"  Erwartete Kosten      {_euro(t['kosten_eur'])}")
    print(f"  Nichts tun            {_euro(t['kosten_nichts_tun_eur'])}")
    print(f"  Ersparnis             {_euro(t['ersparnis_eur'])}")
    print()

    schranken = ergebnis["qualitaetsschranke"]
    for name, wert, grenze in (
        ("PR-AUC", t["pr_auc"], schranken["min_pr_auc"]),
        ("Recall", t["recall"], schranken["min_recall"]),
    ):
        urteil = "erfüllt" if wert >= grenze else "NICHT ERFÜLLT"
        print(f"  Erfolgsschwelle {name:8s} >= {grenze}: {wert:.4f} — {urteil}")

    print()
    print(f"geschrieben: {PATHS.relativ(pfad)}")
    print()
    print("Diese Zahl ist endgültig. Ab hier darf an Modell, Merkmalen oder")
    print("Schwellenwert nichts mehr verändert werden — sonst wäre sie keine")
    print("Aussage über unbekannte Daten mehr.")

    log.info("Testmengenmessung abgeschlossen: PR-AUC %.4f, Recall %.4f", t["pr_auc"], t["recall"])


if __name__ == "__main__":
    _main()
