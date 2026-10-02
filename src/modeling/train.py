"""Trainingslauf mit Protokollierung (Etappe 11).

Ein Aufruf trainiert das in ``params.yaml`` festgelegte Modell, bewertet es auf
der Validierungsmenge und hinterlässt eine vollständige Spur in MLflow:
Parameter, Kennzahlen, Abbildungen, Herkunftsangaben und das Modell selbst.

**Warum überhaupt protokollieren:** Nach dreißig Läufen weiß niemand mehr, welche
Kombination die 0,86 erzeugt hat. Der Zettel mit den Zahlen geht verloren, das
Notebook wird überschrieben.

**Der Schalter ``--no-mlflow``** ist Absicht: In Tests und in der CI soll keine
Laufdatenbank entstehen. Der Lauf funktioniert dann genauso, er hinterlässt nur
keine Spur.

Aufrufe::

    python -m src.modeling.train
    python -m src.modeling.train --no-mlflow
    python -m src.modeling.train --name "rf_tiefer" --n-estimators 300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PATHS, load_params, mlflow_tracking_uri
from src.data.load import EXPECTED_SHA256
from src.data.split import lade_teilmengen, trenne_merkmale_und_ziel
from src.features.build_features import baue_pipeline
from src.modeling.quality_gate import pruefe_qualitaet
from src.modeling.threshold import kosten_je_schwelle
from src.utils.tracking import bibliotheksversionen, herkunft

#: Name des Experiments und Ablageort der Laufdaten — beides aus params.yaml,
#: Abschnitt ``tracking``. Der Ablageort ist über die Umgebungsvariable
#: MLFLOW_TRACKING_URI umstellbar (nötig in der CI und auf Netzlaufwerken, auf
#: denen SQLite nicht sperren kann).
EXPERIMENT = load_params()["tracking"]["experiment"]
TRACKING_URI = mlflow_tracking_uri()


def setze_startwerte(seed: int) -> None:
    """Legt alle Zufallsquellen fest, die das Ergebnis beeinflussen."""
    random.seed(seed)
    np.random.seed(seed)


def bewerte(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    kosten: dict[str, float],
    schwelle: float,
) -> dict[str, float]:
    """Kennzahlen auf der Validierungsmenge, bei der festgelegten Schwelle."""
    wahrscheinlichkeit = pipeline.predict_proba(X_val)[:, 1]
    kurve = kosten_je_schwelle(y_val, wahrscheinlichkeit, kosten, schwellen=np.array([schwelle]))
    zeile = kurve.iloc[0]
    return {
        "pr_auc": float(average_precision_score(y_val, wahrscheinlichkeit)),
        "roc_auc": float(roc_auc_score(y_val, wahrscheinlichkeit)),
        "recall": float(zeile["Recall"]),
        "precision": float(zeile["Precision"]),
        "kosten_eur": float(zeile["Kosten EUR"]),
        "schwelle": float(schwelle),
        "tp": int(zeile["TP"]),
        "fp": int(zeile["FP"]),
        "fn": int(zeile["FN"]),
    }


def trainiere(
    lauf_name: str | None = None,
    mit_mlflow: bool = True,
    ueberschreibungen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Führt einen vollständigen Trainingslauf aus.

    Args:
        lauf_name: Name des Laufs in MLflow.
        mit_mlflow: False schaltet die Protokollierung ab (Tests, CI).
        ueberschreibungen: einzelne Hyperparameter abweichend von params.yaml.

    Returns:
        Die Kennzahlen des Laufs.
    """
    params = load_params()
    seed = params["seed"]
    setze_startwerte(seed)

    hyperparameter = dict(params["model"]["random_forest"])
    hyperparameter.update(ueberschreibungen or {})
    schwelle = params["decision"]["threshold"]

    train, val, _ = lade_teilmengen()
    X_train, y_train = trenne_merkmale_und_ziel(train, params)
    X_val, y_val = trenne_merkmale_und_ziel(val, params)

    pipeline = baue_pipeline(RandomForestClassifier(random_state=seed, n_jobs=-1, **hyperparameter))
    pipeline.fit(X_train, y_train)

    kennzahlen = bewerte(pipeline, X_val, y_val, params["costs"], schwelle)
    angaben = herkunft(EXPECTED_SHA256, seed)

    # ── Qualitätsschranke ───────────────────────────────────────────────
    # Erst prüfen, dann speichern. Ein durchgefallenes Modell überschreibt das
    # bisherige nicht — das ist der ganze Zweck der Schranke.
    schranken = pruefe_qualitaet(kennzahlen, params["quality_gate"])

    # Die Kennzahlen werden immer abgelegt, auch bei Nichtbestehen: Man will
    # nachlesen können, woran es lag.
    ergebnis_pfad = PATHS.reports / "model_results.json"
    ergebnis_pfad.parent.mkdir(parents=True, exist_ok=True)
    ergebnis_pfad.write_text(
        json.dumps(
            {
                "kennzahlen": kennzahlen,
                "hyperparameter": hyperparameter,
                "herkunft": angaben,
                "bibliotheken": bibliotheksversionen(),
                "qualitaetsschranke": {
                    "bestanden": schranken.bestanden,
                    "verstoesse": schranken.verstoesse,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if schranken.bestanden:
        modell_pfad = PATHS.models / "model.joblib"
        modell_pfad.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(
            {
                "pipeline": pipeline,
                "schwelle": schwelle,
                "herkunft": angaben,
                "kennzahlen": kennzahlen,
            },
            modell_pfad,
        )

    if mit_mlflow:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=lauf_name):
            mlflow.log_params({f"rf_{k}": v for k, v in hyperparameter.items()})
            mlflow.log_param("schwelle", schwelle)
            mlflow.log_params({f"herkunft_{k}": v for k, v in angaben.items()})
            mlflow.log_params({f"lib_{k}": v for k, v in bibliotheksversionen().items()})
            mlflow.log_metrics(dict(kennzahlen))
            mlflow.log_metric("schranke_bestanden", int(schranken.bestanden))
            mlflow.set_tag(
                "qualitaetsschranke", "bestanden" if schranken.bestanden else "durchgefallen"
            )
            if schranken.verstoesse:
                mlflow.set_tag("verstoesse", " | ".join(schranken.verstoesse))
            mlflow.log_artifact(str(ergebnis_pfad))

            # cloudpickle statt des neuen skops-Formats: Die Pipeline enthält
            # mit add_engineered_features eine eigene Funktion, die skops aus
            # Sicherheitsgründen nicht ohne ausdrückliche Freigabe lädt.
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            # ── Modellregister ──────────────────────────────────────────
            # Registriert und als "champion" markiert wird nur, was die
            # Schranke bestanden hat. Die Schnittstelle lädt später nicht
            # "das Modell von gestern", sondern "das Modell mit dem Alias
            # champion" — ein Rückfall auf die Vorgängerversion ist damit ein
            # Handgriff statt einer Dateikopie.
            if schranken.bestanden:
                registername = params["model"]["registry_name"]
                version = mlflow.register_model(info.model_uri, registername)
                mlflow.MlflowClient().set_registered_model_alias(
                    registername, "champion", version.version
                )

    return {
        **kennzahlen,
        "schranke_bestanden": schranken.bestanden,
        "verstoesse": schranken.verstoesse,
    }


def _main() -> None:
    zerleger = argparse.ArgumentParser(description="Trainingslauf mit Protokollierung")
    zerleger.add_argument("--name", default=None, help="Name des Laufs in MLflow")
    zerleger.add_argument(
        "--no-mlflow", action="store_true", help="ohne Protokollierung (Tests, CI)"
    )
    zerleger.add_argument(
        "--n-estimators", type=int, default=None, help="Anzahl Bäume abweichend von params.yaml"
    )
    zerleger.add_argument(
        "--max-depth", type=int, default=None, help="Baumtiefe abweichend von params.yaml"
    )
    argumente = zerleger.parse_args()

    ueberschreibungen = {}
    if argumente.n_estimators is not None:
        ueberschreibungen["n_estimators"] = argumente.n_estimators
    if argumente.max_depth is not None:
        ueberschreibungen["max_depth"] = argumente.max_depth

    kennzahlen = trainiere(
        lauf_name=argumente.name,
        mit_mlflow=not argumente.no_mlflow,
        ueberschreibungen=ueberschreibungen or None,
    )

    print("Kennzahlen auf der Validierungsmenge:")
    for name, wert in kennzahlen.items():
        if name in ("schranke_bestanden", "verstoesse"):
            continue
        print(
            f"  {name:12s} {wert:>12.4f}"
            if isinstance(wert, float)
            else f"  {name:12s} {wert:>12d}"
        )

    print(
        "\nQualitätsschranke:",
        "bestanden" if kennzahlen["schranke_bestanden"] else "NICHT BESTANDEN",
    )
    for verstoss in kennzahlen["verstoesse"]:
        print(f"  - {verstoss}")

    if not kennzahlen["schranke_bestanden"]:
        print("\nDas Modell wurde NICHT gespeichert und NICHT registriert.")
        print("Das bisherige models/model.joblib bleibt unverändert.")
        sys.exit(1)

    if not argumente.no_mlflow:
        print(f"\nprotokolliert in {TRACKING_URI}, Experiment '{EXPERIMENT}'")
        print(f"Oberfläche:  mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    _main()
