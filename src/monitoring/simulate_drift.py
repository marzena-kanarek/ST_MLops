"""Drift simulieren und zeigen, dass die Überwachung anschlägt.

Eine Überwachung, die nie angeschlagen hat, ist unbelegt. Dieses Skript
verändert die Eingangsdaten auf drei Weisen, die im Betrieb tatsächlich
vorkommen, und prüft, ob die Maße aus ``drift.py`` das bemerken:

* **temperatur_kalibrierung** — ein Temperatursensor wird neu kalibriert und
  meldet 3 K zu viel. Niemand sagt dem Modell etwas davon.
* **verschleiss_hoch** — Werkzeuge werden länger im Einsatz gelassen als bisher.
* **typverteilung** — die Produktion stellt überwiegend auf die Variante H um.

Zusätzlich läuft **keine_drift** als Kontrollfall: unveränderte Daten. Eine
Überwachung, die auch dort anschlägt, wäre wertlos, weil sie nur Lärm erzeugt.

**Der lehrreiche Fall.** Weil hier wahre Labels vorliegen, lässt sich mitmessen,
was die Überwachung im Betrieb erst Wochen später erfahren würde. Bei der
Temperaturkalibrierung **sinkt die Alarmquote**, während der Recall einbricht:
Das Modell sieht lauter scheinbar unkritische Betriebspunkte und schweigt. Wer
nur die Alarmquote überwacht, sieht einen ruhigen Dienst und übersieht, dass das
Modell blind geworden ist. Genau deshalb werden Datendrift, Vorhersagedrift und
Betriebskennzahlen zusammen überwacht.

**Was diese Simulation nicht kann.** Die wahren Labels werden nicht
mitsimuliert: Sie gehören zu den ursprünglichen Betriebspunkten. Der Recall nach
einer Verschiebung sagt also *„so empfindlich reagiert das Modell auf diese
Verschiebung"* und nicht *„so viele Ausfälle gäbe es in einem so veränderten
Prozess"*. Für die Frage, ob die Überwachung anschlägt, genügt das; für eine
Aussage über den veränderten Prozess bräuchte man ein Simulationsmodell der
Maschine.

Aufruf::

    python -m src.monitoring.simulate_drift
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.config import PATHS, absolut, load_params
from src.data.split import lade_teilmengen, trenne_merkmale_und_ziel
from src.monitoring.drift import (
    WAHRSCHEINLICHKEIT,
    betriebskennzahlen,
    lade_referenz,
    vergleiche,
)
from src.utils.logging_setup import hole_logger

log = hole_logger(__name__)


def kalibrierungsfehler(frame: pd.DataFrame) -> pd.DataFrame:
    """Prozesstemperatursensor meldet nach einer Neukalibrierung 3 K zu viel.

    Das Vorzeichen ist hier das Entscheidende. Die Temperaturdifferenz
    (Prozess minus Luft) wächst dadurch um 3 K, und eine große Differenz
    bedeutet für das Modell **gute Wärmeabfuhr**, also Entwarnung. Die Maschinen
    laufen unverändert, das Modell hält sie aber für unkritischer als vorher.
    """
    frame = frame.copy()
    frame["process_temperature_k"] = frame["process_temperature_k"] + 3.0
    return frame


def hoher_verschleiss(frame: pd.DataFrame) -> pd.DataFrame:
    """Werkzeuge werden 60 Minuten länger genutzt als bisher."""
    frame = frame.copy()
    frame["tool_wear_min"] = frame["tool_wear_min"] + 60
    return frame


def andere_typverteilung(frame: pd.DataFrame) -> pd.DataFrame:
    """Die Produktion stellt überwiegend auf die Variante H um."""
    frame = frame.copy()
    anzahl = int(len(frame) * 0.8)
    frame.loc[frame.index[:anzahl], "type"] = "H"
    return frame


def unveraendert(frame: pd.DataFrame) -> pd.DataFrame:
    """Kontrollfall: nichts ändern."""
    return frame.copy()


SZENARIEN: dict[str, tuple[Callable[[pd.DataFrame], pd.DataFrame], str]] = {
    "keine_drift": (unveraendert, "Kontrollfall, unveränderte Daten"),
    "temperatur_kalibrierung": (kalibrierungsfehler, "Prozesstemperatur 3 K zu hoch gemessen"),
    "verschleiss_hoch": (hoher_verschleiss, "Werkzeugverschleiß +60 min"),
    "typverteilung": (andere_typverteilung, "80 % der Teile auf Variante H"),
}


def bewerte_szenario(
    name: str,
    beschreibung: str,
    X: pd.DataFrame,
    y: pd.Series,
    pipeline,
    referenz: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Wendet ein Szenario an und misst Drift, Alarmquote und Güte."""
    wandel, _ = SZENARIEN[name]
    verschoben = wandel(X)

    wahrscheinlichkeit = pipeline.predict_proba(verschoben)[:, 1]
    schwelle = float(params["decision"]["threshold"])
    vorhersage = (wahrscheinlichkeit >= schwelle).astype(int)

    aktuell = verschoben.copy()
    aktuell[WAHRSCHEINLICHKEIT] = wahrscheinlichkeit

    tabelle = vergleiche(referenz, aktuell, params)
    kennzahlen = betriebskennzahlen(referenz, aktuell, params)

    staerkstes = tabelle.iloc[0] if not tabelle.empty else None
    urteile = list(tabelle["Urteil"]) + [kennzahlen.get("prediction_verdict", "stabil")]
    gesamt = (
        "handeln" if "handeln" in urteile else "beobachten" if "beobachten" in urteile else "stabil"
    )

    kosten = params["costs"]
    richtig_positiv = int(((vorhersage == 1) & (y == 1)).sum())
    falsch_positiv = int(((vorhersage == 1) & (y == 0)).sum())
    falsch_negativ = int(((vorhersage == 0) & (y == 1)).sum())

    return {
        "scenario": name,
        "description": beschreibung,
        "verdict": gesamt,
        "max_psi": float(staerkstes["PSI"]) if staerkstes is not None else 0.0,
        "max_psi_feature": str(staerkstes["Merkmal"]) if staerkstes is not None else "—",
        "prediction_psi": kennzahlen.get("prediction_psi"),
        "prediction_verdict": kennzahlen.get("prediction_verdict"),
        "alarm_rate": round(float(vorhersage.mean()), 4),
        "recall": round(float(recall_score(y, vorhersage, zero_division=0)), 4),
        "precision": round(float(precision_score(y, vorhersage, zero_division=0)), 4),
        "costs_eur": float(
            falsch_negativ * kosten["false_negative_eur"]
            + falsch_positiv * kosten["false_positive_eur"]
            + richtig_positiv * kosten["true_positive_eur"]
        ),
        "features": tabelle.to_dict(orient="records"),
    }


def fuehre_aus(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rechnet alle Szenarien und legt den Bericht ab."""
    params = params or load_params()
    referenz = lade_referenz(params)

    # Bewertet wird auf der Validierungsmenge. Die Testmenge bleibt für die
    # einzige Abschlussmessung unangetastet.
    _, val, _ = lade_teilmengen()
    X, y = trenne_merkmale_und_ziel(val, params)
    pipeline = joblib.load(PATHS.models / "model.joblib")["pipeline"]

    ergebnisse = [
        bewerte_szenario(name, beschreibung, X, y, pipeline, referenz, params)
        for name, (_, beschreibung) in SZENARIEN.items()
    ]

    inhalt = {
        "simulated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_rows": int(len(referenz)),
        "evaluated_rows": int(len(X)),
        "evaluated_on": "validierungsmenge",
        "threshold": float(params["decision"]["threshold"]),
        "thresholds": {
            "psi_warn": params["monitoring"]["psi_warn"],
            "psi_alarm": params["monitoring"]["psi_alarm"],
        },
        "scenarios": ergebnisse,
    }

    pfad = absolut(params["monitoring"]["simulation_report"])
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(inhalt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    inhalt["_report_path"] = PATHS.relativ(pfad)
    return inhalt


def _main() -> None:
    zerleger = argparse.ArgumentParser(
        description="Simuliert Drift und prüft, ob die Überwachung anschlägt"
    )
    zerleger.add_argument(
        "--details", action="store_true", help="PSI je Merkmal für jedes Szenario ausgeben"
    )
    argumente = zerleger.parse_args()

    inhalt = fuehre_aus()
    szenarien = inhalt["scenarios"]
    grundfall = szenarien[0]

    def euro(betrag: float) -> str:
        return f"{betrag:,.0f}".replace(",", ".") + " EUR"

    print(
        f"Referenz: {inhalt['reference_rows']} Zeilen   "
        f"bewertet auf {inhalt['evaluated_rows']} Zeilen der "
        f"{inhalt['evaluated_on']}   Schwelle {inhalt['threshold']}\n"
    )

    kopf = (
        f"{'Szenario':26s} {'größtes PSI':>22s} {'Vorh.-PSI':>10s} "
        f"{'Alarmquote':>11s} {'Recall':>8s} {'Precision':>10s} "
        f"{'Kosten':>13s}  Urteil"
    )
    print(kopf)
    print("─" * len(kopf))
    for s in szenarien:
        psi_text = f"{s['max_psi']:.3f} ({s['max_psi_feature'][:12]})"
        print(
            f"{s['scenario']:26s} {psi_text:>22s} "
            f"{s['prediction_psi']:>10.3f} "
            f"{s['alarm_rate']:>10.1%} {s['recall']:>8.3f} "
            f"{s['precision']:>10.3f} {euro(s['costs_eur']):>13s}  "
            f"{s['verdict']}"
        )

    if argumente.details:
        for s in szenarien:
            print(f"\n── {s['scenario']} — {s['description']}")
            print(
                pd.DataFrame(s["features"])[
                    ["Merkmal", "PSI", "Urteil", "Test", "p-Wert", "signifikant"]
                ].to_string(index=False)
            )

    # ── Der lehrreiche Fall ─────────────────────────────────────────────
    # Gesucht wird das Szenario, bei dem die Alarmquote praktisch unverändert
    # bleibt, der Recall aber deutlich fällt. "Unverändert" ist der gefährliche
    # Fall, nicht nur "sinkend": Eine Alarmquote, die gleich bleibt, sieht auf
    # jedem Betriebsschaubild nach Normalbetrieb aus.
    #
    # Welches Szenario das ist, wird aus den Zahlen gesucht und nicht im Text
    # festgeschrieben - die Messung soll die Aussage bestimmen.
    heimlich = [
        s
        for s in szenarien[1:]
        if abs(s["alarm_rate"] - grundfall["alarm_rate"]) <= 0.25 * grundfall["alarm_rate"]
        and s["recall"] <= grundfall["recall"] - 0.05
    ]
    print(f"\n{'─' * 78}")
    if heimlich:
        s = max(heimlich, key=lambda e: grundfall["recall"] - e["recall"])
        print(f"Der lehrreiche Fall — {s['scenario']} ({s['description']}):")
        richtung = "sinkt" if s["alarm_rate"] < grundfall["alarm_rate"] else "praktisch unverändert"
        print(
            f"  Alarmquote {grundfall['alarm_rate']:.1%} -> {s['alarm_rate']:.1%}"
            f"   ({richtung} - der Dienst wirkt wie im Normalbetrieb)"
        )
        print(
            f"  Recall     {grundfall['recall']:.3f} -> {s['recall']:.3f}"
            "   (bricht ein - übersehene Ausfälle)"
        )
        print(f"  Kosten     {euro(grundfall['costs_eur'])} -> {euro(s['costs_eur'])}")
        print("  Wer nur die Alarmquote überwacht, sieht einen ruhigen Dienst und")
        print("  übersieht, dass das Modell blind geworden ist. Die Datendrift")
        print(
            f"  zeigt es trotzdem an: PSI {s['max_psi']:.3f} auf "
            f"{s['max_psi_feature']}, Vorhersagedrift "
            f"{s['prediction_psi']:.3f}."
        )
    else:
        print("In diesem Lauf gibt es kein Szenario, in dem die Alarmquote sinkt")
        print("und der Recall gleichzeitig fällt. Die Betriebskennzahlen allein")
        print("bleiben trotzdem unzureichend - sie sagen nichts über die Güte.")

    # ── Die Gegenrichtung ───────────────────────────────────────────────
    # Ein PSI-Alarm bedeutet "die Eingangsdaten haben sich verändert", nicht
    # "das Modell ist schlechter geworden". Wenn ein Szenario genau das zeigt,
    # gehört es dazu - sonst führt die Überwachung zu blinden Neutrainings.
    harmlos = [
        s
        for s in szenarien[1:]
        if s["verdict"] == "handeln"
        and s["recall"] >= grundfall["recall"] - 0.01
        and abs(s["costs_eur"] - grundfall["costs_eur"]) < 1.0
    ]
    if harmlos:
        s = harmlos[0]
        print(f"\nDie Gegenrichtung — {s['scenario']} ({s['description']}):")
        print(f"  Datendrift deutlich: PSI {s['max_psi']:.3f} auf {s['max_psi_feature']}")
        print(f"  Güte unverändert:    Recall {s['recall']:.3f}, Kosten {euro(s['costs_eur'])}")
        print("  Ein PSI-Alarm heißt 'die Eingangsdaten haben sich verändert',")
        print("  nicht 'das Modell ist schlechter geworden'. Dieses Merkmal trägt")
        print("  kaum zur Vorhersage bei. Deshalb löst ein PSI-Alarm im Konzept")
        print("  eine Prüfung aus und nicht automatisch ein Neutraining.")

    # ── Selbstprüfung ───────────────────────────────────────────────────
    print(f"\n{'─' * 78}")
    maengel = []
    if grundfall["verdict"] != "stabil":
        maengel.append("der Kontrollfall ohne Drift schlägt an (Fehlalarm)")
    for s in szenarien[1:]:
        if s["verdict"] == "stabil":
            maengel.append(f"{s['scenario']} wird nicht erkannt")

    if maengel:
        print("ÜBERWACHUNG UNZUREICHEND:")
        for mangel in maengel:
            print(f"  - {mangel}")
        raise SystemExit(1)

    print("Selbstprüfung bestanden: Kontrollfall ruhig, alle drei Driftszenarien erkannt.")
    print(f"Bericht: {inhalt['_report_path']}")


if __name__ == "__main__":
    _main()
