"""Überwachung: Drift messen.

**Warum das nötig ist:** ML-Modelle verfallen leise. Niemand bekommt eine
Fehlermeldung, wenn ein Sensor neu kalibriert wird oder die Produktion auf eine
andere Werkstoffvariante umstellt. Die Güte sinkt einfach, und auffallen tut es
erst, wenn sich jemand beschwert.

Gemessen werden drei verschiedene Dinge — und zwar alle drei, weil keines allein
reicht:

1. **Datendrift** — verschieben sich die Eingangsverteilungen? Maß ist der
   Population Stability Index (PSI), ergänzt um einen Kolmogorow-Smirnow-Test
   für numerische und einen Chi²-Test für kategoriale Merkmale. PSI sagt, *wie
   stark* sich etwas verschoben hat, der Test sagt, ob die Verschiebung bei
   dieser Stichprobengröße überhaupt von Zufall zu unterscheiden ist.
2. **Vorhersagedrift** — verschiebt sich die Verteilung der ausgegebenen
   Wahrscheinlichkeiten? Das sieht man sofort, ohne auf wahre Labels zu warten,
   die im Betrieb erst Wochen später eintreffen.
3. **Betriebskennzahlen** — Alarmquote, Antwortzeit, Fehlerrate.

**Warum alle drei:** Es gibt Verschiebungen, bei denen die Alarmquote *sinkt*,
während das Modell blind wird. Wer nur die Alarmquote überwacht, sieht einen
ruhigen Dienst. ``simulate_drift.py`` führt genau diesen Fall vor.

Aufrufe::

    python -m src.monitoring.drift                      # Protokoll gegen Referenz
    python -m src.monitoring.drift --aktuell daten.csv  # Datei gegen Referenz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.config import PATHS, absolut, load_params
from src.utils.logging_setup import hole_logger

log = hole_logger(__name__)

#: Spaltenname der Vorhersagewahrscheinlichkeit in der Referenzstichprobe.
WAHRSCHEINLICHKEIT = "probability"

URTEILE = ("stabil", "beobachten", "handeln")


# ── Maße ────────────────────────────────────────────────────────────────


def psi(referenz: np.ndarray | pd.Series, aktuell: np.ndarray | pd.Series, bins: int = 10) -> float:
    """Population Stability Index zwischen zwei Verteilungen.

    Die Klassengrenzen kommen aus den Perzentilen der **Referenz**: Jede Klasse
    enthält dort etwa gleich viele Zeilen. Die äußeren Grenzen werden auf
    ±unendlich gesetzt, damit aktuelle Werte außerhalb des Referenzbereichs
    nicht verloren gehen — sie sind ja gerade das Interessante.

    Die untere Abschneidegrenze 1e-6 verhindert ``log(0)``: Eine Klasse, die in
    den aktuellen Daten gar nicht vorkommt, soll einen großen, aber endlichen
    Beitrag liefern.

    Returns:
        0 bei identischen Verteilungen, sonst positiv. Deutung: unter 0,1
        stabil, 0,1 bis 0,25 beobachten, darüber Handlungsbedarf.
    """
    referenz = np.asarray(referenz, dtype=float)
    aktuell = np.asarray(aktuell, dtype=float)
    if len(referenz) == 0 or len(aktuell) == 0:
        return float("nan")

    grenzen = np.percentile(referenz, np.linspace(0, 100, bins + 1))
    grenzen = np.unique(grenzen)  # bei vielen gleichen Werten (z. B. 0)
    if len(grenzen) < 2:
        return 0.0
    grenzen[0], grenzen[-1] = -np.inf, np.inf

    r = np.histogram(referenz, bins=grenzen)[0] / len(referenz)
    a = np.histogram(aktuell, bins=grenzen)[0] / len(aktuell)
    r, a = np.clip(r, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - r) * np.log(a / r)))


def psi_kategorial(referenz: pd.Series, aktuell: pd.Series) -> float:
    """PSI für kategoriale Merkmale — über die Anteile der Ausprägungen."""
    ausprägungen = sorted(set(referenz.dropna().unique()) | set(aktuell.dropna().unique()))
    r = np.array([(referenz == a).mean() for a in ausprägungen], dtype=float)
    a = np.array([(aktuell == x).mean() for x in ausprägungen], dtype=float)
    r, a = np.clip(r, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - r) * np.log(a / r)))


def ks(referenz: pd.Series, aktuell: pd.Series) -> tuple[float, float]:
    """Kolmogorow-Smirnow-Test: größter Abstand der Verteilungsfunktionen."""
    ergebnis = ks_2samp(np.asarray(referenz, dtype=float), np.asarray(aktuell, dtype=float))
    return float(ergebnis.statistic), float(ergebnis.pvalue)


def chi2(referenz: pd.Series, aktuell: pd.Series) -> tuple[float, float]:
    """Chi²-Test auf gleiche Verteilung der Ausprägungen."""
    ausprägungen = sorted(set(referenz.dropna().unique()) | set(aktuell.dropna().unique()))
    tabelle = np.array(
        [
            [int((referenz == a).sum()) for a in ausprägungen],
            [int((aktuell == a).sum()) for a in ausprägungen],
        ]
    )
    # Spalten, die in beiden Stichproben leer sind, würde chi² nicht verkraften.
    tabelle = tabelle[:, tabelle.sum(axis=0) > 0]
    if tabelle.shape[1] < 2:
        return 0.0, 1.0
    statistik, p_wert = chi2_contingency(tabelle)[:2]
    return float(statistik), float(p_wert)


def urteil(wert: float, params: dict[str, Any] | None = None) -> str:
    """Übersetzt einen PSI-Wert in stabil / beobachten / handeln."""
    schranken = (params or load_params())["monitoring"]
    if np.isnan(wert):
        return "unbekannt"
    if wert >= schranken["psi_alarm"]:
        return "handeln"
    if wert >= schranken["psi_warn"]:
        return "beobachten"
    return "stabil"


# ── Referenzstichprobe ──────────────────────────────────────────────────


def referenzpfad(params: dict[str, Any] | None = None) -> Path:
    """Pfad der Referenzstichprobe aus params.yaml."""
    return absolut((params or load_params())["monitoring"]["reference_sample"])


def schreibe_referenz(params: dict[str, Any] | None = None) -> tuple[Path, int]:
    """Legt die Referenzstichprobe aus der Validierungsmenge an.

    Geschrieben werden die Merkmalsspalten, die Zielgröße und die
    Wahrscheinlichkeit, die das gespeicherte Modell jeder Zeile gibt. Damit
    enthält eine Datei beide Maßstäbe: Eingangsverteilungen und
    Vorhersageverteilung.

    Returns:
        (Pfad, Anzahl Zeilen).
    """
    import joblib

    params = params or load_params()
    from src.data.split import lade_teilmengen

    _, val, _ = lade_teilmengen()
    umfang = min(int(params["monitoring"]["sample_size"]), len(val))
    stichprobe = val.sample(n=umfang, random_state=params["seed"]).reset_index(drop=True)

    modell_pfad = PATHS.models / "model.joblib"
    inhalt = joblib.load(modell_pfad)
    ziel = params["data"]["target"]
    merkmale = stichprobe.drop(columns=[ziel])
    # Auf sechs Stellen gerundet, damit die Datei byteweise reproduzierbar ist.
    # Grund: Der Wald rechnet mit n_jobs=-1, und die Reihenfolge, in der die
    # Mittelwerte der 150 Bäume aufsummiert werden, schwankt zwischen Läufen.
    # Das ändert das Ergebnis um ein Maschinenepsilon (gemessen: 2,2e-16) und
    # damit die letzte Stelle in der CSV-Datei. Für Verteilungsvergleiche sind
    # sechs Stellen weit mehr als nötig; eine Referenzdatei, deren Hash sich
    # ohne Grund ändert, ist dagegen wertlos.
    stichprobe[WAHRSCHEINLICHKEIT] = inhalt["pipeline"].predict_proba(merkmale)[:, 1].round(6)

    pfad = referenzpfad(params)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    stichprobe.to_csv(pfad, index=False)
    log.info("Referenzstichprobe geschrieben: %s (%d Zeilen)", PATHS.relativ(pfad), len(stichprobe))
    return pfad, len(stichprobe)


def lade_referenz(params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Liest die Referenzstichprobe.

    Raises:
        FileNotFoundError: wenn sie fehlt; dann erst die Pipeline laufen lassen.
    """
    pfad = referenzpfad(params)
    if not pfad.exists():
        raise FileNotFoundError(
            f"Keine Referenzstichprobe unter {pfad}. "
            "Bitte 'python -m src.pipelines.run_pipeline' ausführen."
        )
    return pd.read_csv(pfad)


def aus_protokoll(grenze: int | None = None) -> pd.DataFrame:
    """Baut die aktuellen Daten aus dem Vorhersageprotokoll (Etappe 17).

    Das ist der Betriebsfall: verglichen wird nicht gegen eine Testdatei, sondern
    gegen das, was die Schnittstelle tatsächlich gesehen hat.

    Returns:
        DataFrame mit den Eingangswerten und der Spalte ``probability``. Leer,
        wenn noch nichts protokolliert wurde.
    """
    from src.serving.prediction_log import aus_params as protokoll_aus_params

    protokoll = protokoll_aus_params()
    zeilen: list[dict[str, Any]] = []

    if protokoll.sqlite_pfad is not None and protokoll.sqlite_pfad.exists():
        for eintrag in protokoll.zeilen(grenze or 1_000_000):
            eingaben = json.loads(eintrag["inputs"]) if eintrag["inputs"] else {}
            zeilen.append({**eingaben, WAHRSCHEINLICHKEIT: eintrag["probability"]})
    elif protokoll.jsonl_pfad is not None and protokoll.jsonl_pfad.exists():
        for text in protokoll.jsonl_pfad.read_text(encoding="utf-8").splitlines():
            if not text.strip():
                continue
            eintrag = json.loads(text)
            zeilen.append(
                {**(eintrag.get("inputs") or {}), WAHRSCHEINLICHKEIT: eintrag["probability"]}
            )
        if grenze:
            zeilen = zeilen[-grenze:]

    return pd.DataFrame(zeilen)


# ── Vergleich ───────────────────────────────────────────────────────────


def vergleiche(
    referenz: pd.DataFrame,
    aktuell: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Vergleicht alle gemeinsamen Spalten und gibt eine Tabelle zurück.

    Je Merkmal: PSI, der passende Test mit p-Wert und ein Urteil. Numerische
    Merkmale bekommen Kolmogorow-Smirnow, kategoriale Chi².
    """
    params = params or load_params()
    bins = int(params["monitoring"]["psi_bins"])
    alpha = float(params["monitoring"]["alpha"])
    ziel = params["data"]["target"]

    # Die Wahrscheinlichkeitsspalte ist kein Eingangsmerkmal: Sie wird in
    # betriebskennzahlen() als Vorhersagedrift geführt und würde die
    # Merkmalstabelle sonst doppelt belegen.
    ausgenommen = {ziel, WAHRSCHEINLICHKEIT}
    gemeinsam = [s for s in referenz.columns if s in aktuell.columns and s not in ausgenommen]
    zeilen = []
    for spalte in gemeinsam:
        r, a = referenz[spalte].dropna(), aktuell[spalte].dropna()
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(a):
            wert = psi(r, a, bins=bins)
            statistik, p_wert = ks(r, a)
            test = "KS"
        else:
            wert = psi_kategorial(r, a)
            statistik, p_wert = chi2(r.astype(str), a.astype(str))
            test = "Chi²"

        zeilen.append(
            {
                "Merkmal": spalte,
                "PSI": round(wert, 4),
                "Urteil": urteil(wert, params),
                "Test": test,
                "Statistik": round(statistik, 4),
                "p-Wert": round(p_wert, 6),
                "signifikant": bool(p_wert < alpha),
                "Referenz Mittel": round(float(r.mean()), 3)
                if pd.api.types.is_numeric_dtype(r)
                else "—",
                "Aktuell Mittel": round(float(a.mean()), 3)
                if pd.api.types.is_numeric_dtype(a)
                else "—",
            }
        )

    tabelle = pd.DataFrame(zeilen)
    if not tabelle.empty:
        tabelle = tabelle.sort_values("PSI", ascending=False).reset_index(drop=True)
    return tabelle


def betriebskennzahlen(
    referenz: pd.DataFrame,
    aktuell: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Alarmquote und Vorhersagedrift — die beiden Zahlen ohne wahre Labels."""
    params = params or load_params()
    schwelle = float(params["decision"]["threshold"])

    ergebnis: dict[str, Any] = {"threshold": schwelle}
    if WAHRSCHEINLICHKEIT in referenz and WAHRSCHEINLICHKEIT in aktuell:
        r, a = referenz[WAHRSCHEINLICHKEIT], aktuell[WAHRSCHEINLICHKEIT]
        wert = psi(r, a, bins=int(params["monitoring"]["psi_bins"]))
        ergebnis.update(
            {
                "prediction_psi": round(wert, 4),
                "prediction_verdict": urteil(wert, params),
                "alarm_rate_reference": round(float((r >= schwelle).mean()), 4),
                "alarm_rate_current": round(float((a >= schwelle).mean()), 4),
                "mean_probability_reference": round(float(r.mean()), 4),
                "mean_probability_current": round(float(a.mean()), 4),
            }
        )
    return ergebnis


def bericht(
    referenz: pd.DataFrame,
    aktuell: pd.DataFrame,
    quelle: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stellt den vollständigen Driftbericht zusammen und legt ihn ab."""
    params = params or load_params()
    tabelle = vergleiche(referenz, aktuell, params)
    kennzahlen = betriebskennzahlen(referenz, aktuell, params)

    mindestzeilen = int(params["monitoring"]["min_rows"])
    genug = len(aktuell) >= mindestzeilen

    urteile = list(tabelle["Urteil"]) + [kennzahlen.get("prediction_verdict", "stabil")]
    if not genug:
        gesamturteil = "zu wenige Daten"
    elif "handeln" in urteile:
        gesamturteil = "handeln"
    elif "beobachten" in urteile:
        gesamturteil = "beobachten"
    else:
        gesamturteil = "stabil"

    inhalt = {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": quelle,
        "reference_rows": int(len(referenz)),
        "current_rows": int(len(aktuell)),
        "min_rows": mindestzeilen,
        "enough_data": genug,
        "verdict": gesamturteil,
        "thresholds": {
            "psi_warn": params["monitoring"]["psi_warn"],
            "psi_alarm": params["monitoring"]["psi_alarm"],
            "alpha": params["monitoring"]["alpha"],
        },
        "features": tabelle.to_dict(orient="records"),
        "operations": kennzahlen,
    }

    pfad = absolut(params["monitoring"]["drift_report"])
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(json.dumps(inhalt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    inhalt["_report_path"] = PATHS.relativ(pfad)
    return inhalt


def drucke(inhalt: dict[str, Any], tabelle: pd.DataFrame) -> None:
    """Gibt den Bericht lesbar auf der Kommandozeile aus."""
    print(
        f"Referenz: {inhalt['reference_rows']} Zeilen   "
        f"aktuell: {inhalt['current_rows']} Zeilen   "
        f"Quelle: {inhalt['source']}\n"
    )

    if not tabelle.empty:
        print(tabelle.to_string(index=False))
    else:
        print("Keine gemeinsamen Spalten zum Vergleichen.")

    betrieb = inhalt["operations"]
    if "prediction_psi" in betrieb:
        print(
            f"\nVorhersagedrift: PSI {betrieb['prediction_psi']} ({betrieb['prediction_verdict']})"
        )
        print(f"  Alarmquote Referenz : {betrieb['alarm_rate_reference']:.1%}")
        print(f"  Alarmquote aktuell  : {betrieb['alarm_rate_current']:.1%}")

    print(f"\nGesamturteil: {inhalt['verdict'].upper()}")
    if not inhalt["enough_data"]:
        print(
            f"  (weniger als {inhalt['min_rows']} aktuelle Zeilen — "
            "zu wenig für ein belastbares Urteil)"
        )
    print(f"Bericht: {inhalt.get('_report_path')}")


def _main() -> None:
    zerleger = argparse.ArgumentParser(
        description="Vergleicht aktuelle Daten gegen die Referenzstichprobe"
    )
    zerleger.add_argument(
        "--aktuell",
        default=None,
        help="CSV- oder Parquet-Datei mit aktuellen Daten. "
        "Ohne Angabe wird das Vorhersageprotokoll verwendet.",
    )
    zerleger.add_argument(
        "--grenze", type=int, default=None, help="nur die letzten N Protokolleinträge"
    )
    argumente = zerleger.parse_args()

    params = load_params()
    referenz = lade_referenz(params)

    if argumente.aktuell:
        pfad = Path(argumente.aktuell)
        aktuell = pd.read_parquet(pfad) if pfad.suffix == ".parquet" else pd.read_csv(pfad)
        quelle = str(pfad)
    else:
        aktuell = aus_protokoll(argumente.grenze)
        quelle = "Vorhersageprotokoll"
        if aktuell.empty:
            print("Das Vorhersageprotokoll ist leer — noch keine Vorhersagen.")
            print("Dienst starten und /predict aufrufen, oder --aktuell angeben.")
            return

    inhalt = bericht(referenz, aktuell, quelle, params)
    drucke(inhalt, vergleiche(referenz, aktuell, params))


if __name__ == "__main__":
    _main()
