"""Tests der Driftüberwachung (Etappe 18).

Eine Überwachung muss zwei Dinge können, und beide werden hier geprüft: bei
echter Verschiebung **anschlagen** und bei unveränderten Daten **schweigen**.
Nur die erste Hälfte zu prüfen, ergäbe eine Überwachung, die immer Alarm gibt
und deshalb ignoriert wird.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS, load_params
from src.monitoring import drift

ZUFALL = np.random.default_rng(42)


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture
def vollstaendige_umgebung(params) -> None:
    """Überspringt, wenn Teilmengen, Modell oder Referenz fehlen.

    Die drei Voraussetzungen müssen **einzeln** geprüft werden. Im Git liegt nur
    die Referenzstichprobe; Teilmengen und Modell entstehen erst durch einen
    Pipelinelauf. Ein frischer Klon hat also die Referenz, aber nicht die beiden
    anderen — prüft man nur die Referenz, schlägt der Test in der CI fehl, statt
    sich abzumelden. Genau das ist in Etappe 20 passiert.
    """
    fehlend = [
        beschreibung
        for pfad, beschreibung in (
            (drift.referenzpfad(params), "Referenzstichprobe"),
            (PATHS.processed / "train.parquet", "Teilmengen unter data/processed/"),
            (PATHS.models / "model.joblib", "Modell unter models/"),
        )
        if not pfad.exists()
    ]
    if fehlend:
        pytest.skip(
            "fehlt: "
            + ", ".join(fehlend)
            + " — erst 'python -m src.pipelines.run_pipeline' ausführen"
        )


@pytest.fixture
def referenz() -> pd.DataFrame:
    """Eine künstliche Referenz mit bekannten Verteilungen."""
    return pd.DataFrame(
        {
            "type": ZUFALL.choice(["L", "M", "H"], size=500, p=[0.5, 0.3, 0.2]),
            "air_temperature_k": ZUFALL.normal(300.0, 2.0, size=500),
            "torque_nm": ZUFALL.normal(40.0, 10.0, size=500),
            drift.WAHRSCHEINLICHKEIT: ZUFALL.beta(1, 20, size=500),
        }
    )


# ── Das Maß selbst ──────────────────────────────────────────────────────


def test_psi_ist_null_bei_identischen_verteilungen() -> None:
    werte = ZUFALL.normal(0, 1, size=1000)
    assert drift.psi(werte, werte) == pytest.approx(0.0, abs=1e-9)


def test_psi_bleibt_klein_bei_zwei_stichproben_derselben_verteilung() -> None:
    """Zwei Stichproben derselben Quelle dürfen keinen Alarm auslösen."""
    a = ZUFALL.normal(0, 1, size=2000)
    b = ZUFALL.normal(0, 1, size=2000)
    assert drift.psi(a, b) < 0.10


def test_psi_schlaegt_bei_verschiebung_an() -> None:
    a = ZUFALL.normal(0, 1, size=2000)
    b = ZUFALL.normal(1.5, 1, size=2000)  # um 1,5 Standardabweichungen
    assert drift.psi(a, b) > 0.25


def test_psi_waechst_mit_der_staerke_der_verschiebung() -> None:
    a = ZUFALL.normal(0, 1, size=3000)
    schwach = drift.psi(a, a + 0.2)
    stark = drift.psi(a, a + 2.0)
    assert schwach < stark


def test_psi_ist_nie_negativ() -> None:
    for verschiebung in (-3.0, -0.5, 0.0, 0.5, 3.0):
        a = ZUFALL.normal(0, 1, size=500)
        assert drift.psi(a, a + verschiebung) >= 0.0


def test_psi_verkraftet_eine_spalte_mit_nur_einem_wert() -> None:
    """Viele gleiche Werte lassen Perzentilgrenzen zusammenfallen — kein Absturz."""
    konstant = np.zeros(200)
    assert drift.psi(konstant, konstant) == 0.0
    assert not np.isnan(drift.psi(konstant, np.ones(200)))


def test_psi_leere_eingabe_ergibt_nan() -> None:
    assert np.isnan(drift.psi(np.array([]), np.array([1.0, 2.0])))


# ── Die Tests zum Maß ───────────────────────────────────────────────────


def test_ks_erkennt_verschiebung_und_schweigt_sonst() -> None:
    a = pd.Series(ZUFALL.normal(0, 1, size=1000))
    gleich = pd.Series(ZUFALL.normal(0, 1, size=1000))
    verschoben = pd.Series(ZUFALL.normal(1.0, 1, size=1000))

    assert drift.ks(a, gleich)[1] > 0.05  # kein Unterschied
    assert drift.ks(a, verschoben)[1] < 0.05  # Unterschied


def test_chi2_erkennt_andere_kategorienverteilung() -> None:
    a = pd.Series(["L"] * 500 + ["M"] * 300 + ["H"] * 200)
    gleich = pd.Series(["L"] * 250 + ["M"] * 150 + ["H"] * 100)
    anders = pd.Series(["H"] * 800 + ["L"] * 200)

    assert drift.chi2(a, gleich)[1] > 0.05
    assert drift.chi2(a, anders)[1] < 0.05


def test_chi2_mit_einer_einzigen_kategorie_ist_harmlos() -> None:
    nur_l = pd.Series(["L"] * 100)
    assert drift.chi2(nur_l, nur_l) == (0.0, 1.0)


# ── Urteile ─────────────────────────────────────────────────────────────


def test_urteil_folgt_den_schranken_aus_params(params) -> None:
    warn = params["monitoring"]["psi_warn"]
    alarm = params["monitoring"]["psi_alarm"]

    assert drift.urteil(warn / 2, params) == "stabil"
    assert drift.urteil(warn, params) == "beobachten"
    assert drift.urteil((warn + alarm) / 2, params) == "beobachten"
    assert drift.urteil(alarm, params) == "handeln"
    assert drift.urteil(alarm * 10, params) == "handeln"


def test_urteil_bei_unbekanntem_wert() -> None:
    assert drift.urteil(float("nan")) == "unbekannt"


# ── Der Vergleich als Ganzes ────────────────────────────────────────────


def test_vergleich_mit_sich_selbst_ist_stabil(referenz, params) -> None:
    """Der Kontrollfall: eine Überwachung, die hier anschlägt, ist wertlos."""
    tabelle = drift.vergleiche(referenz, referenz, params)
    assert set(tabelle["Urteil"]) == {"stabil"}
    assert not tabelle["signifikant"].any()


def test_vergleich_erkennt_verschobene_spalte(referenz, params) -> None:
    aktuell = referenz.copy()
    aktuell["torque_nm"] = aktuell["torque_nm"] + 15.0

    tabelle = drift.vergleiche(referenz, aktuell, params).set_index("Merkmal")
    assert tabelle.loc["torque_nm", "Urteil"] == "handeln"
    assert tabelle.loc["torque_nm", "signifikant"]
    # Die übrigen Spalten dürfen davon unberührt bleiben.
    assert tabelle.loc["air_temperature_k", "Urteil"] == "stabil"


def test_wahrscheinlichkeit_ist_kein_merkmal(referenz, params) -> None:
    """Sie wird als Vorhersagedrift geführt, nicht in der Merkmalstabelle."""
    tabelle = drift.vergleiche(referenz, referenz, params)
    assert drift.WAHRSCHEINLICHKEIT not in set(tabelle["Merkmal"])


def test_vorhersagedrift_und_alarmquote(referenz, params) -> None:
    aktuell = referenz.copy()
    aktuell[drift.WAHRSCHEINLICHKEIT] = np.clip(aktuell[drift.WAHRSCHEINLICHKEIT] + 0.3, 0, 1)
    kennzahlen = drift.betriebskennzahlen(referenz, aktuell, params)

    assert kennzahlen["prediction_psi"] > params["monitoring"]["psi_alarm"]
    assert kennzahlen["prediction_verdict"] == "handeln"
    assert kennzahlen["alarm_rate_current"] > kennzahlen["alarm_rate_reference"]


def test_bericht_enthaelt_die_zugesagten_felder(referenz, params, tmp_path) -> None:
    eigene = dict(params)
    eigene["monitoring"] = {**params["monitoring"], "drift_report": str(tmp_path / "drift.json")}

    inhalt = drift.bericht(referenz, referenz, "test", eigene)
    for feld in (
        "checked_at",
        "source",
        "reference_rows",
        "current_rows",
        "verdict",
        "thresholds",
        "features",
        "operations",
    ):
        assert feld in inhalt
    assert inhalt["verdict"] == "stabil"
    assert (tmp_path / "drift.json").exists()


def test_zu_wenige_zeilen_fuehren_zu_keinem_urteil(referenz, params, tmp_path) -> None:
    """Bei 20 Zeilen schlägt jeder Test irgendwann zufällig an."""
    eigene = dict(params)
    eigene["monitoring"] = {**params["monitoring"], "drift_report": str(tmp_path / "drift.json")}

    inhalt = drift.bericht(referenz, referenz.head(20), "test", eigene)
    assert inhalt["enough_data"] is False
    assert inhalt["verdict"] == "zu wenige Daten"


# ── Referenzstichprobe und Simulation ───────────────────────────────────


def test_referenzstichprobe_ist_brauchbar(params) -> None:
    pfad = drift.referenzpfad(params)
    if not pfad.exists():
        pytest.skip("Keine Referenz: python -m src.pipelines.run_pipeline")

    referenz = drift.lade_referenz(params)
    assert len(referenz) >= params["monitoring"]["min_rows"]
    assert drift.WAHRSCHEINLICHKEIT in referenz.columns
    assert params["data"]["target"] in referenz.columns
    for spalte in ("type", "air_temperature_k", "torque_nm", "tool_wear_min"):
        assert spalte in referenz.columns
    # Ursachenspalten dürfen auch hier nicht auftauchen.
    assert not set(referenz.columns) & set(params["data"]["leakage_columns"])


def test_fehlende_referenz_meldet_sich_deutlich(params, tmp_path) -> None:
    eigene = dict(params)
    eigene["monitoring"] = {
        **params["monitoring"],
        "reference_sample": str(tmp_path / "gibt_es_nicht.csv"),
    }
    with pytest.raises(FileNotFoundError):
        drift.lade_referenz(eigene)


@pytest.mark.langsam
def test_simulation_erkennt_drift_und_meldet_keinen_fehlalarm(
    tmp_path, params, vollstaendige_umgebung
) -> None:
    """Die Abschlussbedingung des Leitfadens, als Test festgehalten."""
    from src.monitoring import simulate_drift

    eigene = dict(params)
    eigene["monitoring"] = {**params["monitoring"], "simulation_report": str(tmp_path / "sim.json")}
    inhalt = simulate_drift.fuehre_aus(eigene)
    nach_name = {s["scenario"]: s for s in inhalt["scenarios"]}

    assert nach_name["keine_drift"]["verdict"] == "stabil"
    for name in ("temperatur_kalibrierung", "verschleiss_hoch", "typverteilung"):
        assert nach_name[name]["verdict"] == "handeln", f"{name} nicht erkannt"


@pytest.mark.langsam
def test_der_lehrreiche_fall_ist_reproduzierbar(tmp_path, params, vollstaendige_umgebung) -> None:
    """Alarmquote praktisch unverändert, Recall bricht ein.

    Das ist die Aussage, auf der das Überwachungskonzept aufbaut. Ändert sich
    das Modell so, dass sie nicht mehr gilt, muss das Konzept angepasst werden —
    und dieser Test macht darauf aufmerksam.
    """
    from src.monitoring import simulate_drift

    if not drift.referenzpfad(params).exists():
        pytest.skip("Keine Referenz: python -m src.pipelines.run_pipeline")

    eigene = dict(params)
    eigene["monitoring"] = {**params["monitoring"], "simulation_report": str(tmp_path / "sim.json")}
    nach_name = {s["scenario"]: s for s in simulate_drift.fuehre_aus(eigene)["scenarios"]}
    grundfall = nach_name["keine_drift"]
    kalibrierung = nach_name["temperatur_kalibrierung"]

    # Betriebskennzahl wirkt ruhig ...
    assert (
        abs(kalibrierung["alarm_rate"] - grundfall["alarm_rate"]) <= 0.25 * grundfall["alarm_rate"]
    )
    # ... die Güte aber bricht ein ...
    assert kalibrierung["recall"] < grundfall["recall"] - 0.05
    assert kalibrierung["costs_eur"] > grundfall["costs_eur"]
    # ... und die Datendrift zeigt es an.
    assert kalibrierung["max_psi"] > params["monitoring"]["psi_alarm"]


@pytest.mark.langsam
def test_referenzstichprobe_ist_byteweise_reproduzierbar(
    params, vollstaendige_umgebung, tmp_path
) -> None:
    """Zweimal schreiben muss dieselbe Datei ergeben.

    Hintergrund: Der Wald rechnet mit ``n_jobs=-1``, und die Reihenfolge, in der
    die 150 Bäume aufsummiert werden, schwankt zwischen Läufen. Das verschiebt
    das Ergebnis um ein Maschinenepsilon und damit die letzte Stelle in der
    CSV-Datei. Eine Referenz, deren Hash sich ohne Grund ändert, ist als Maßstab
    wertlos — deshalb wird gerundet, und deshalb wird das hier geprüft.
    """
    import hashlib

    from src.monitoring.drift import referenzpfad, schreibe_referenz

    # Geschrieben wird in ein Wegwerfverzeichnis. Die Datei unter data/processed/
    # liegt im Git und ist der Maßstab der Driftprüfung — ein Test darf sie nicht
    # überschreiben. In Etappe 20 ist genau das aufgefallen: Nach einem Lauf mit
    # absichtlich kaputter Merkmalsformel stand dort eine falsch gerechnete
    # Referenz, und die nächste Driftprüfung meldete Drift, die es nicht gab.
    eigene = dict(params)
    eigene["monitoring"] = {
        **params["monitoring"],
        "reference_sample": str(tmp_path / "reference_sample.csv"),
    }
    pfad = referenzpfad(eigene)
    assert pfad != referenzpfad(params), "der Test darf die echte Referenz nicht anfassen"

    schreibe_referenz(eigene)
    erster = hashlib.sha256(pfad.read_bytes()).hexdigest()
    schreibe_referenz(eigene)
    zweiter = hashlib.sha256(pfad.read_bytes()).hexdigest()
    assert erster == zweiter
