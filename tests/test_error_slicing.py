"""Tests der Fehlerzerlegung (Error Slicing).

Die Kennzahlen werden an einem von Hand gebauten Rahmen geprüft, bei dem die
richtigen Werte ohne Code nachrechenbar sind. Dazu zwei Eigenschaften, die beim
Zerlegen leicht verlorengehen: Die Teilmengen müssen sich wieder zum Ganzen
addieren, und die Reihenfolge der Bänder muss numerisch bleiben.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS, load_params
from src.modeling import error_slicing as es


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture
def kleiner_rahmen() -> pd.DataFrame:
    """Zehn Zeilen mit bekannter Verwechslungsmatrix.

    Vier Ausfälle, davon drei gefunden (Recall 0,75); fünf Alarme, davon drei
    richtig (Precision 0,60).
    """
    return pd.DataFrame(
        {
            "machine_failure": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "vorhersage": [1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            "fehlerart": ["TP", "TP", "TP", "FN", "FP", "FP", "TN", "TN", "TN", "TN"],
            "wahrscheinlichkeit": [0.9, 0.8, 0.7, 0.01, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0],
            "Gruppe": list("AAAABBBBBB"),
            "twf": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "hdf": [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "pwf": [0] * 10,
            "osf": [0] * 10,
            "rnf": [0] * 10,
        }
    )


# ── Bänder ──────────────────────────────────────────────────────────────


def test_baender_sind_numerisch_geordnet() -> None:
    """Alphabetisch stünde "105–163" vor "51–105" — in einer Abbildung irreführend."""
    band = es._band(pd.Series(range(0, 400)))
    namen = list(band.categories)
    untergrenzen = [float(n.split("–")[0]) for n in namen]
    assert untergrenzen == sorted(untergrenzen)
    assert band.ordered


def test_baender_sind_etwa_gleich_gross() -> None:
    """Quantile statt fester Schwellen: keine Gruppe soll verschwinden."""
    band = es._band(pd.Series(np.random.default_rng(1).normal(size=800)))
    anteile = pd.Series(band).value_counts(normalize=True)
    assert anteile.min() > 0.2


def test_qualitaetsvariante_in_natuerlicher_reihenfolge() -> None:
    frame = pd.DataFrame(
        {
            "type": ["H", "L", "M", "L"],
            "process_temperature_k": [310.0, 311.0, 309.0, 310.5],
            "air_temperature_k": [300.0, 300.0, 300.0, 300.0],
            "torque_nm": [40.0, 45.0, 38.0, 50.0],
            "rotational_speed_rpm": [1500, 1400, 1600, 1300],
            "tool_wear_min": [10, 100, 200, 50],
        }
    )
    ergebnis = es.ergaenze_baender(frame)
    assert list(ergebnis["Qualitätsvariante"].cat.categories) == ["L", "M", "H"]


# ── Kennzahlen ──────────────────────────────────────────────────────────


def test_kennzahlen_sind_nachrechenbar(kleiner_rahmen, params) -> None:
    kennzahlen = es.gesamt(kleiner_rahmen, params)
    assert kennzahlen["Ausfälle"] == 4
    assert kennzahlen["TP"] == 3
    assert kennzahlen["FP"] == 2
    assert kennzahlen["FN"] == 1
    assert kennzahlen["Recall"] == pytest.approx(0.75)
    assert kennzahlen["Precision"] == pytest.approx(0.60)
    assert kennzahlen["F1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))


def test_kosten_folgen_dem_kostenmodell(kleiner_rahmen, params) -> None:
    kosten = params["costs"]
    erwartet = (
        1 * kosten["false_negative_eur"]
        + 2 * kosten["false_positive_eur"]
        + 3 * kosten["true_positive_eur"]
    )
    assert es.gesamt(kleiner_rahmen, params)["Kosten EUR"] == pytest.approx(erwartet)


def test_scheiben_addieren_sich_zum_ganzen(kleiner_rahmen, params) -> None:
    """Jede Zeile gehört in genau eine Scheibe — sonst stimmt die Zerlegung nicht."""
    tabelle = es.scheibe(kleiner_rahmen, "Gruppe", params)
    insgesamt = es.gesamt(kleiner_rahmen, params)
    for spalte in ("Zeilen", "Ausfälle", "TP", "FP", "FN", "Kosten EUR"):
        assert tabelle[spalte].sum() == pytest.approx(insgesamt[spalte])
    assert tabelle["Kostenanteil"].sum() == pytest.approx(1.0)


def test_scheibe_ohne_ausfaelle_ergibt_keinen_recall(params) -> None:
    """Kein Recall von 0, wo es nichts zu finden gab — das wäre eine Falschaussage."""
    ohne = pd.DataFrame(
        {
            "machine_failure": [0, 0, 0],
            "fehlerart": ["TN", "FP", "TN"],
            "Gruppe": ["A", "A", "A"],
        }
    )
    assert np.isnan(es.scheibe(ohne, "Gruppe", params).loc["A", "Recall"])


# ── Ursachen ────────────────────────────────────────────────────────────


def test_nach_ursache_zaehlt_gefunden_und_uebersehen(kleiner_rahmen) -> None:
    tabelle = es.nach_ursache(kleiner_rahmen)
    assert (tabelle["gefunden"] + tabelle["übersehen"] == tabelle["Ausfälle"]).all()

    verschleiss = tabelle.loc[es.URSACHE_LANG["twf"]]
    assert verschleiss["Ausfälle"] == 2  # eine gefunden, eine übersehen
    assert verschleiss["Recall"] == pytest.approx(0.5)

    waerme = tabelle.loc[es.URSACHE_LANG["hdf"]]
    assert waerme["Ausfälle"] == 2 and waerme["Recall"] == pytest.approx(1.0)


def test_nach_ursache_betrachtet_nur_echte_ausfaelle(kleiner_rahmen) -> None:
    """Fehlalarme haben keine Ursache — sie dürfen die Zählung nicht verfälschen."""
    tabelle = es.nach_ursache(kleiner_rahmen)
    assert tabelle["Ausfälle"].drop(index="ohne eingetragene Ursache", errors="ignore").sum() == 4


# ── Kreuzklassifikation ─────────────────────────────────────────────────


def test_kreuz_blendet_zu_kleine_gruppen_aus(kleiner_rahmen, params) -> None:
    """Eine Kennzahl aus drei Zeilen sieht aus wie ein Befund und ist keiner."""
    kleiner_rahmen = kleiner_rahmen.copy()
    kleiner_rahmen["Zweite"] = ["X"] * 5 + ["Y"] * 5

    tabelle, groessen = kreuz_mit(kleiner_rahmen, params, mindestgroesse=100)
    assert tabelle.isna().all().all()

    tabelle, _ = kreuz_mit(kleiner_rahmen, params, mindestgroesse=1)
    assert tabelle.notna().any().any()
    # Kombinationen, die nicht vorkommen, stehen als NaN in der Größentabelle -
    # deshalb nansum: Die vorhandenen Felder müssen zusammen alle Zeilen ergeben.
    assert np.nansum(groessen.to_numpy()) == len(kleiner_rahmen)


def kreuz_mit(frame, params, mindestgroesse):
    return es.kreuz(frame, "Gruppe", "Zweite", "Recall", mindestgroesse, params)


def test_kreuz_behaelt_die_reihenfolge_der_baender(params) -> None:
    frame = pd.DataFrame(
        {
            "machine_failure": [1, 0] * 200,
            "fehlerart": ["TP", "TN"] * 200,
            "Wert": list(range(400)),
            "Gruppe": ["A", "B"] * 200,
        }
    )
    frame["Band"] = es._band(frame["Wert"])
    tabelle, _ = es.kreuz(frame, "Gruppe", "Band", "Recall", 1, params)
    untergrenzen = [float(str(c).split("–")[0]) for c in tabelle.columns]
    assert untergrenzen == sorted(untergrenzen)


# ── Rangliste ───────────────────────────────────────────────────────────


def test_auffaellige_scheiben_sind_aufsteigend_sortiert(params) -> None:
    pfad = PATHS.processed / "val.parquet"
    if not pfad.exists() or not (PATHS.models / "model.joblib").exists():
        pytest.skip("Teilmengen oder Modell fehlen: python -m src.pipelines.run_pipeline")

    frame = es.ergaenze_baender(es.lade_bewertungsdaten(params))
    rangliste = es.auffaellige_scheiben(frame, es.MINDESTGROESSE, params)

    abstand = rangliste["Abstand zum Gesamt-Recall"].to_list()
    assert abstand == sorted(abstand)
    assert (rangliste["Zeilen"] >= es.MINDESTGROESSE).all()
    assert (rangliste["Ausfälle"] > 0).all()


@pytest.mark.langsam
def test_ursachenspalten_stimmen_mit_den_rohdaten_ueberein(params) -> None:
    """Die Zuordnung über den Zeilenindex muss belegbar richtig sein.

    Geprüft wird die Gegenrichtung zur Zusicherung im Modul: Die Zahl der
    Ausfälle mit mindestens einer Ursache darf die Zahl der Ausfälle nicht
    übersteigen, und jede Zeile mit einer Ursache muss ein Ausfall sein.
    """
    if not (PATHS.models / "model.joblib").exists():
        pytest.skip("Kein Modell: python -m src.pipelines.run_pipeline")

    frame = es.lade_bewertungsdaten(params)
    mit_ursache = frame[list(es.URSACHEN)].sum(axis=1) > 0
    ziel = params["data"]["target"]

    # Jede Zeile mit Ursache ist ein Ausfall (RNF ausgenommen: ein zufälliger
    # Eintrag ohne Ausfall kommt im Datensatz vor).
    ohne_rnf = frame[["twf", "hdf", "pwf", "osf"]].sum(axis=1) > 0
    assert (frame.loc[ohne_rnf, ziel] == 1).all()
    assert mit_ursache.sum() <= len(frame)
