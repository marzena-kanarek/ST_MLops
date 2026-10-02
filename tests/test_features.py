"""Merkmalstests: Stimmen die Formeln? (Etappe 13, Testart 2)

ML-Code hat die Tücke, auch dann durchzulaufen, wenn er falsch ist. Ein
vertauschtes Vorzeichen wirft keine Ausnahme, es verschlechtert nur die
Kennzahl — und das merkt niemand. Diese Tests rechnen die Formeln an Werten
nach, die man im Kopf prüfen kann.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features.build_features import (
    ABGELEITETE_MERKMALE,
    add_engineered_features,
)


def test_temperaturdifferenz_wird_korrekt_berechnet():
    frame = pd.DataFrame(
        {
            "air_temperature_k": [300.0],
            "process_temperature_k": [310.0],
            "rotational_speed_rpm": [1500],
            "torque_nm": [40.0],
            "tool_wear_min": [50],
        }
    )
    ergebnis = add_engineered_features(frame)
    assert ergebnis["temp_difference_k"].iloc[0] == pytest.approx(10.0)


def test_temperaturdifferenz_hat_das_richtige_vorzeichen():
    """Prozess minus Luft, nicht umgekehrt.

    Fängt genau den Fehler ab, der am häufigsten passiert und am wenigsten
    auffällt: Die Rechnung läuft, das Ergebnis ist negativ, das Modell wird
    schlechter.
    """
    frame = pd.DataFrame(
        {
            "air_temperature_k": [295.0],
            "process_temperature_k": [308.0],
            "rotational_speed_rpm": [1500],
            "torque_nm": [40.0],
            "tool_wear_min": [50],
        }
    )
    assert add_engineered_features(frame)["temp_difference_k"].iloc[0] > 0


def test_leistung_entspricht_der_physikalischen_formel():
    """P = M · 2πn / 60, nachgerechnet an einem glatten Beispiel."""
    drehmoment, drehzahl = 50.0, 1200
    frame = pd.DataFrame(
        {
            "air_temperature_k": [300.0],
            "process_temperature_k": [310.0],
            "rotational_speed_rpm": [drehzahl],
            "torque_nm": [drehmoment],
            "tool_wear_min": [50],
        }
    )
    erwartet = drehmoment * drehzahl * 2 * math.pi / 60
    assert add_engineered_features(frame)["power_w"].iloc[0] == pytest.approx(erwartet)


def test_leistung_steigt_mit_drehmoment():
    """Verhaltenstest: doppeltes Drehmoment, doppelte Leistung."""
    basis = pd.DataFrame(
        {
            "air_temperature_k": [300.0],
            "process_temperature_k": [310.0],
            "rotational_speed_rpm": [1500],
            "torque_nm": [30.0],
            "tool_wear_min": [50],
        }
    )
    doppelt = basis.copy()
    doppelt["torque_nm"] = 60.0

    leistung_basis = add_engineered_features(basis)["power_w"].iloc[0]
    leistung_doppelt = add_engineered_features(doppelt)["power_w"].iloc[0]
    assert leistung_doppelt == pytest.approx(2 * leistung_basis)


def test_verschleiss_unter_last_ist_ein_produkt():
    frame = pd.DataFrame(
        {
            "air_temperature_k": [300.0],
            "process_temperature_k": [310.0],
            "rotational_speed_rpm": [1500],
            "torque_nm": [40.0],
            "tool_wear_min": [200],
        }
    )
    assert add_engineered_features(frame)["wear_torque_min_nm"].iloc[0] == pytest.approx(8000.0)


def test_eingabe_wird_nicht_veraendert(beispiel_frame):
    """add_engineered_features arbeitet auf einer Kopie.

    Sonst hätte ein Aufruf Nebenwirkungen auf den DataFrame des Aufrufers — ein
    Fehler, der sich in langen Notebooks tagelang versteckt.
    """
    vorher = beispiel_frame.copy(deep=True)
    add_engineered_features(beispiel_frame)
    pd.testing.assert_frame_equal(beispiel_frame, vorher)


def test_alle_drei_merkmale_entstehen(beispiel_frame):
    ergebnis = add_engineered_features(beispiel_frame)
    for merkmal in ABGELEITETE_MERKMALE:
        assert merkmal in ergebnis.columns
    assert ergebnis.shape[1] == beispiel_frame.shape[1] + len(ABGELEITETE_MERKMALE)


def test_pipeline_rechnet_die_merkmale_selbst(trainiertes_kleinstmodell, basiszustand):
    """Die Abschlussbedingung aus Etappe 8.

    Die Pipeline bekommt nur die sechs Rohspalten. Wenn sie die abgeleiteten
    Merkmale nicht selbst erzeugte, würde predict_proba scheitern.
    """
    assert not set(ABGELEITETE_MERKMALE) & set(basiszustand.columns)
    wahrscheinlichkeit = trainiertes_kleinstmodell.predict_proba(basiszustand)
    assert wahrscheinlichkeit.shape == (1, 2)
