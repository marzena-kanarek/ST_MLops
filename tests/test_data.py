"""Datentests: Hält der Datenvertrag? (Etappe 13, Testart 1)

Fast alle Produktionsfehler in ML-Systemen sind Datenfehler, keine
Modellfehler. Diese Tests sind die billigste Versicherung im ganzen Projekt.
"""

from __future__ import annotations

import pytest

from src.data.load import COLUMN_MAPPING, LEAKAGE_COLUMNS, TARGET_COLUMN
from src.data.validate import load_contract, validate_dataframe


def test_gueltiger_frame_besteht_den_vertrag(beispiel_frame):
    ergebnis = validate_dataframe(beispiel_frame)
    assert ergebnis.is_valid, ergebnis.report()


def test_fehlende_pflichtspalte_wird_erkannt(beispiel_frame):
    kaputt = beispiel_frame.drop(columns=["torque_nm"])
    ergebnis = validate_dataframe(kaputt)
    assert not ergebnis.is_valid
    assert any("torque_nm" in fehler for fehler in ergebnis.errors)


def test_physikalisch_unmoeglicher_wert_wird_erkannt(beispiel_frame):
    kaputt = beispiel_frame.copy()
    kaputt.loc[0, "torque_nm"] = -5.0  # negatives Drehmoment gibt es nicht
    ergebnis = validate_dataframe(kaputt)
    assert not ergebnis.is_valid
    assert any("torque_nm" in fehler for fehler in ergebnis.errors)


def test_unbekannte_kategorie_wird_erkannt(beispiel_frame):
    kaputt = beispiel_frame.copy()
    kaputt.loc[0, "type"] = "X"
    ergebnis = validate_dataframe(kaputt)
    assert not ergebnis.is_valid
    assert any("type" in fehler for fehler in ergebnis.errors)


def test_auffaelliger_wert_ist_nur_eine_warnung(beispiel_frame):
    """Ein Wert außerhalb des üblichen, aber physikalisch möglichen Bereichs
    darf den Lauf nicht abbrechen."""
    auffaellig = beispiel_frame.copy()
    auffaellig.loc[0, "tool_wear_min"] = 300  # möglich, aber ungewöhnlich
    ergebnis = validate_dataframe(auffaellig)
    assert ergebnis.is_valid, "Eine Warnung darf nicht als Fehler gelten"
    assert ergebnis.warnings


def test_raise_if_invalid_bricht_mit_lesbarer_meldung_ab(beispiel_frame):
    kaputt = beispiel_frame.drop(columns=["type"])
    with pytest.raises(ValueError, match="Datenvertrag verletzt"):
        validate_dataframe(kaputt).raise_if_invalid()


def test_zielspalte_ist_optional(beispiel_frame):
    """Anfragen an die Schnittstelle enthalten die Zielgröße nicht."""
    ohne_ziel = beispiel_frame.drop(columns=[TARGET_COLUMN])
    assert validate_dataframe(ohne_ziel).is_valid


def test_vertrag_und_spaltennamen_passen_zusammen():
    """Jede Spalte des Vertrags muss im Spaltenmapping vorkommen.

    Verhindert, dass eine Umbenennung in load.py den Vertrag ins Leere laufen
    lässt, ohne dass jemand es merkt.
    """
    vertragsspalten = set(load_contract()["columns"])
    technische_namen = set(COLUMN_MAPPING.values())
    assert vertragsspalten <= technische_namen


def test_leakage_spalten_sind_benannt():
    """Die Ursachenspalten müssen als solche hinterlegt sein."""
    assert set(LEAKAGE_COLUMNS) == {"twf", "hdf", "pwf", "osf", "rnf"}
