"""Tests der zentralen Konfiguration (Etappe 14).

Diese Tests sichern die Zusagen ab, die src/config.py macht: Pfade sind absolut
und liegen im Projekt, jeder Aufruf von ``load_params`` bekommt eine eigene
Kopie, und die in params.yaml genannten Abschnitte sind tatsächlich vorhanden.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config import (
    PARAMS_PATH,
    PATHS,
    PROJECT_ROOT,
    absolut,
    baue_pfade,
    load_params,
    mlflow_tracking_uri,
)


def test_projektwurzel_enthaelt_die_konfiguration() -> None:
    """Die Wurzel ist richtig bestimmt, wenn params.yaml dort liegt."""
    assert PARAMS_PATH.exists()
    assert PARAMS_PATH.parent == PROJECT_ROOT
    assert (PROJECT_ROOT / "src").is_dir()


def test_alle_pfade_sind_absolut_und_liegen_im_projekt() -> None:
    """Absolute Pfade sind der Grund für dieses Modul — relative brechen im Test."""
    for pfad in (PATHS.raw_file, PATHS.processed, PATHS.models, PATHS.reports, PATHS.figures):
        assert pfad.is_absolute(), f"{pfad} ist nicht absolut"
        assert PROJECT_ROOT in pfad.parents or pfad == PROJECT_ROOT


def test_abbildungen_liegen_unter_den_berichten() -> None:
    """reports/figures/ gehört in reports/ — sonst stimmt params.yaml nicht."""
    assert PATHS.reports in PATHS.figures.parents


def test_pfade_sind_unveraenderlich() -> None:
    """``frozen=True``: ein versehentliches Überschreiben schlägt sofort fehl."""
    with pytest.raises(FrozenInstanceError):
        PATHS.models = Path("/tmp")  # type: ignore[misc]


def test_load_params_gibt_jedes_mal_eine_eigene_kopie() -> None:
    """Wer das Ergebnis verändert, darf damit nicht die Konfiguration verändern.

    In validate.py wird genau das gemacht (``check_row_count = True``). Ohne
    eigene Kopie würde das für alle folgenden Aufrufe gelten.
    """
    erster = load_params()
    erster["data_contract"]["check_row_count"] = True
    erster["seed"] = 999

    zweiter = load_params()
    assert zweiter["seed"] == 42
    assert zweiter["data_contract"]["check_row_count"] is False


def test_erwartete_abschnitte_sind_vorhanden() -> None:
    """Alles, was der Code aus params.yaml liest, muss dort stehen."""
    params = load_params()
    for abschnitt in (
        "seed",
        "paths",
        "data",
        "data_contract",
        "model",
        "costs",
        "decision",
        "quality_gate",
        "comparison",
        "tuning",
        "tracking",
        "api",
    ):
        assert abschnitt in params, f"Abschnitt '{abschnitt}' fehlt in params.yaml"


def test_absolut_laesst_absolute_pfade_unberuehrt() -> None:
    """Ein bereits absoluter Pfad wird nicht an die Wurzel gehängt."""
    assert absolut("models") == PROJECT_ROOT / "models"
    assert absolut(Path("/tmp/x.csv")) == Path("/tmp/x.csv")


def test_fehlende_konfigurationsdatei_meldet_sich_deutlich(tmp_path: Path) -> None:
    """Keine stille Rückgabe von None, sondern ein lesbarer Fehler."""
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "gibt_es_nicht.yaml")


def test_pfade_folgen_der_konfiguration(tmp_path: Path) -> None:
    """Wird params.yaml geändert, ändern sich die Pfade mit — ohne Codeänderung."""
    eigene = tmp_path / "params.yaml"
    eigene.write_text(
        "paths:\n"
        "  raw_file: 'daten/roh.csv'\n"
        "  processed_dir: 'daten/fertig'\n"
        "  models_dir: 'ablage'\n"
        "  reports_dir: 'berichte'\n"
        "  figures_dir: 'berichte/bilder'\n",
        encoding="utf-8",
    )
    pfade = baue_pfade(load_params(eigene))
    assert pfade.models == PROJECT_ROOT / "ablage"
    assert pfade.raw == PROJECT_ROOT / "daten"


def test_tracking_uri_folgt_der_umgebungsvariable(monkeypatch) -> None:
    """Die Umgebungsvariable hat Vorrang — nötig in der CI."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///woanders.db")
    assert mlflow_tracking_uri() == "sqlite:///woanders.db"

    monkeypatch.delenv("MLFLOW_TRACKING_URI")
    standard = mlflow_tracking_uri()
    assert standard.startswith("sqlite:///")
    assert standard.endswith(load_params()["tracking"]["backend_store"])
