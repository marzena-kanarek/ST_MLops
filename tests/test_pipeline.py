"""Tests der Kette als Ganzes (Etappe 15).

Der vollständige Lauf wird hier **nicht** ausgeführt — er dauert zu lange für
eine Testsuite, die bei jeder Änderung durchlaufen soll. Geprüft wird, was ohne
Lauf prüfbar ist: dass die Schrittliste und die Funktionen zusammenpassen, dass
die Zufallsquellen wirklich festgelegt werden, und dass ein vorhandenes Manifest
die zugesagten Felder enthält.
"""

from __future__ import annotations

import json
import random
from datetime import datetime

import numpy as np
import pytest

from src.config import PARAMS_PATH, PATHS
from src.pipelines.run_pipeline import (
    FUNKTIONEN,
    MANIFEST_NAME,
    SCHRITTE,
    jetzt,
    set_seeds,
    text_hash,
)


def test_jeder_schritt_hat_eine_funktion() -> None:
    """Ein Name in SCHRITTE ohne Funktion würde erst mitten im Lauf auffallen."""
    assert set(SCHRITTE) == set(FUNKTIONEN)
    assert len(SCHRITTE) == len(FUNKTIONEN)


def test_reihenfolge_der_schritte_ist_festgelegt() -> None:
    """Prüfen vor Aufteilen, Aufteilen vor Trainieren, Trainieren vor Bewerten."""
    assert SCHRITTE.index("daten") < SCHRITTE.index("pruefen")
    assert SCHRITTE.index("pruefen") < SCHRITTE.index("aufteilen")
    assert SCHRITTE.index("aufteilen") < SCHRITTE.index("trainieren")
    assert SCHRITTE.index("trainieren") < SCHRITTE.index("bewerten")


def test_set_seeds_macht_zufall_wiederholbar() -> None:
    """Zwei Läufe mit demselben Startwert müssen dieselben Zahlen liefern."""
    set_seeds(42)
    erste = (random.random(), float(np.random.rand()))
    set_seeds(42)
    zweite = (random.random(), float(np.random.rand()))
    assert erste == zweite


def test_set_seeds_setzt_den_hash_startwert_fuer_unterprozesse() -> None:
    """PYTHONHASHSEED wirkt nur auf neu gestartete Prozesse — gesetzt wird er trotzdem."""
    import os

    set_seeds(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_zeitstempel_ist_nach_iso_8601_mit_zeitzone() -> None:
    """Ein Zeitstempel ohne Zeitzone ist im Manifest wertlos."""
    stempel = jetzt()
    geparst = datetime.fromisoformat(stempel)
    assert geparst.tzinfo is not None


def test_params_hash_entspricht_der_datei() -> None:
    """Der params_sha256 im Manifest muss die tatsächliche Datei beschreiben."""
    import hashlib

    assert text_hash(PARAMS_PATH) == hashlib.sha256(PARAMS_PATH.read_bytes()).hexdigest()


def test_manifest_enthaelt_die_zugesagten_felder() -> None:
    """Prüft ein vorhandenes Manifest gegen die in der Dokumentation zugesagte Form."""
    pfad = PATHS.reports / MANIFEST_NAME
    if not pfad.exists():
        pytest.skip("Noch kein Lauf vorhanden: python -m src.pipelines.run_pipeline")

    manifest = json.loads(pfad.read_text(encoding="utf-8"))
    for feld in (
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "provenance",
        "environment",
        "steps",
    ):
        assert feld in manifest, f"Feld '{feld}' fehlt im Manifest"

    for feld in ("raw_data_sha256", "params_sha256", "git_commit", "seed"):
        assert feld in manifest["provenance"], f"provenance.{feld} fehlt"

    assert set(manifest["steps"]) <= set(SCHRITTE)
    if manifest["status"] == "ok":
        assert manifest["model_sha256"]
        assert manifest["quality_gate"]["passed"] is True
