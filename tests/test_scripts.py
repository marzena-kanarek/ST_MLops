"""Tests der Skripte unter scripts/.

Die Skripte sind das, was jemand als Erstes ausführt — beim Wiederherstellen der
Rohdaten, beim Prüfen der Notebooks, beim Bauen der Abgabe. Sie müssen deshalb
auch dann laufen, wenn das Projekt **nicht** als Paket installiert ist: In einem
frischen Klon ist es das nicht, und `python scripts/<datei>.py` legt nur das
Verzeichnis `scripts/` auf den Suchpfad, nicht die Projektwurzel.

Genau daran ist der Aufruf einmal gescheitert (`ModuleNotFoundError: No module
named 'src'`). Diese Tests halten die Korrektur fest.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.config import PROJECT_ROOT

SKRIPTE = sorted((PROJECT_ROOT / "scripts").glob("*.py"))


def _nutzt_src(pfad: Path) -> bool:
    baum = ast.parse(pfad.read_text(encoding="utf-8"))
    return any(
        isinstance(k, ast.ImportFrom) and (k.module or "").split(".")[0] == "src"
        for k in ast.walk(baum)
    )


def test_es_gibt_skripte() -> None:
    assert SKRIPTE, "keine Skripte unter scripts/ gefunden"


@pytest.mark.parametrize("pfad", SKRIPTE, ids=lambda p: p.name)
def test_wurzel_wird_vor_dem_src_import_gesetzt(pfad: Path) -> None:
    """Der Suchpfad muss gesetzt sein, **bevor** `src` importiert wird."""
    if not _nutzt_src(pfad):
        pytest.skip(f"{pfad.name} importiert kein src")

    quelle = pfad.read_text(encoding="utf-8")
    assert "sys.path.insert" in quelle, (
        f"{pfad.name} verlässt sich auf ein installiertes Paket — "
        "in einem frischen Klon bricht es ab"
    )
    assert quelle.index("sys.path.insert") < quelle.index("from src.")


@pytest.mark.parametrize("pfad", SKRIPTE, ids=lambda p: p.name)
def test_skript_startet_ohne_installiertes_paket(pfad: Path, tmp_path) -> None:
    """Aus einem fremden Arbeitsverzeichnis und ohne PYTHONPATH gestartet.

    Geprüft wird nicht der Rückgabewert — `check_notebooks.py` endet
    absichtlich mit 1, solange Notebooks offen sind — sondern dass die Importe
    durchlaufen.
    """
    umgebung = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    ergebnis = subprocess.run(
        [sys.executable, str(pfad), "--help"],
        cwd=tmp_path,
        env=umgebung,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "ModuleNotFoundError" not in ergebnis.stderr, ergebnis.stderr[-400:]
    assert "Traceback" not in ergebnis.stderr, ergebnis.stderr[-400:]
