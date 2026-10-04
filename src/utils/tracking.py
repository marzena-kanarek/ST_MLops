"""Herkunftsangaben für jeden Trainingslauf.

Ein Lauf ist nur dann nachvollziehbar, wenn festgehalten wird, **womit** er
gerechnet hat: welche Daten, welcher Code, welche Bibliotheksversionen. Diese
Angaben werden bei jedem Lauf mitprotokolliert und landen ab Etappe 15 auch im
Pipeline-Manifest.
"""

from __future__ import annotations

import platform
import subprocess
import sys

from src.config import PROJECT_ROOT


def git_commit(kurz: bool = True) -> str:
    """Gibt den aktuellen Git-Commit zurück.

    Returns:
        Die Commit-Kennung, ergänzt um ``+dirty``, wenn es nicht festgeschriebene
        Änderungen gibt. ``"unbekannt"``, wenn kein Git-Repository vorliegt.
    """
    try:
        kennung = subprocess.run(
            ["git", "rev-parse", "--short" if kurz else "HEAD", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        schmutzig = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        return f"{kennung}+dirty" if schmutzig else kennung
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unbekannt"


def bibliotheksversionen() -> dict[str, str]:
    """Sammelt die Versionen der Bibliotheken, die das Ergebnis beeinflussen."""
    versionen = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "mlflow"):
        try:
            from importlib.metadata import version

            versionen[name] = version(name)
        except Exception:  # noqa: BLE001 - fehlendes Paket ist kein Fehler
            versionen[name] = "nicht installiert"
    return versionen


def herkunft(daten_hash: str, seed: int) -> dict[str, str | int]:
    """Stellt alle Herkunftsangaben eines Laufs zusammen.

    Args:
        daten_hash: SHA-256 der Rohdatei.
        seed: verwendeter Zufallszahl-Startwert.
    """
    return {
        "raw_data_sha256": daten_hash,
        "git_commit": git_commit(),
        "seed": seed,
        "python": platform.python_version(),
        "plattform": f"{platform.system()} {platform.machine()}",
        "ausfuehrbares_python": sys.executable,
    }
