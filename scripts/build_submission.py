
Das Archiv enthält mehr als das Git-Repository: Die **Rohdatei** kommt mit, damit
die Kette ohne Internetzugang läuft, und das **trainierte Modell**, damit die
Schnittstelle ohne vorherigen Trainingslauf startet. Beides liegt bewusst nicht
im Git (fremdes Material, erzeugtes Artefakt), gehört aber in eine Abgabe, die
jemand ohne Netz und ohne Vorwissen prüfen soll.

Nicht mitkommt, was nur auf diesem Rechner Sinn hat: virtuelle Umgebung,
Zwischenspeicher, Laufdatenbanken, Betriebsprotokolle, Git-Verlauf.

Am Ende prüft das Skript das fertige Archiv gegen eine Liste von Dateien, die
darin sein müssen, und gibt seinen SHA-256 aus — damit man belegen kann, welche
Fassung abgegeben wurde.

Aufruf::

    python scripts/build_submission.py
    python scripts/build_submission.py --ohne-modell
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

# Dieses Skript läuft auch ohne installiertes Paket: "python scripts/<datei>.py"
# legt nur das Verzeichnis scripts/ auf den Suchpfad, nicht die Projektwurzel —
# ein Import von "src" scheitert dann mit ModuleNotFoundError. In einem frischen
# Klon ist das Paket noch gar nicht installiert, und genau dann soll das Skript
# funktionieren. Deshalb die Wurzel ausdrücklich voranstellen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, PROJECT_ROOT, load_params
from src.data.load import datei_hash

#: Zielverzeichnis des Archivs. Liegt im Projekt, aber nicht im Git.
AUSGABE = PROJECT_ROOT / "dist"

#: Was zusätzlich zum Git-Inhalt mitkommt.
ZUSAETZLICH = ("data/raw/ai4i2020.csv",)
MIT_MODELL = ("models/model.joblib",)

#: Was auf keinen Fall mitkommt. Betriebsprotokolle sind Laufdaten dieses
#: Rechners und können personenbezogene oder nur verwirrende Spuren enthalten.
AUSGESCHLOSSEN = (
    "reports/api.log",
    "reports/predictions.jsonl",
    "reports/predictions.db",
)

#: Ohne diese Dateien ist die Abgabe unvollständig — geprüft wird im Archiv.
PFLICHT = (
    "README.md",
    "Makefile",
    "params.yaml",
    "pyproject.toml",
    "requirements.txt",
    "requirements-lock.txt",
    "requirements-api.txt",
    "Dockerfile.api",
    "docker-compose.yml",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    "references/datenbeschreibung.md",
    "monitoring/monitoring_concept.md",
    "src/config.py",
    "src/pipelines/run_pipeline.py",
    "src/serving/api.py",
    "scripts/fetch_data.py",
    "data/raw/ai4i2020.csv",
    "data/processed/reference_sample.csv",
    "reports/pipeline_run.json",
    "reports/final_test_evaluation.json",
)


def git_dateien() -> list[str]:
    """Alles, was ein ``git add -A`` aufnehmen würde.

    Also die versionierten Dateien plus die noch nicht eingecheckten, aber nicht
    ignorierten. Damit enthält das Archiv genau den Projektstand, auch wenn noch
    nicht alles festgeschrieben ist.
    """
    ergebnis = subprocess.run(
        ["git", "ls-files", "-c", "-o", "--exclude-standard"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [z for z in ergebnis.stdout.splitlines() if z]


def sammle(mit_modell: bool = True) -> list[str]:
    """Stellt die Dateiliste für das Archiv zusammen."""
    dateien = set(git_dateien())
    dateien |= set(ZUSAETZLICH)
    if mit_modell:
        dateien |= set(MIT_MODELL)
    dateien -= set(AUSGESCHLOSSEN)
    return sorted(d for d in dateien if (PROJECT_ROOT / d).is_file())


def baue(ziel: Path, dateien: list[str]) -> Path:
    """Schreibt das Archiv. Oberste Ebene ist ein Ordner mit dem Projektnamen."""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    wurzel = ziel.stem
    with zipfile.ZipFile(ziel, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archiv:
        for datei in dateien:
            archiv.write(PROJECT_ROOT / datei, f"{wurzel}/{datei}")
    return ziel


def pruefe(ziel: Path, erwarteter_daten_hash: str) -> list[str]:
    """Öffnet das fertige Archiv und prüft es. Gibt die Beanstandungen zurück."""
    maengel: list[str] = []
    wurzel = ziel.stem
    with zipfile.ZipFile(ziel) as archiv:
        enthalten = {n.split("/", 1)[1] for n in archiv.namelist() if "/" in n}

        for pflicht in PFLICHT:
            if pflicht not in enthalten:
                maengel.append(f"fehlt: {pflicht}")

        for verboten in AUSGESCHLOSSEN:
            if verboten in enthalten:
                maengel.append(f"darf nicht mit: {verboten}")

        for muster in (".venv/", ".git/", "mlruns/", "__pycache__/"):
            treffer = [n for n in enthalten if muster in n]
            if treffer:
                maengel.append(f"{len(treffer)} Datei(en) aus {muster} im Archiv")

        # Die Rohdatei im Archiv muss die eingefrorene sein.
        with archiv.open(f"{wurzel}/data/raw/ai4i2020.csv") as datei:
            digest = hashlib.sha256()
            for block in iter(lambda: datei.read(65_536), b""):
                digest.update(block)
        if digest.hexdigest() != erwarteter_daten_hash:
            maengel.append("der SHA-256 der Rohdatei im Archiv weicht ab")

        if archiv.testzip() is not None:
            maengel.append("das Archiv ist beschädigt")

    return maengel


def _main() -> None:
    zerleger = argparse.ArgumentParser(description="ZIP-Archiv für die Abgabe bauen")
    zerleger.add_argument(
        "--ohne-modell",
        action="store_true",
        help="models/model.joblib nicht mitpacken",
    )
    zerleger.add_argument("--name", default=None, help="abweichender Dateiname")
    argumente = zerleger.parse_args()

    params = load_params()
    dateien = sammle(mit_modell=not argumente.ohne_modell)
    name = argumente.name or f"ST_MLops_Abgabe_{date.today().isoformat()}"
    ziel = baue(AUSGABE / f"{name}.zip", dateien)

    print(f"{len(dateien)} Dateien gepackt")
    print(f"Archiv: {PATHS.relativ(ziel)}  ({ziel.stat().st_size / 1_048_576:.2f} MiB)")
    print()

    # Was steckt wo drin - gibt einen schnellen Überblick für die Abgabe.
    nach_ordner: dict[str, int] = {}
    for datei in dateien:
        oben = datei.split("/")[0] if "/" in datei else "(Wurzel)"
        nach_ordner[oben] = nach_ordner.get(oben, 0) + 1
    for ordner, anzahl in sorted(nach_ordner.items(), key=lambda p: -p[1]):
        print(f"  {anzahl:4d}  {ordner}")

    print()
    maengel = pruefe(ziel, params["data"]["raw_sha256"])
    if maengel:
        print("BEANSTANDUNGEN:")
        for mangel in maengel:
            print(f"  - {mangel}")
        sys.exit(1)

    print("Prüfung des Archivs: alle Pflichtdateien vorhanden, keine Laufdaten,")
    print("Rohdatei unverändert, Archiv unbeschädigt.")
    print(f"SHA-256 des Archivs: {datei_hash(ziel)}")


if __name__ == "__main__":
    _main()
