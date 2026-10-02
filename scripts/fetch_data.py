"""Rohdaten holen und gegen den eingefrorenen Hash prüfen.

Die Rohdatei liegt nicht im Git (``.gitignore``: ``data/raw/*``) — sie ist
fremdes Material unter CC-BY-Lizenz und gehört nicht in ein Codearchiv. Damit
die Kette trotzdem auf einem fremden Rechner anläuft, stellt dieses Skript sie
wieder her:

1. Liegt die Datei schon da und stimmt ihr SHA-256 mit ``params.yaml``
   überein, wird nichts getan.
2. Sonst wird das ZIP-Archiv vom UCI-Repository geladen (oder ein lokal
   vorhandenes verwendet), die CSV-Datei entpackt und nach ``data/raw/`` gelegt.
3. Anschließend wird der SHA-256 geprüft. **Weicht er ab, bricht das Skript
   ab** und löscht die heruntergeladene Datei nicht — man soll sie ansehen
   können. Ein abweichender Hash heißt: Die Quelle hat die Datei verändert, und
   keine bisherige Kennzahl ist mehr mit einer neuen vergleichbar.

Das ist derselbe Hash, den ``src/data/load.py`` bei jedem Laden prüft, und
derselbe, der im Pipeline-Manifest unter ``provenance.raw_data_sha256`` steht.

Aufrufe::

    python scripts/fetch_data.py                      # vom UCI-Repository laden
    python scripts/fetch_data.py --zip ~/Downloads/ai4i.zip   # lokales Archiv
    python scripts/fetch_data.py --force              # auch bei vorhandener Datei
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Dieses Skript läuft auch ohne installiertes Paket: "python scripts/<datei>.py"
# legt nur das Verzeichnis scripts/ auf den Suchpfad, nicht die Projektwurzel —
# ein Import von "src" scheitert dann mit ModuleNotFoundError. In einem frischen
# Klon ist das Paket noch gar nicht installiert, und genau dann soll das Skript
# funktionieren. Deshalb die Wurzel ausdrücklich voranstellen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PATHS, load_params
from src.data.load import datei_hash

#: Direktverweis auf das Archiv, siehe references/datenbeschreibung.md.
QUELLE = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)

#: Name der gesuchten Datei im Archiv.
IM_ARCHIV = "ai4i2020.csv"


def lade_archiv(ziel: Path, url: str = QUELLE) -> Path:
    """Lädt das ZIP-Archiv herunter.

    Raises:
        RuntimeError: wenn der Download scheitert — mit dem Hinweis auf den
            Weg über ``--zip``, falls der Rechner keinen Netzzugang hat.
    """
    print(f"lade {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as antwort, open(ziel, "wb") as datei:
            shutil.copyfileobj(antwort, datei)
    except Exception as fehler:  # noqa: BLE001 - jede Netzstörung endet hier gleich
        raise RuntimeError(
            f"Download fehlgeschlagen: {fehler}\n"
            "Archiv von Hand laden und mit --zip <Pfad> übergeben "
            "(URL siehe references/datenbeschreibung.md)."
        ) from fehler
    print(f"geladen: {ziel.stat().st_size:,} Byte".replace(",", "."))
    return ziel


def entpacke(archiv: Path, ziel: Path, name: str = IM_ARCHIV) -> Path:
    """Holt die CSV-Datei aus dem Archiv und legt sie unter ``ziel`` ab."""
    with zipfile.ZipFile(archiv) as zip_datei:
        enthalten = zip_datei.namelist()
        passend = [e for e in enthalten if Path(e).name.lower() == name.lower()]
        if not passend:
            raise RuntimeError(f"'{name}' ist nicht im Archiv. Enthalten sind: {enthalten}")
        ziel.parent.mkdir(parents=True, exist_ok=True)
        with zip_datei.open(passend[0]) as quelle, open(ziel, "wb") as datei:
            shutil.copyfileobj(quelle, datei)
    return ziel


def pruefe(pfad: Path, erwartet: str) -> str:
    """Vergleicht den Hash der Datei mit dem eingefrorenen Wert."""
    tatsaechlich = datei_hash(pfad)
    if tatsaechlich != erwartet:
        raise RuntimeError(
            "Der SHA-256 der geholten Datei weicht vom eingefrorenen Stand ab.\n"
            f"  Datei:     {pfad}\n"
            f"  erwartet:  {erwartet}\n"
            f"  berechnet: {tatsaechlich}\n"
            "Die Quelle hat die Datei verändert. Keine bisherige Kennzahl ist "
            "mit einer neuen vergleichbar — das muss geklärt werden, bevor "
            "weitergerechnet wird."
        )
    return tatsaechlich


def hole(
    zip_pfad: Path | None = None,
    erneut: bool = False,
    url: str = QUELLE,
) -> tuple[Path, bool]:
    """Stellt die Rohdatei her und prüft sie.

    Args:
        zip_pfad: lokales Archiv statt Download.
        erneut: auch holen, wenn die Datei schon stimmt.
        url: abweichende Bezugsquelle.

    Returns:
        (Pfad der Rohdatei, ob sie neu geholt wurde).
    """
    params = load_params()
    erwartet = params["data"]["raw_sha256"]
    ziel = PATHS.raw_file

    if ziel.exists() and not erneut:
        if datei_hash(ziel) == erwartet:
            print(f"vorhanden und unverändert: {PATHS.relativ(ziel)}")
            return ziel, False
        print(f"vorhanden, aber abweichender Hash: {PATHS.relativ(ziel)} — wird ersetzt")

    with tempfile.TemporaryDirectory() as verzeichnis:
        archiv = Path(zip_pfad) if zip_pfad else lade_archiv(Path(verzeichnis) / "ai4i.zip", url)
        if not archiv.exists():
            raise FileNotFoundError(f"Archiv nicht gefunden: {archiv}")
        entpacke(archiv, ziel)

    hash_wert = pruefe(ziel, erwartet)
    print(f"geschrieben: {PATHS.relativ(ziel)}")
    print(f"SHA-256 stimmt: {hash_wert}")
    return ziel, True


def _main() -> None:
    zerleger = argparse.ArgumentParser(
        description="Rohdaten holen und gegen den eingefrorenen Hash prüfen"
    )
    zerleger.add_argument("--zip", default=None, help="lokales ZIP-Archiv statt Download")
    zerleger.add_argument("--force", action="store_true", help="auch bei vorhandener Datei holen")
    zerleger.add_argument("--url", default=QUELLE, help="abweichende Bezugsquelle")
    argumente = zerleger.parse_args()

    try:
        hole(
            zip_pfad=Path(argumente.zip) if argumente.zip else None,
            erneut=argumente.force,
            url=argumente.url,
        )
    except (RuntimeError, FileNotFoundError) as fehler:
        print(f"\nFEHLER: {fehler}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
