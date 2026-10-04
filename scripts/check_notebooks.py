
Dieses Skript prüft nur, es verändert nichts. Rückgabewert 1, wenn etwas zu tun
ist — damit es sich auch in eine Prüfkette einbauen lässt.

Aufruf::

    python scripts/check_notebooks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Dieses Skript läuft auch ohne installiertes Paket: "python scripts/<datei>.py"
# legt nur das Verzeichnis scripts/ auf den Suchpfad, nicht die Projektwurzel —
# ein Import von "src" scheitert dann mit ModuleNotFoundError. In einem frischen
# Klon ist das Paket noch gar nicht installiert, und genau dann soll das Skript
# funktionieren. Deshalb die Wurzel ausdrücklich voranstellen.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT

NOTEBOOKS = PROJECT_ROOT / "notebooks"


def pruefe(pfad: Path) -> dict[str, object]:
    """Liest ein Notebook und beurteilt seinen Zustand."""
    nb = json.loads(pfad.read_text(encoding="utf-8"))
    code = [z for z in nb["cells"] if z["cell_type"] == "code"]
    mit_inhalt = [z for z in code if "".join(z["source"]).strip()]
    nummern = [z.get("execution_count") for z in mit_inhalt]
    ausgefuehrt = [n for n in nummern if n is not None]

    in_folge = ausgefuehrt == list(range(1, len(ausgefuehrt) + 1))
    ohne_ausgabe = len(mit_inhalt) - len(ausgefuehrt)

    maengel = []
    if not ausgefuehrt:
        maengel.append("keine einzige Zelle ausgeführt — keine Belege im Notebook")
    else:
        if ohne_ausgabe:
            maengel.append(f"{ohne_ausgabe} Zelle(n) ohne Ausgabe")
        if not in_folge:
            maengel.append("Ausführungsnummern nicht in Folge (Restart & Run All)")

    return {
        "name": pfad.name,
        "zellen": len(mit_inhalt),
        "ausgefuehrt": len(ausgefuehrt),
        "maengel": maengel,
    }


def _main() -> None:
    if not NOTEBOOKS.is_dir():
        print(f"Kein Notebook-Verzeichnis unter {NOTEBOOKS}")
        sys.exit(1)

    ergebnisse = [pruefe(p) for p in sorted(NOTEBOOKS.glob("*.ipynb"))]

    kopf = f"{'Notebook':34s} {'Zellen':>7s} {'ausgeführt':>11s}  Befund"
    print(kopf)
    print("─" * (len(kopf) + 30))
    for e in ergebnisse:
        print(
            f"{e['name']:34s} {e['zellen']:>7d} {e['ausgefuehrt']:>11d}  "
            f"{'; '.join(e['maengel']) or 'in Ordnung'}"
        )

    betroffen = [e for e in ergebnisse if e["maengel"]]
    print()
    if not betroffen:
        print("Alle Notebooks sind in einem Durchlauf entstanden und enthalten ihre Ergebnisse.")
        return

    print(f"{len(betroffen)} von {len(ergebnisse)} Notebooks brauchen einen Durchlauf:")
    print("  In VS Code: Kernel wählen, dann 'Restart' und 'Run All', danach speichern.")
    print("  Hinweis: Notebook 05 rechnet eine Zufallssuche (einige Minuten),")
    print("  07 und 08 schreiben in MLflow — beide brauchen die lokale Umgebung.")
    sys.exit(1)


if __name__ == "__main__":
    _main()
