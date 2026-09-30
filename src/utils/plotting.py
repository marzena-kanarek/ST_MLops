"""Einheitliches Aussehen aller Abbildungen des Projekts.

Eine Stelle für Farben, Schriftgrößen und Gitterlinien — damit jede Abbildung
in Notebooks und Bericht gleich aussieht und nicht jedes Notebook seinen
eigenen Stil erfindet.

Die zwei Kategorienfarben sind auf Unterscheidbarkeit geprüft, auch bei
Rot-Grün- und Blau-Gelb-Sehschwäche.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

#: Kategorienfarben in fester Reihenfolge — nie rotieren, nie neu zuweisen.
FARBEN: dict[str, str] = {
    "kein_ausfall": "#2E5FA3",   # Blau
    "ausfall": "#D1690E",        # Orange
    "neutral": "#8C9BAA",
    "text": "#1A1A1A",
    "text_sekundaer": "#5B6B7B",
    "gitter": "#E3E8EE",
}

#: Beschriftungen der Zielklassen.
KLASSEN: dict[int, str] = {0: "kein Ausfall", 1: "Ausfall"}

#: Divergierende Farbskala für Korrelationen: Orange (negativ) -> Grau -> Blau (positiv).
CMAP_KORRELATION = LinearSegmentedColormap.from_list(
    "korrelation", [FARBEN["ausfall"], "#EDEDED", FARBEN["kein_ausfall"]]
)

#: Sequenzielle Skala für Größenordnungen: ein Farbton, hell nach dunkel.
CMAP_SEQUENZIELL = LinearSegmentedColormap.from_list(
    "sequenziell", ["#EAF0F7", FARBEN["kein_ausfall"], "#15304F"]
)


def setze_stil() -> None:
    """Setzt den Projektstil für alle folgenden Abbildungen."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.labelsize": 9.5,
            "axes.labelcolor": FARBEN["text_sekundaer"],
            "text.color": FARBEN["text"],
            "axes.edgecolor": FARBEN["gitter"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,          # Gitter hinter die Marken
            "grid.color": FARBEN["gitter"],
            "grid.linewidth": 0.7,
            "xtick.color": FARBEN["text_sekundaer"],
            "ytick.color": FARBEN["text_sekundaer"],
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
        }
    )


def nur_y_gitter(ax: plt.Axes) -> None:
    """Gitter nur waagerecht — bei Balken stört das senkrechte Gitter."""
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)


def nur_x_gitter(ax: plt.Axes) -> None:
    """Gitter nur senkrecht — für liegende Balken."""
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)


def speichere(fig: plt.Figure, name: str) -> Path:
    """Legt eine Abbildung unter reports/figures/<name>.png ab.

    Args:
        fig: die Abbildung.
        name: Dateiname ohne Endung, z. B. "01_klassenverhaeltnis".

    Returns:
        Der Pfad der geschriebenen Datei.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pfad = FIGURES_DIR / f"{name}.png"
    fig.savefig(pfad)
    return pfad
