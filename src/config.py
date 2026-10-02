"""Zentrale Konfiguration: Pfade und Parameter an einer Stelle (Etappe 14).

Dieses Modul ist die **einzige** Stelle im Projekt, die weiß, wo die
Projektwurzel liegt und wie ``params.yaml`` gelesen wird. Alle anderen Module
fragen hier nach, statt sich den Pfad selbst zusammenzusetzen.

Zwei Dinge erledigt es:

1. **Pfade.** ``PATHS.processed`` statt ``Path("../data/processed")``. Der
   relative Pfad funktioniert im Notebook und bricht im Test, weil dort das
   Arbeitsverzeichnis ein anderes ist. Aus der Lage *dieser Datei* abgeleitete
   absolute Pfade funktionieren überall gleich — im Notebook, im Test, in der
   CI und im Container.

2. **Parameter.** ``load_params()`` liest ``params.yaml`` und hält das Ergebnis
   im Zwischenspeicher, damit nicht jeder Aufruf die Datei erneut einliest. Der
   Zwischenspeicher berücksichtigt den Änderungszeitpunkt: Wird ``params.yaml``
   während einer laufenden Jupyter-Sitzung bearbeitet, wird beim nächsten
   Aufruf die neue Fassung gelesen.

Jeder Aufruf erhält eine eigene Kopie. Das ist Absicht: Wer das Ergebnis
verändert — etwa ``vertrag["check_row_count"] = True`` —, soll damit nicht
unbemerkt die Konfiguration aller anderen verändern.

Geheimnisse (Zugangsdaten, Token) gehören **nicht** hierher und nicht in
``params.yaml``, sondern in eine ``.env``-Datei, die nicht im Git liegt. Die
Vorlage mit den Schlüsselnamen ist ``.env.example``.

Aufruf von der Kommandozeile — zeigt die aufgelöste Konfiguration::

    python -m src.config
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

#: Projektwurzel, aus der Lage dieser Datei abgeleitet.
#: src/config.py -> parents[0] = src, parents[1] = Wurzel.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Die zentrale Konfigurationsdatei. Über die Umgebungsvariable MLOPS_PARAMS
#: umstellbar — gedacht für Probeläufe und die CI, die gegen eine abweichende
#: Konfiguration rechnen sollen, ohne die echte Datei anzufassen.
PARAMS_PATH = Path(os.environ.get("MLOPS_PARAMS") or PROJECT_ROOT / "params.yaml")


@lru_cache(maxsize=8)
def _gelesen(pfad: str, _aenderungszeit: float) -> dict[str, Any]:
    """Liest eine YAML-Datei. Zwischengespeichert je Pfad und Änderungszeit.

    Der Änderungszeitpunkt steckt im Schlüssel des Zwischenspeichers, damit eine
    bearbeitete Datei nicht aus dem alten Speicher beantwortet wird.
    """
    with open(pfad, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_params(pfad: str | Path | None = None) -> dict[str, Any]:
    """Liest ``params.yaml`` und gibt eine veränderbare Kopie zurück.

    Args:
        pfad: abweichende Konfigurationsdatei, etwa in Tests.

    Raises:
        FileNotFoundError: wenn die Datei fehlt.
    """
    ziel = Path(pfad) if pfad is not None else PARAMS_PATH
    if not ziel.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {ziel}")
    return copy.deepcopy(_gelesen(str(ziel), ziel.stat().st_mtime))


@dataclass(frozen=True)
class Pfade:
    """Alle Ablageorte des Projekts als absolute Pfade.

    ``frozen=True``: Die Pfade stehen nach dem Erzeugen fest. Ein versehentliches
    ``PATHS.models = ...`` an irgendeiner Stelle im Code schlägt damit sofort
    fehl, statt still das Verhalten aller anderen Module zu ändern.
    """

    root: Path
    raw_file: Path
    processed: Path
    models: Path
    reports: Path
    figures: Path

    @property
    def raw(self) -> Path:
        """Verzeichnis der Rohdaten."""
        return self.raw_file.parent

    def ensure(self) -> None:
        """Legt alle Ausgabeverzeichnisse an, falls sie fehlen.

        Nicht angelegt wird ``raw``: Fehlende Rohdaten sind ein Fehler und
        sollen als solcher auffallen, nicht als leeres Verzeichnis durchgehen.
        """
        for verzeichnis in (self.processed, self.models, self.reports, self.figures):
            verzeichnis.mkdir(parents=True, exist_ok=True)

    def relativ(self, pfad: Path) -> str:
        """Pfad relativ zur Projektwurzel — für lesbare Ausgaben."""
        try:
            return str(pfad.resolve().relative_to(self.root))
        except ValueError:
            return str(pfad)


def absolut(relativer_pfad: str | Path) -> Path:
    """Löst einen Pfad aus ``params.yaml`` gegen die Projektwurzel auf."""
    pfad = Path(relativer_pfad)
    return pfad if pfad.is_absolute() else PROJECT_ROOT / pfad


def baue_pfade(params: dict[str, Any] | None = None) -> Pfade:
    """Baut die Pfade aus dem Abschnitt ``paths`` von params.yaml."""
    eintraege = (params or load_params())["paths"]
    return Pfade(
        root=PROJECT_ROOT,
        raw_file=absolut(eintraege["raw_file"]),
        processed=absolut(eintraege["processed_dir"]),
        models=absolut(eintraege["models_dir"]),
        reports=absolut(eintraege["reports_dir"]),
        figures=absolut(eintraege["figures_dir"]),
    )


#: Die Pfade des Projekts. Importieren statt selbst bauen.
PATHS = baue_pfade()


def mlflow_tracking_uri(params: dict[str, Any] | None = None) -> str:
    """Ablageort der MLflow-Laufdaten.

    Die Umgebungsvariable ``MLFLOW_TRACKING_URI`` hat Vorrang. Nötig in der CI
    und auf Netzlaufwerken, auf denen SQLite nicht sperren kann.
    """
    aus_umgebung = os.environ.get("MLFLOW_TRACKING_URI")
    if aus_umgebung:
        return aus_umgebung
    datei = absolut((params or load_params())["tracking"]["backend_store"])
    return f"sqlite:///{datei}"


def _main() -> None:
    """Zeigt, womit das Projekt gerade rechnet."""
    params = load_params()

    print("Projektwurzel:", PATHS.root)
    print("Konfiguration:", PATHS.relativ(PARAMS_PATH))
    print()
    print("Pfade")
    for name, pfad in (
        ("Rohdatei", PATHS.raw_file),
        ("Teilmengen", PATHS.processed),
        ("Modelle", PATHS.models),
        ("Berichte", PATHS.reports),
        ("Abbildungen", PATHS.figures),
    ):
        zustand = "vorhanden" if pfad.exists() else "fehlt"
        print(f"  {name:12s} {PATHS.relativ(pfad):24s} [{zustand}]")

    print()
    print("Eckwerte")
    print(f"  seed                  {params['seed']}")
    print(f"  Zielgröße             {params['data']['target']}")
    print(f"  Modell                {params['model']['name']}")
    print(f"  Entscheidungsschwelle {params['decision']['threshold']}")
    print(
        f"  Kosten FN/FP/TP       {params['costs']['false_negative_eur']} / "
        f"{params['costs']['false_positive_eur']} / "
        f"{params['costs']['true_positive_eur']} EUR"
    )
    print(f"  Schranke PR-AUC       >= {params['quality_gate']['min_pr_auc']}")
    print(f"  Schranke Recall       >= {params['quality_gate']['min_recall']}")
    print()
    print("MLflow:", mlflow_tracking_uri(params))


if __name__ == "__main__":
    _main()
