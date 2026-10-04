"""Das geladene Modell als Dienst .

Dieses Modul trennt zwei Dinge, die gern vermischt werden: **das Modell laden
und befragen** (hier) und **HTTP sprechen** (in ``api.py``). Die Trennung hat
einen praktischen Nutzen — der Dienst ist ohne laufenden Webserver testbar.

Geladen wird **einmal**, beim Start des Dienstes. Ein ``joblib.load`` in der
Endpunktfunktion würde bei jeder Anfrage die Modelldatei von der Platte lesen;
die Antwortzeit stiege von wenigen Millisekunden auf Hunderte.

Was geladen wird, ist die vollständige Pipeline aus ``model.joblib``: Sie
enthält die Merkmalskonstruktion, die Vorverarbeitung und das Modell. Die
Schnittstelle rechnet deshalb **keine** Merkmale selbst — genau das verhindert
Training-Serving-Skew.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import PATHS, absolut, load_params
from src.data.load import datei_hash
from src.data.validate import load_contract, validate_dataframe
from src.features.build_features import ABGELEITETE_MERKMALE

#: Die Felder, die eine Anfrage mitbringen muss — in der Reihenfolge, in der die
#: Pipeline sie beim Training gesehen hat.
EINGABEFELDER: tuple[str, ...] = (
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)


class ModellNichtGeladen(RuntimeError):
    """Wird geworfen, wenn der Dienst ohne Modell befragt wird."""


class ModellDienst:
    """Hält die geladene Pipeline und beantwortet Anfragen.

    Attribute:
        pipeline: die vollständige sklearn-Pipeline.
        schwelle: der Entscheidungsschwellenwert, der **mit dem Modell
            gespeichert** wurde — nicht der aus params.yaml. Ein Modell und seine
            Entscheidungsregel gehören zusammen; sonst beantwortet ein altes
            Modell Anfragen mit einer neuen Schwelle.
    """

    def __init__(self, pfad: Path, inhalt: dict[str, Any]) -> None:
        self.pfad = pfad
        self.pipeline = inhalt["pipeline"]
        self.schwelle: float = float(inhalt["schwelle"])
        self.herkunft: dict[str, Any] = inhalt.get("herkunft", {})
        self.kennzahlen: dict[str, Any] = inhalt.get("kennzahlen", {})
        self.sha256 = datei_hash(pfad)
        self.geladen_am = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._params = load_params()
        self._vertrag = load_contract()

    # ── Vorhersage ──────────────────────────────────────────────────────

    def rahmen(self, zustaende: list[dict[str, Any]]) -> pd.DataFrame:
        """Baut aus Anfragen einen DataFrame in der erwarteten Spaltenfolge."""
        return pd.DataFrame(zustaende)[list(EINGABEFELDER)]

    def wahrscheinlichkeiten(self, frame: pd.DataFrame) -> list[float]:
        """Ausfallwahrscheinlichkeit je Zeile."""
        return [float(p) for p in self.pipeline.predict_proba(frame)[:, 1]]

    def alarm(self, wahrscheinlichkeit: float) -> bool:
        """Die Entscheidung: Alarm ab dem gespeicherten Schwellenwert."""
        return wahrscheinlichkeit >= self.schwelle

    def hinweise(self, frame: pd.DataFrame) -> list[str]:
        """Weiche Grenzen des Datenvertrags als Hinweise.

        Harte Grenzen hat Pydantic schon abgelehnt, bevor diese Zeile läuft. Hier
        geht es nur um das „ungewöhnlich, aber möglich" — etwa ein Drehmoment
        oberhalb von allem, was in den Trainingsdaten vorkam. Das ist genau der
        Fall, in dem man einer Vorhersage weniger trauen sollte.
        """
        ergebnis = validate_dataframe(frame, self._vertrag)
        return list(ergebnis.warnings)

    # ── Angaben über das Modell ─────────────────────────────────────────

    def modellangabe(self) -> dict[str, Any]:
        """Die kurze Fassung für jede Antwort."""
        return {
            "name": self._params["model"]["name"],
            "sha256": self.sha256,
            "git_commit": str(self.herkunft.get("git_commit", "unbekannt")),
            "raw_data_sha256": str(self.herkunft.get("raw_data_sha256", "unbekannt")),
            "trained_pr_auc": round(float(self.kennzahlen.get("pr_auc", 0.0)), 4),
        }

    def auskunft(self) -> dict[str, Any]:
        """Die ausführliche Fassung für ``/model-info``."""
        return {
            "name": self._params["model"]["name"],
            "registry_name": self._params["model"]["registry_name"],
            "threshold": self.schwelle,
            "model_file": PATHS.relativ(self.pfad),
            "model_sha256": self.sha256,
            "loaded_at": self.geladen_am,
            "features": list(EINGABEFELDER),
            "engineered_features": list(ABGELEITETE_MERKMALE),
            "metrics": {k: float(v) for k, v in self.kennzahlen.items()},
            "provenance": self.herkunft,
            "costs": self._params["costs"],
        }


def modelldatei(params: dict[str, Any] | None = None) -> Path:
    """Pfad der Modelldatei aus params.yaml (``api.model_file``)."""
    return absolut((params or load_params())["api"]["model_file"])


def lade_modell(pfad: Path | None = None) -> ModellDienst:
    """Lädt die Modelldatei und gibt den Dienst zurück.

    Raises:
        FileNotFoundError: wenn keine Modelldatei vorliegt. Dann zuerst
            ``python -m src.pipelines.run_pipeline`` ausführen.
    """
    pfad = pfad or modelldatei()
    if not pfad.exists():
        raise FileNotFoundError(
            f"Keine Modelldatei unter {pfad}. "
            "Bitte zuerst 'python -m src.pipelines.run_pipeline' ausführen."
        )
    beginn = time.perf_counter()
    inhalt = joblib.load(pfad)
    dienst = ModellDienst(pfad, inhalt)
    dienst.ladedauer_s = round(time.perf_counter() - beginn, 3)  # type: ignore[attr-defined]
    return dienst


def _main() -> None:
    """Lädt das Modell und beantwortet einen Beispielzustand — ohne Webserver."""
    dienst = lade_modell()
    print(f"geladen: {PATHS.relativ(dienst.pfad)} in {dienst.ladedauer_s} s")
    print(f"Schwellenwert: {dienst.schwelle}")
    print(f"SHA-256:       {dienst.sha256[:16]}…\n")

    beispiele = [
        {
            "type": "L",
            "air_temperature_k": 298.1,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1551,
            "torque_nm": 42.8,
            "tool_wear_min": 0,
        },
        {
            "type": "L",
            "air_temperature_k": 302.0,
            "process_temperature_k": 310.5,
            "rotational_speed_rpm": 1320,
            "torque_nm": 64.0,
            "tool_wear_min": 215,
        },
    ]
    frame = dienst.rahmen(beispiele)
    for zeile, p in zip(beispiele, dienst.wahrscheinlichkeiten(frame), strict=True):
        urteil = "ALARM" if dienst.alarm(p) else "ruhig"
        print(
            f"  Verschleiß {zeile['tool_wear_min']:>3} min, "
            f"Drehmoment {zeile['torque_nm']:>5} Nm -> "
            f"p = {p:.3f}  {urteil}"
        )

    hinweise = dienst.hinweise(frame)
    print("\nHinweise:", hinweise or "keine")


if __name__ == "__main__":
    _main()
