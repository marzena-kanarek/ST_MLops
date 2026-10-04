"""Ein- und Ausgabeformate der Schnittstelle.

**Der entscheidende Punkt an diesen Klassen:** Die Wertebereiche stehen hier
nicht als Zahlen im Code, sondern werden aus dem Datenvertrag in ``params.yaml``
gelesen. Dieselbe Zusage, gegen die die Trainingsdaten geprüft werden, prüft
damit auch jede Anfrage an die Schnittstelle. Wird eine Grenze im Vertrag
geändert, ändert sich die Schnittstelle mit — es gibt keine zweite Liste, die
auseinanderlaufen könnte.

Die Aufteilung folgt dem Vertrag:

* **harte Grenzen** (``min``/``max``) -> Pydantic lehnt die Anfrage mit
  Statuscode 422 und einer verständlichen Begründung ab.
* **weiche Grenzen** (``warn_min``/``warn_max``) -> die Anfrage wird beantwortet,
  der Hinweis steht als ``warnings`` in der Antwort. Ein Betriebspunkt kann
  ungewöhnlich sein, ohne unmöglich zu sein; ihn abzulehnen wäre falsch, ihn
  stillschweigend zu beantworten auch.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from src.data.validate import load_contract

_VERTRAG: dict[str, Any] = load_contract()
_SPALTEN: dict[str, dict[str, Any]] = _VERTRAG["columns"]


def _messgroesse(spalte: str, beschreibung: str) -> Any:
    """Baut einen Zahlentyp mit den harten Grenzen des Datenvertrags."""
    regeln = _SPALTEN[spalte]
    einheit = regeln.get("einheit", "")
    return Annotated[
        float,
        Field(
            ge=regeln["min"],
            le=regeln["max"],
            description=f"{beschreibung}" + (f" in {einheit}" if einheit else ""),
        ),
    ]


#: Erlaubte Qualitätsvarianten — ebenfalls aus dem Vertrag.
VARIANTEN = tuple(_SPALTEN["type"]["allowed"])


class Betriebszustand(BaseModel):
    """Ein Betriebspunkt der Maschine — die Eingabe einer Vorhersage.

    Die Ursachenspalten (``twf`` … ``rnf``) kommen hier bewusst **nicht** vor:
    Sie sind zum Vorhersagezeitpunkt nicht bekannt. Eine Schnittstelle, die nach
    ihnen fragt, wäre im Betrieb nicht bedienbar.
    """

    model_config = {
        "extra": "forbid",  # Tippfehler im Feldnamen sollen auffallen, nicht durchrutschen
        "json_schema_extra": {
            "examples": [
                {
                    "type": "L",
                    "air_temperature_k": 298.1,
                    "process_temperature_k": 308.6,
                    "rotational_speed_rpm": 1551,
                    "torque_nm": 42.8,
                    "tool_wear_min": 0,
                }
            ]
        },
    }

    type: Literal["L", "M", "H"] = Field(
        description="Qualitätsvariante des Werkstücks: L, M oder H"
    )
    air_temperature_k: _messgroesse("air_temperature_k", "Lufttemperatur")
    process_temperature_k: _messgroesse("process_temperature_k", "Prozesstemperatur")
    rotational_speed_rpm: _messgroesse("rotational_speed_rpm", "Drehzahl")
    torque_nm: _messgroesse("torque_nm", "Drehmoment")
    tool_wear_min: _messgroesse("tool_wear_min", "Werkzeugverschleiß")


class Stapel(BaseModel):
    """Mehrere Betriebspunkte in einer Anfrage."""

    model_config = {"extra": "forbid"}

    items: list[Betriebszustand] = Field(
        min_length=1,
        max_length=1000,
        description="1 bis 1000 Betriebspunkte",
    )


class Modellangabe(BaseModel):
    """Welches Modell geantwortet hat — gehört in jede Antwort."""

    name: str
    sha256: str = Field(description="SHA-256 der geladenen Modelldatei")
    git_commit: str
    raw_data_sha256: str
    trained_pr_auc: float


class Vorhersage(BaseModel):
    """Die Antwort auf einen Betriebspunkt.

    Zurückgegeben wird nicht nur die Wahrscheinlichkeit, sondern auch die
    **Entscheidung**, der verwendete **Schwellenwert**, das **Modell** und eine
    **Anfrage-Kennung**. Nur so lässt sich später nachvollziehen, wie eine
    Antwort zustande kam.
    """

    request_id: str
    failure_probability: float = Field(ge=0.0, le=1.0)
    alarm: bool = Field(description="True, wenn Wahrscheinlichkeit >= Schwellenwert")
    threshold: float
    warnings: list[str] = Field(
        default_factory=list,
        description="Hinweise aus den weichen Grenzen des Datenvertrags",
    )
    model: Modellangabe
    latency_ms: float


class StapelAntwort(BaseModel):
    """Die Antwort auf einen Stapel."""

    request_id: str
    count: int
    alarms: int
    threshold: float
    predictions: list[float] = Field(description="Ausfallwahrscheinlichkeit je Zeile")
    alarm_flags: list[bool]
    warnings: list[str]
    model: Modellangabe
    latency_ms: float


class Gesundheit(BaseModel):
    """Antwort von ``/health`` — für Startprüfungen und Überwachung."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_file: str | None = None
    uptime_s: float
    api_version: str


class ModellInfo(BaseModel):
    """Antwort von ``/model-info`` — was genau ist geladen."""

    name: str
    registry_name: str
    threshold: float
    model_file: str
    model_sha256: str
    loaded_at: str
    features: list[str] = Field(description="Erwartete Eingabefelder")
    engineered_features: list[str] = Field(description="In der Pipeline erzeugte Merkmale")
    metrics: dict[str, float] = Field(description="Kennzahlen des Trainingslaufs")
    provenance: dict[str, Any]
    costs: dict[str, float]
