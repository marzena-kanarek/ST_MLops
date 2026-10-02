"""Gemeinsame Vorrichtungen für alle Tests.

Die Fixtures sind bewusst klein gehalten: Tests sollen in Sekunden durchlaufen,
sonst werden sie nicht ausgeführt. Das trainierte Kleinstmodell wird einmal je
Testlauf gebaut (``scope="session"``) statt für jeden Test neu.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data.split import lade_teilmengen, trenne_merkmale_und_ziel
from src.data.validate import load_params
from src.features.build_features import baue_pipeline
from src.utils import logging_setup


@pytest.fixture(scope="session", autouse=True)
def protokoll_nur_auf_den_bildschirm():
    """Während der Tests nicht in reports/api.log schreiben.

    Ein Testlauf ist kein Betriebsvorgang; seine Meldungen gehören nicht in das
    Protokoll des Dienstes. Auf dem Bildschirm bleiben sie sichtbar — pytest
    zeigt sie bei einem Fehlschlag an.
    """
    logging_setup.richte_ein(mit_datei=False, erneut=True)
    yield


@pytest.fixture(scope="session")
def params() -> dict:
    """Der Inhalt von params.yaml."""
    return load_params()


@pytest.fixture
def beispiel_frame() -> pd.DataFrame:
    """Ein kleiner, gültiger DataFrame mit allen Pflichtspalten.

    Von Hand geschrieben statt aus der Datei gelesen: Tests sollen auch dann
    laufen, wenn data/processed/ noch nicht erzeugt wurde.
    """
    return pd.DataFrame(
        {
            "type": ["L", "M", "H", "L"],
            "air_temperature_k": [298.1, 300.4, 302.7, 299.0],
            "process_temperature_k": [308.6, 310.2, 312.1, 309.5],
            "rotational_speed_rpm": [1551, 1408, 1498, 2861],
            "torque_nm": [42.8, 46.3, 49.4, 4.6],
            "tool_wear_min": [0, 85, 198, 12],
            "machine_failure": [0, 0, 1, 0],
        }
    )


@pytest.fixture(scope="session")
def teilmengen():
    """Training und Validierung aus data/processed/.

    Überspringt die abhängigen Tests, wenn die Teilmengen fehlen — dann wurde
    ``python -m src.data.split`` noch nicht ausgeführt.
    """
    try:
        train, val, _ = lade_teilmengen()
    except FileNotFoundError:
        pytest.skip("data/processed/ fehlt — erst 'python -m src.data.split' ausführen")
    return train, val


@pytest.fixture(scope="session")
def trainiertes_kleinstmodell(teilmengen, params):
    """Ein absichtlich kleines Modell, nur zum Prüfen des Verhaltens.

    30 Bäume mit Tiefe 8 statt der 150 aus params.yaml: Für die Frage, ob
    Wahrscheinlichkeiten zwischen 0 und 1 liegen und mehr Verschleiß das Risiko
    erhöht, reicht das — und es trainiert in unter einer Sekunde.
    """
    train, _ = teilmengen
    X, y = trenne_merkmale_und_ziel(train, params)
    pipeline = baue_pipeline(
        RandomForestClassifier(n_estimators=30, max_depth=8, random_state=params["seed"], n_jobs=1)
    )
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def basiszustand() -> pd.DataFrame:
    """Ein unauffälliger Betriebspunkt als Ausgangslage für Verhaltenstests."""
    return pd.DataFrame(
        [
            {
                "type": "L",
                "air_temperature_k": 300.0,
                "process_temperature_k": 310.0,
                "rotational_speed_rpm": 1500,
                "torque_nm": 40.0,
                "tool_wear_min": 50,
            }
        ]
    )


@pytest.fixture
def risiko(trainiertes_kleinstmodell):
    """Kurzform: gibt die Ausfallwahrscheinlichkeit für einen Zustand zurück."""

    def _risiko(zustand: pd.DataFrame) -> float:
        return float(trainiertes_kleinstmodell.predict_proba(zustand)[0, 1])

    return _risiko
