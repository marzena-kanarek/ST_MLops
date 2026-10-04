"""Merkmalskonstruktion innerhalb der Pipeline.

Die abgeleiteten Größen stammen aus der Fachdomäne, nicht aus dem Ausprobieren.
Jede von ihnen fasst zusammen, was ein Baum sonst mühsam mit vielen Schnitten
annähern müsste.

**Warum das „innerhalb der Pipeline" entscheidend ist:** Würden die Merkmale im
Notebook berechnet und das Modell auf dem Ergebnis trainiert, müsste die
Schnittstelle exakt dieselbe Rechnung wiederholen. Zwei Stellen,
dieselbe Formel — sobald eine abweicht, bekommt das Modell im Betrieb andere
Zahlen als im Training. Nichts stürzt ab, die Güte sinkt nur leise. Steckt die
Berechnung dagegen als ``FunctionTransformer`` in der Pipeline, wandert sie mit
in die gespeicherte ``model.joblib``, und es gibt keine zweite Stelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

#: Namen der abgeleiteten Merkmale, in der Reihenfolge ihrer Erzeugung.
ABGELEITETE_MERKMALE: tuple[str, ...] = (
    "temp_difference_k",
    "power_w",
    "wear_torque_min_nm",
)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt die physikalisch begründeten Merkmale.

    * ``temp_difference_k`` — Prozess- minus Lufttemperatur. Beschreibt, wie gut
      die Wärme abgeführt wird; genau darum geht es bei der Ausfallart HDF.
      Nebeneffekt: Der gemeinsame, uninformative Anteil beider Temperaturen
      (0,88 Korrelation) wird durch die aussagekräftige Differenz ersetzt.
    * ``power_w`` — mechanische Leistung P = M · 2πn / 60. Fasst Drehmoment und
      Drehzahl in einer Größe zusammen. Beide Ausfallränder (hohe Last bei
      niedriger Drehzahl, niedrige Last bei hoher Drehzahl) werden damit zu
      *einer* auffälligen Zahl statt zu zwei getrennten Bereichen.
    * ``wear_torque_min_nm`` — Verschleiß mal Drehmoment. Ein verschlissenes
      Werkzeug unter hoher Last ist gefährlicher als beides für sich; das ist
      die Logik hinter der Ausfallart OSF (Überlastung).

    Args:
        frame: DataFrame mit den Rohmerkmalen.

    Returns:
        Eine Kopie mit drei zusätzlichen Spalten. Das Original bleibt unberührt.
    """
    frame = frame.copy()
    frame["temp_difference_k"] = frame["process_temperature_k"] - frame["air_temperature_k"]
    frame["power_w"] = frame["torque_nm"] * frame["rotational_speed_rpm"] * 2 * np.pi / 60
    frame["wear_torque_min_nm"] = frame["tool_wear_min"] * frame["torque_nm"]
    return frame


def baue_vorverarbeitung(skalieren: bool) -> ColumnTransformer:
    """One-hot für Kategorien, optional Standardisierung für Zahlen.

    Die Spalten werden über ihren Typ ausgewählt, nicht über feste Namen — so
    greift der Schritt auch auf den neu erzeugten Merkmalen, ohne dass hier eine
    Liste nachgepflegt werden muss.
    """
    return ColumnTransformer(
        [
            (
                "kategorial",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                make_column_selector(dtype_exclude="number"),
            ),
            (
                "numerisch",
                StandardScaler() if skalieren else "passthrough",
                make_column_selector(dtype_include="number"),
            ),
        ]
    )


def baue_pipeline(
    modell,
    mit_merkmalen: bool = True,
    skalieren: bool = False,
) -> Pipeline:
    """Setzt die vollständige Kette zusammen: Merkmale, Vorverarbeitung, Modell.

    Args:
        modell: ein sklearn-Schätzer.
        mit_merkmalen: wenn False, wird der Merkmalsschritt weggelassen — nur
            für den Vergleich in Etappe 8 gedacht.
        skalieren: True für Verfahren, die vergleichbare Größenordnungen
            brauchen (logistische Regression). Baumverfahren brauchen es nicht.

    Returns:
        Eine Pipeline, die Rohwerte entgegennimmt. Nach ``fit`` enthält sie die
        gesamte Kette — genau das macht sie als ``model.joblib`` vollständig.
    """
    schritte: list[tuple[str, object]] = []
    if mit_merkmalen:
        schritte.append(("merkmale", FunctionTransformer(add_engineered_features)))
    schritte.append(("vorverarbeitung", baue_vorverarbeitung(skalieren)))
    schritte.append(("modell", modell))
    return Pipeline(schritte)
