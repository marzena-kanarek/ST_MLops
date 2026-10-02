"""Baselines als Messlatte (Etappe 7).

Ohne Messlatte weiß niemand, ob 0,74 PR-AUC gut ist. Deshalb werden zuerst zwei
bewusst einfache Vorhersagen gebaut, gegen die sich jedes komplexere Modell
rechtfertigen muss:

* **Zufall im richtigen Verhältnis** (``DummyClassifier``) — rät nach dem
  Klassenverhältnis. Die PR-AUC entspricht ungefähr dem Anteil der positiven
  Klasse. Alles, was nicht deutlich darüber liegt, misst nichts.
* **Logistische Regression** auf den Rohspalten — die einfachste ernsthafte
  Lösung. Sie zieht eine gerade Trennlinie; schlägt ein Baumverfahren sie
  deutlich, ist das der Beleg, dass die Nichtlinearität wirklich gebraucht wird.

Bewertet wird auf der **Validierungsmenge**. Die Testmenge bleibt unangetastet.

Aufruf von der Kommandozeile::

    python -m src.modeling.baseline
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PATHS, load_params
from src.data.split import lade_teilmengen, trenne_merkmale_und_ziel
from src.modeling.threshold import VOREINSTELLUNG_SCHWELLE


def spaltengruppen(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Teilt die Spalten in kategorial und numerisch.

    Geprüft wird auf „nicht numerisch" statt auf einen bestimmten Typ — ab
    pandas 3 haben Textspalten einen eigenen str-Typ und sind nicht mehr
    ``object``.
    """
    numerisch = [s for s in X.columns if pd.api.types.is_numeric_dtype(X[s])]
    kategorial = [s for s in X.columns if s not in numerisch]
    return kategorial, numerisch


def baue_vorverarbeitung(X: pd.DataFrame, skalieren: bool = True) -> ColumnTransformer:
    """Kategorien one-hot, Zahlen optional standardisiert.

    Die logistische Regression braucht vergleichbare Größenordnungen — Drehzahl
    liegt bei 1500, Drehmoment bei 40. Baumverfahren brauchen das nicht.
    """
    kategorial, numerisch = spaltengruppen(X)
    schritte: list[tuple[str, Any, list[str]]] = [
        ("kategorial", OneHotEncoder(handle_unknown="ignore", drop="first"), kategorial)
    ]
    schritte.append(("numerisch", StandardScaler() if skalieren else "passthrough", numerisch))
    return ColumnTransformer(schritte)


def baue_baselines(
    X: pd.DataFrame,
    seed: int,
    einstellungen: dict[str, Any] | None = None,
) -> dict[str, Pipeline]:
    """Die Messlatten, in aufsteigender Ernsthaftigkeit.

    „Kein Alarm" ist der betriebliche Ausgangszustand: gar nichts tun. Es ist
    der Vergleichspunkt für die Kostenspalte — jedes Modell muss zeigen, dass es
    billiger ist als Nichtstun.

    Args:
        X: Trainingsmerkmale, nur zur Bestimmung der Spaltengruppen.
        seed: Zufallszahl-Startwert.
        einstellungen: Abschnitt ``comparison`` aus params.yaml.
    """
    einstellungen = einstellungen or load_params()["comparison"]
    return {
        "Kein Alarm (immer 0)": Pipeline(
            [
                ("modell", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "Zufall (stratifiziert)": Pipeline(
            [
                ("modell", DummyClassifier(strategy="stratified", random_state=seed)),
            ]
        ),
        "Logistische Regression": Pipeline(
            [
                ("vorverarbeitung", baue_vorverarbeitung(X, skalieren=True)),
                (
                    "modell",
                    LogisticRegression(random_state=seed, **einstellungen["logistic_regression"]),
                ),
            ]
        ),
    }


def erwartete_kosten(
    y_wahr: pd.Series,
    y_vorhersage: pd.Series,
    kosten: dict[str, float],
) -> float:
    """Erwartete Kosten in Euro für eine Vorhersage.

    Übersetzt die Verwechslungsmatrix in die Größe, über die im Betrieb
    tatsächlich entschieden wird.
    """
    tn, fp, fn, tp = confusion_matrix(y_wahr, y_vorhersage, labels=[0, 1]).ravel()
    return float(
        fn * kosten["false_negative_eur"]
        + fp * kosten["false_positive_eur"]
        + tp * kosten["true_positive_eur"]
    )


def bewerte(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    kosten: dict[str, float],
    bezeichnung: str,
    schwelle: float = VOREINSTELLUNG_SCHWELLE,
) -> dict[str, Any]:
    """Trainiert und bewertet ein Modell auf der Validierungsmenge."""
    pipeline.fit(X_train, y_train)
    wahrscheinlichkeit = pipeline.predict_proba(X_val)[:, 1]
    vorhersage = (wahrscheinlichkeit >= schwelle).astype(int)
    return {
        "Modell": bezeichnung,
        "PR-AUC": float(average_precision_score(y_val, wahrscheinlichkeit)),
        "ROC-AUC": float(roc_auc_score(y_val, wahrscheinlichkeit)),
        "Recall": float(recall_score(y_val, vorhersage, zero_division=0)),
        "Precision": float(precision_score(y_val, vorhersage, zero_division=0)),
        "Kosten EUR": erwartete_kosten(y_val, vorhersage, kosten),
    }


def _main() -> None:
    """Trainiert beide Baselines und legt die Kennzahlen ab."""
    params = load_params()
    seed = params["seed"]
    kosten = params["costs"]

    train, val, _ = lade_teilmengen()
    X_train, y_train = trenne_merkmale_und_ziel(train, params)
    X_val, y_val = trenne_merkmale_und_ziel(val, params)

    ergebnisse = [
        bewerte(pipeline, X_train, y_train, X_val, y_val, kosten, bezeichnung)
        for bezeichnung, pipeline in baue_baselines(X_train, seed, params["comparison"]).items()
    ]

    tabelle = pd.DataFrame(ergebnisse).set_index("Modell")
    print(
        f"Bewertung auf der Validierungsmenge ({len(y_val)} Zeilen, {int(y_val.sum())} Ausfälle)\n"
    )
    print(tabelle.round(3).to_string())

    anteil = float(y_train.mean())
    print(f"\nZum Vergleich: Anteil der positiven Klasse = {anteil:.3f}")
    print("Die PR-AUC des Zufallsmodells muss ungefähr auf diesem Wert liegen.")

    ziel_pfad = PATHS.reports / "baseline_results.json"
    ziel_pfad.parent.mkdir(parents=True, exist_ok=True)
    ziel_pfad.write_text(
        json.dumps(
            {"positiv_anteil_train": anteil, "ergebnisse": ergebnisse},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\ngeschrieben: {PATHS.relativ(ziel_pfad)}")


if __name__ == "__main__":
    _main()
