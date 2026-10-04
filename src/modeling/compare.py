"""Modelle vergleichen und abstimmen.

Vorgehen in drei Schritten, bewusst in dieser Reihenfolge:

1. **Kandidaten mit Kreuzvalidierung auf der Trainingsmenge vergleichen.** Ein
   einzelner Wert auf der Validierungsmenge schwankt bei 51 Ausfällen stark;
   fünf Teilungen zeigen auch die Streuung.
2. **Auf der Validierungsmenge bestätigen.** Erst hier fällt die Wahl.
3. **Danach** Hyperparameter abstimmen — und zwar mit ``RandomizedSearchCV``
   statt ``GridSearchCV``: Bei gleichem Zeitbudget findet die Zufallssuche in
   der Regel bessere Werte, weil sie nicht an einem Gitter klebt.

Die Testmenge wird in diesem Modul nicht angefasst.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from xgboost import XGBClassifier

from src.config import load_params
from src.features.build_features import baue_pipeline

#: Bewertungsmaß für alle Vergleiche — PR-AUC heißt in sklearn average_precision.
SCORING = "average_precision"


def kandidaten(
    seed: int,
    positiv_verhaeltnis: float = 1.0,
    einstellungen: dict[str, Any] | None = None,
) -> dict[str, tuple[Any, bool]]:
    """Die drei Kandidaten: linear, Bagging, Boosting.

    Args:
        seed: Zufallszahl-Startwert.
        positiv_verhaeltnis: negative/positive Fälle — Startwert für
            ``scale_pos_weight`` bei XGBoost.
        einstellungen: Abschnitt ``comparison`` aus params.yaml.

    Returns:
        Name -> (Modell, ob standardisiert werden muss).
    """
    einstellungen = einstellungen or load_params()["comparison"]
    return {
        "Logistische Regression": (
            LogisticRegression(random_state=seed, **einstellungen["logistic_regression"]),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1, **einstellungen["random_forest"]),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="aucpr",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
                **einstellungen["xgboost"],
            ),
            False,
        ),
    }


def kreuzvalidiere(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    falten: int | None = None,
) -> pd.DataFrame:
    """Vergleicht alle Kandidaten mit geschichteter Kreuzvalidierung.

    Geschichtet (``StratifiedKFold``), weil sonst einzelne Falten kaum Ausfälle
    enthalten könnten. Die Zahl der Falten steht in params.yaml
    (``tuning.cv_folds``).
    """
    falten = falten or load_params()["tuning"]["cv_folds"]
    aufteilung = StratifiedKFold(n_splits=falten, shuffle=True, random_state=seed)
    verhaeltnis = float((y == 0).sum() / (y == 1).sum())

    zeilen = []
    for name, (modell, skalieren) in kandidaten(seed, verhaeltnis).items():
        pipeline = baue_pipeline(modell, mit_merkmalen=True, skalieren=skalieren)
        werte = cross_val_score(pipeline, X, y, cv=aufteilung, scoring=SCORING)
        zeilen.append(
            {
                "Modell": name,
                "PR-AUC (Mittel)": float(werte.mean()),
                "Streuung": float(werte.std()),
                "schlechteste Falte": float(werte.min()),
                "beste Falte": float(werte.max()),
            }
        )
    return pd.DataFrame(zeilen).set_index("Modell")


def suchraum(name: str, positiv_verhaeltnis: float) -> dict[str, Any]:
    """Suchräume für die Zufallssuche.

    Die Präfixe ``modell__`` sprechen den letzten Schritt der Pipeline an.

    Diese Bereiche stehen bewusst im Code und nicht in params.yaml: Sie legen
    nicht fest, *womit das System rechnet*, sondern *welcher Bereich einmalig
    durchsucht wurde*. Das Ergebnis der Suche — die tatsächlich verwendeten
    Werte — steht in params.yaml unter ``model``.
    """
    if name == "Random Forest":
        return {
            "modell__n_estimators": randint(100, 600),
            "modell__max_depth": [6, 10, 15, 20, None],
            "modell__min_samples_leaf": randint(1, 10),
            "modell__max_features": ["sqrt", "log2", 0.5],
            "modell__class_weight": [None, "balanced", "balanced_subsample"],
        }
    if name == "XGBoost":
        return {
            "modell__n_estimators": randint(100, 600),
            "modell__max_depth": randint(3, 10),
            "modell__learning_rate": loguniform(0.01, 0.3),
            "modell__subsample": uniform(0.6, 0.4),
            "modell__colsample_bytree": uniform(0.6, 0.4),
            "modell__min_child_weight": randint(1, 10),
            "modell__scale_pos_weight": [1.0, np.sqrt(positiv_verhaeltnis), positiv_verhaeltnis],
        }
    if name == "Logistische Regression":
        return {
            "modell__C": loguniform(0.01, 100),
            "modell__class_weight": [None, "balanced"],
        }
    raise ValueError(f"Kein Suchraum für '{name}' hinterlegt.")


def stimme_ab(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    versuche: int | None = None,
    falten: int | None = None,
) -> RandomizedSearchCV:
    """Sucht Hyperparameter mit Zufallssuche auf der Trainingsmenge.

    Zahl der Versuche und Falten stehen in params.yaml (``tuning``).
    """
    abstimmung = load_params()["tuning"]
    versuche = versuche or abstimmung["n_iter"]
    falten = falten or abstimmung["cv_folds"]
    verhaeltnis = float((y == 0).sum() / (y == 1).sum())
    modell, skalieren = kandidaten(seed, verhaeltnis)[name]
    pipeline = baue_pipeline(modell, mit_merkmalen=True, skalieren=skalieren)

    suche = RandomizedSearchCV(
        pipeline,
        param_distributions=suchraum(name, verhaeltnis),
        n_iter=versuche,
        scoring=SCORING,
        cv=StratifiedKFold(n_splits=falten, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
    )
    suche.fit(X, y)
    return suche
