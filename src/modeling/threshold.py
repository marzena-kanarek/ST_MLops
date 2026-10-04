"""Entscheidungsregel: den Schwellenwert begründet wählen.

``predict()`` benutzt stillschweigend 0,5. Dieser Wert ist bei seltenen
Ereignissen und ungleichen Fehlerkosten praktisch immer falsch — er ist eine
Voreinstellung, keine Entscheidung.

Gesucht wird der Schwellenwert, der die **erwarteten Kosten** minimiert. Gesucht
wird er auf der **Validierungsmenge**; die Testmenge bleibt unangetastet. Der
gefundene Wert gehört anschließend zum Modell wie seine Gewichte — er wird
zusammen mit ihm gespeichert, nicht in einem Notebook notiert.

Aufruf von der Kommandozeile::

    python -m src.modeling.threshold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.config import load_params

#: Die stillschweigende Voreinstellung von ``predict()``. Dies ist **kein**
#: Einstellwert, sondern der Vergleichspunkt, gegen den die eigene Entscheidung
#: begründet wird — deshalb steht er hier und nicht in params.yaml. Der
#: *gewählte* Schwellenwert steht dort (``decision.threshold``).
VOREINSTELLUNG_SCHWELLE = 0.5


def standardraster(params: dict | None = None) -> np.ndarray:
    """Das Raster der zu prüfenden Schwellenwerte aus params.yaml.

    Steht in der Konfiguration (``decision.grid``), nicht im Code: Wer das
    Raster verfeinern will, soll dafür kein Python anfassen müssen.
    """
    raster = (params or load_params())["decision"]["grid"]
    return np.linspace(raster["start"], raster["stop"], raster["steps"])


def kosten_je_schwelle(
    y_wahr: pd.Series,
    wahrscheinlichkeit: np.ndarray,
    kosten: dict[str, float],
    schwellen: np.ndarray | None = None,
) -> pd.DataFrame:
    """Rechnet für jeden Schwellenwert die erwarteten Kosten aus.

    Args:
        y_wahr: tatsächliche Zielwerte.
        wahrscheinlichkeit: geschätzte Ausfallwahrscheinlichkeiten.
        kosten: Abschnitt ``costs`` aus params.yaml.
        schwellen: zu prüfende Werte. Standard: das Raster aus params.yaml
            (``decision.grid``).

    Returns:
        DataFrame mit Schwelle, Verwechslungsmatrix, Recall, Precision und Kosten.
    """
    if schwellen is None:
        schwellen = standardraster()

    zeilen = []
    for schwelle in schwellen:
        vorhersage = (wahrscheinlichkeit >= schwelle).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_wahr, vorhersage, labels=[0, 1]).ravel()
        zeilen.append(
            {
                "Schwelle": float(schwelle),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "Recall": float(recall_score(y_wahr, vorhersage, zero_division=0)),
                "Precision": float(precision_score(y_wahr, vorhersage, zero_division=0)),
                "Kosten EUR": float(
                    fn * kosten["false_negative_eur"]
                    + fp * kosten["false_positive_eur"]
                    + tp * kosten["true_positive_eur"]
                ),
            }
        )
    return pd.DataFrame(zeilen)


def finde_beste_schwelle(kurve: pd.DataFrame) -> dict[str, float]:
    """Wählt den Schwellenwert mit den geringsten erwarteten Kosten.

    Bei mehreren gleich guten Werten wird der **größte** genommen: Er erzeugt
    weniger Fehlalarme bei gleichen Kosten und ist damit im Betrieb der
    ruhigere.
    """
    minimum = kurve["Kosten EUR"].min()
    kandidaten = kurve[kurve["Kosten EUR"] == minimum]
    beste = kandidaten.iloc[-1]
    return {
        "schwelle": float(beste["Schwelle"]),
        "kosten_eur": float(beste["Kosten EUR"]),
        "recall": float(beste["Recall"]),
        "precision": float(beste["Precision"]),
        "tp": int(beste["TP"]),
        "fp": int(beste["FP"]),
        "fn": int(beste["FN"]),
        "gleichwertige_kandidaten": int(len(kandidaten)),
    }


def _main() -> None:
    """Trainiert das gewählte Modell und sucht die Entscheidungsschwelle."""
    from sklearn.ensemble import RandomForestClassifier

    from src.data.split import lade_teilmengen, trenne_merkmale_und_ziel
    from src.features.build_features import baue_pipeline

    params = load_params()
    kosten = params["costs"]

    train, val, _ = lade_teilmengen()
    X_train, y_train = trenne_merkmale_und_ziel(train, params)
    X_val, y_val = trenne_merkmale_und_ziel(val, params)

    modell = RandomForestClassifier(
        random_state=params["seed"], n_jobs=-1, **params["model"]["random_forest"]
    )
    pipeline = baue_pipeline(modell)
    pipeline.fit(X_train, y_train)

    wahrscheinlichkeit = pipeline.predict_proba(X_val)[:, 1]
    kurve = kosten_je_schwelle(y_val, wahrscheinlichkeit, kosten, standardraster(params))
    beste = finde_beste_schwelle(kurve)

    bei_05 = kurve.iloc[(kurve["Schwelle"] - VOREINSTELLUNG_SCHWELLE).abs().argmin()]
    nichts_tun = int(y_val.sum()) * kosten["false_negative_eur"]

    def euro(betrag: float) -> str:
        """Tausenderpunkte wie im Deutschen."""
        return f"{betrag:>10,.0f}".replace(",", ".") + " EUR"

    print(f"Validierungsmenge: {len(y_val)} Zeilen, {int(y_val.sum())} Ausfälle\n")
    print(f"gewählter Schwellenwert : {beste['schwelle']:.2f}")
    print(f"  Kosten                : {euro(beste['kosten_eur'])}")
    print(f"  Recall / Precision    : {beste['recall']:.3f} / {beste['precision']:.3f}")
    print(f"  TP/FP/FN              : {beste['tp']} / {beste['fp']} / {beste['fn']}")
    print()
    print(f"zum Vergleich, Schwelle 0,50: {euro(bei_05['Kosten EUR'])}")
    print(f"zum Vergleich, nichts tun   : {euro(nichts_tun)}")
    print()
    print(f"Ersparnis gegenüber 0,50    : {euro(bei_05['Kosten EUR'] - beste['kosten_eur'])}")


if __name__ == "__main__":
    _main()
