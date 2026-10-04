"""Fehler nach Segmenten zerlegen (Error Slicing).

Eine Gesamtzahl wie „Recall 0,86" verdeckt, **wo** das Modell danebenliegt. Error
Slicing zerlegt dieselben Vorhersagen in Teilmengen und fragt je Teilmenge: Wie
viele Ausfälle werden gefunden, wie viele Alarme sind falsch, und was kostet das?

Zwei Sichten:

* **Einzelscheiben** — je Qualitätsvariante, je Verschleißband, je Drehmomentband
  und so weiter. Das ist die Hauptansicht.
* **Kreuzklassifikation** — zwei Merkmale zugleich, als zweite Diagnoseebene.
  Kleine Gruppen erzeugen instabile Kennzahlen, deshalb wird die Gruppengröße
  immer mitgeführt und für die Abbildungen eine Mindestgröße verlangt.

**Die Ursachenspalten kommen hier zurück — als Diagnose, nicht als Merkmal.**
``TWF``, ``HDF``, ``PWF``, ``OSF`` und ``RNF`` dürfen dem Modell nicht gezeigt
werden (Data Leakage). Für die Fehleranalyse sind sie dagegen genau
richtig: Sie beantworten die Frage, welche *Art* von Ausfall das Modell
übersieht. Gelesen werden sie aus den Rohdaten und über den Zeilenindex
zugeordnet; in die Vorhersage geht nichts davon ein.

Gerechnet wird auf der **Validierungsmenge**. Die Testmenge ist nach der
einmaligen Messungverbraucht; sie hier noch einmal zu zerlegen
hieße, nach dem Ergebnis weiterzusuchen.

Aufruf::

    python -m src.modeling.error_slicing
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import PATHS, load_params
from src.data.load import load_raw_data
from src.data.split import entferne_leakage_und_kennungen, lade_teilmengen, split_data
from src.utils.logging_setup import hole_logger

log = hole_logger(__name__)

BERICHT = "error_slicing.json"

#: Mindestgröße einer Gruppe, damit ihre Kennzahlen gezeigt werden. Bei zehn
#: Zeilen und einem Ausfall springt der Recall zwischen 0 und 1 - das ist kein
#: Befund, sondern Rauschen.
MINDESTGROESSE = 50

#: Die Ursachenspalten, in der Reihenfolge der Datensatzbeschreibung.
URSACHEN = ("twf", "hdf", "pwf", "osf", "rnf")

URSACHE_LANG = {
    "twf": "TWF — Werkzeugverschleiß",
    "hdf": "HDF — Wärmeabfuhr",
    "pwf": "PWF — Leistung außerhalb des Bereichs",
    "osf": "OSF — Überlastung",
    "rnf": "RNF — Zufallsausfall",
}


# ── Daten zusammenstellen ───────────────────────────────────────────────


def lade_bewertungsdaten(params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Validierungsmenge mit Wahrscheinlichkeit, Entscheidung, Fehlerart und Ursachen.

    Die Ursachenspalten werden über den Zeilenindex aus den Rohdaten geholt. Damit
    das zulässig ist, wird die Aufteilung aus denselben Rohdaten mit demselben
    Startwert noch einmal gerechnet und **geprüft**, dass sie Zeile für Zeile mit
    der abgelegten Validierungsmenge übereinstimmt.
    """
    params = params or load_params()
    ziel = params["data"]["target"]

    _, val, _ = lade_teilmengen()

    roh = load_raw_data()
    ohne = entferne_leakage_und_kennungen(roh, params)
    _, val_mit_index, _ = split_data(ohne, params)

    vergleich = val_mit_index.reset_index(drop=True)
    if not vergleich.equals(val):
        raise RuntimeError(
            "Die neu gerechnete Aufteilung stimmt nicht mit data/processed/val.parquet "
            "überein. Ohne diese Übereinstimmung lassen sich die Ursachenspalten "
            "nicht zuordnen. Bitte 'python -m src.pipelines.run_pipeline' ausführen."
        )

    frame = val_mit_index.copy()
    for spalte in URSACHEN:
        frame[spalte] = roh.loc[frame.index, spalte].to_numpy()

    pipeline = joblib.load(PATHS.models / "model.joblib")
    schwelle = float(pipeline["schwelle"])
    merkmale = val_mit_index.drop(columns=[ziel])

    frame["wahrscheinlichkeit"] = pipeline["pipeline"].predict_proba(merkmale)[:, 1]
    frame["vorhersage"] = (frame["wahrscheinlichkeit"] >= schwelle).astype(int)
    frame["fehlerart"] = np.select(
        [
            (frame[ziel] == 1) & (frame["vorhersage"] == 1),
            (frame[ziel] == 0) & (frame["vorhersage"] == 1),
            (frame[ziel] == 1) & (frame["vorhersage"] == 0),
        ],
        ["TP", "FP", "FN"],
        default="TN",
    )
    frame.attrs["schwelle"] = schwelle
    return frame.reset_index(drop=True)


def _band(werte: pd.Series, teile: int = 4) -> pd.Categorical:
    """Teilt eine Messgröße in Quantilbänder mit lesbaren, geordneten Namen.

    Die Namen sind kurz ("51–105") und die Kategorie ist **geordnet**: Sonst
    sortiert pandas sie als Text, und in jeder Tabelle und jeder Abbildung stünde
    das Band "105–163" vor "51–105". Das ist kein Schönheitsfehler — eine
    Abbildung mit vertauschten Bändern liest man falsch.
    """
    schnitt = pd.qcut(werte, q=teile, duplicates="drop")
    spanne = float(werte.max() - werte.min())
    stellen = 1 if spanne < 50 else 0
    etiketten = [
        f"{g.left:.{stellen}f}–{g.right:.{stellen}f}".replace("-0.0–", "0–")
        for g in schnitt.cat.categories
    ]
    # Die Untergrenze des ersten Bandes setzt qcut knapp unter das Minimum.
    if etiketten:
        erste = schnitt.cat.categories[0]
        etiketten[0] = (
            f"{max(erste.left, float(werte.min())):.{stellen}f}–{erste.right:.{stellen}f}"
        )
    return pd.Categorical(
        schnitt.cat.rename_categories(etiketten), categories=etiketten, ordered=True
    )


def ergaenze_baender(frame: pd.DataFrame) -> pd.DataFrame:
    """Teilt die Messgrößen in Bänder.

    Die Grenzen kommen aus den Quantilen der Daten, nicht aus angenommenen
    physikalischen Schwellen: So ist jede Gruppe ähnlich groß, und es steckt keine
    Vermutung über den Ausfallmechanismus in der Einteilung. Ob sich die Fehler an
    einer bestimmten Stelle häufen, soll die Auswertung zeigen und nicht die
    Einteilung vorwegnehmen.
    """
    frame = frame.copy()
    frame["temp_difference_k"] = frame["process_temperature_k"] - frame["air_temperature_k"]
    frame["power_w"] = frame["torque_nm"] * frame["rotational_speed_rpm"] * 2 * np.pi / 60

    for spalte, name in (
        ("tool_wear_min", "Verschleißband"),
        ("torque_nm", "Drehmomentband"),
        ("rotational_speed_rpm", "Drehzahlband"),
        ("temp_difference_k", "Temperaturdifferenzband"),
        ("power_w", "Leistungsband"),
    ):
        frame[name] = _band(frame[spalte])

    frame = frame.rename(columns={"type": "Qualitätsvariante"})
    # L, M, H ist die Reihenfolge des Datensatzes (low, medium, high) und die, in
    # der man sie lesen will - alphabetisch stünde H vorn.
    vorhanden = [v for v in ("L", "M", "H") if v in set(frame["Qualitätsvariante"])]
    frame["Qualitätsvariante"] = pd.Categorical(
        frame["Qualitätsvariante"], categories=vorhanden, ordered=True
    )
    return frame


#: Die Einzelscheiben, in der Reihenfolge der Auswertung.
SCHEIBEN = (
    "Qualitätsvariante",
    "Verschleißband",
    "Drehmomentband",
    "Drehzahlband",
    "Temperaturdifferenzband",
    "Leistungsband",
)

#: Kreuzklassifikationen als zweite Diagnoseebene.
KREUZE = (
    ("Qualitätsvariante", "Verschleißband"),
    ("Drehmomentband", "Drehzahlband"),
    ("Temperaturdifferenzband", "Drehzahlband"),
)


# ── Kennzahlen ──────────────────────────────────────────────────────────


def _zeile(teil: pd.DataFrame, kosten: dict[str, float], ziel: str) -> dict[str, Any]:
    """Kennzahlen einer Teilmenge."""
    tp = int((teil["fehlerart"] == "TP").sum())
    fp = int((teil["fehlerart"] == "FP").sum())
    fn = int((teil["fehlerart"] == "FN").sum())
    positive = tp + fn

    recall = tp / positive if positive else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if positive and (tp + fp) and (precision + recall) > 0
        else np.nan
    )

    return {
        "Zeilen": int(len(teil)),
        "Ausfälle": positive,
        "Ausfallrate": float(teil[ziel].mean()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Kosten EUR": float(
            fn * kosten["false_negative_eur"]
            + fp * kosten["false_positive_eur"]
            + tp * kosten["true_positive_eur"]
        ),
    }


def gesamt(frame: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Die Kennzahlen über alle Zeilen — der Bezugspunkt jeder Scheibe."""
    params = params or load_params()
    return _zeile(frame, params["costs"], params["data"]["target"])


def scheibe(frame: pd.DataFrame, spalte: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Kennzahlen je Ausprägung einer Spalte."""
    params = params or load_params()
    kosten, ziel = params["costs"], params["data"]["target"]

    zeilen = {
        str(wert): _zeile(teil, kosten, ziel) for wert, teil in frame.groupby(spalte, observed=True)
    }
    tabelle = pd.DataFrame(zeilen).T
    tabelle.index.name = spalte
    gesamtkosten = tabelle["Kosten EUR"].sum()
    tabelle["Kostenanteil"] = tabelle["Kosten EUR"] / gesamtkosten if gesamtkosten else 0.0
    return tabelle


def _ordnung(frame: pd.DataFrame, spalte: str, vorhanden) -> list[str]:
    """Die Ausprägungen einer Spalte in ihrer sachlichen Reihenfolge."""
    werte = frame[spalte]
    gesehen = {str(v) for v in vorhanden}
    if isinstance(werte.dtype, pd.CategoricalDtype):
        return [str(k) for k in werte.cat.categories if str(k) in gesehen]
    return sorted(gesehen)


def kreuz(
    frame: pd.DataFrame,
    zeilen_spalte: str,
    spalten_spalte: str,
    kennzahl: str = "Recall",
    mindestgroesse: int = MINDESTGROESSE,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kreuzklassifikation zweier Merkmale.

    Returns:
        (Kennzahl je Feld, Gruppengröße je Feld). Felder unter der
        Mindestgröße werden auf ``NaN`` gesetzt — eine Zahl aus zwölf Zeilen
        sieht aus wie ein Befund und ist keiner.
    """
    params = params or load_params()
    kosten, ziel = params["costs"], params["data"]["target"]

    werte, groessen = {}, {}
    for (a, b), teil in frame.groupby([zeilen_spalte, spalten_spalte], observed=True):
        kennzahlen = _zeile(teil, kosten, ziel)
        werte[(str(a), str(b))] = kennzahlen[kennzahl]
        groessen[(str(a), str(b))] = kennzahlen["Zeilen"]

    tabelle = pd.Series(werte).unstack()
    anzahl = pd.Series(groessen).unstack()

    # unstack sortiert alphabetisch und verliert damit die Ordnung der Bänder:
    # "105–163" stünde vor "51–105". In einer Abbildung führt das in die Irre,
    # deshalb wird die Reihenfolge der Kategorie wiederhergestellt.
    tabelle = tabelle.reindex(
        index=_ordnung(frame, zeilen_spalte, tabelle.index),
        columns=_ordnung(frame, spalten_spalte, tabelle.columns),
    )
    anzahl = anzahl.reindex(index=tabelle.index, columns=tabelle.columns)
    tabelle = tabelle.where(anzahl >= mindestgroesse)
    tabelle.index.name = zeilen_spalte
    tabelle.columns.name = spalten_spalte
    return tabelle, anzahl


def nach_ursache(frame: pd.DataFrame) -> pd.DataFrame:
    """Welche **Art** von Ausfall findet das Modell, welche nicht?

    Nur auf den tatsächlichen Ausfällen gerechnet. Eine Zeile kann mehrere
    Ursachen tragen, deshalb überlappen sich die Gruppen; die Summe der Zeilen
    ist größer als die Zahl der Ausfälle.
    """
    ausfaelle = frame[frame["fehlerart"].isin(["TP", "FN"])]
    zeilen = []
    for ursache in URSACHEN:
        teil = ausfaelle[ausfaelle[ursache] == 1]
        if teil.empty:
            zeilen.append(
                {
                    "Ursache": URSACHE_LANG[ursache],
                    "Ausfälle": 0,
                    "gefunden": 0,
                    "übersehen": 0,
                    "Recall": np.nan,
                    "mittlere Wahrscheinlichkeit": np.nan,
                }
            )
            continue
        gefunden = int((teil["fehlerart"] == "TP").sum())
        zeilen.append(
            {
                "Ursache": URSACHE_LANG[ursache],
                "Ausfälle": int(len(teil)),
                "gefunden": gefunden,
                "übersehen": int(len(teil)) - gefunden,
                "Recall": gefunden / len(teil),
                "mittlere Wahrscheinlichkeit": float(teil["wahrscheinlichkeit"].mean()),
            }
        )

    ohne = ausfaelle[ausfaelle[list(URSACHEN)].sum(axis=1) == 0]
    if not ohne.empty:
        gefunden = int((ohne["fehlerart"] == "TP").sum())
        zeilen.append(
            {
                "Ursache": "ohne eingetragene Ursache",
                "Ausfälle": int(len(ohne)),
                "gefunden": gefunden,
                "übersehen": int(len(ohne)) - gefunden,
                "Recall": gefunden / len(ohne),
                "mittlere Wahrscheinlichkeit": float(ohne["wahrscheinlichkeit"].mean()),
            }
        )
    return pd.DataFrame(zeilen).set_index("Ursache")


def auffaellige_scheiben(
    frame: pd.DataFrame,
    mindestgroesse: int = MINDESTGROESSE,
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Die Scheiben, die man zuerst ansehen sollte.

    Sortiert nach Abstand zum Gesamt-Recall: Wo findet das Modell deutlich
    weniger Ausfälle als im Durchschnitt? Gruppen unter der Mindestgröße und
    Gruppen ohne Ausfälle bleiben außen vor — dort gibt es nichts zu finden.
    """
    params = params or load_params()
    gesamt_recall = gesamt(frame, params)["Recall"]

    zeilen = []
    for spalte in SCHEIBEN:
        tabelle = scheibe(frame, spalte, params)
        for name, zeile in tabelle.iterrows():
            if zeile["Zeilen"] < mindestgroesse or zeile["Ausfälle"] == 0:
                continue
            zeilen.append(
                {
                    "Scheibe": f"{spalte} = {name}",
                    "Zeilen": int(zeile["Zeilen"]),
                    "Ausfälle": int(zeile["Ausfälle"]),
                    "Recall": zeile["Recall"],
                    "Abstand zum Gesamt-Recall": zeile["Recall"] - gesamt_recall,
                    "FN": int(zeile["FN"]),
                    "Kosten EUR": zeile["Kosten EUR"],
                    "Kostenanteil": zeile["Kostenanteil"],
                }
            )
    return pd.DataFrame(zeilen).sort_values("Abstand zum Gesamt-Recall").reset_index(drop=True)


# ── Kommandozeile ───────────────────────────────────────────────────────


def _prozent(wert: float) -> str:
    return "—" if pd.isna(wert) else f"{wert:.1%}"


def _main() -> None:
    params = load_params()
    frame = ergaenze_baender(lade_bewertungsdaten(params))
    insgesamt = gesamt(frame, params)

    print(
        f"Validierungsmenge: {insgesamt['Zeilen']} Zeilen, "
        f"{insgesamt['Ausfälle']} Ausfälle, Schwellenwert {frame.attrs['schwelle']}"
    )
    print(
        f"Gesamt: Recall {insgesamt['Recall']:.3f}  "
        f"Precision {insgesamt['Precision']:.3f}  "
        f"F1 {insgesamt['F1']:.3f}  "
        f"Kosten {insgesamt['Kosten EUR']:,.0f} EUR".replace(",", ".")
    )

    print("\n── Nach Ausfallursache " + "─" * 50)
    ursachen = nach_ursache(frame)
    print(
        ursachen.to_string(
            formatters={
                "Recall": _prozent,
                "mittlere Wahrscheinlichkeit": "{:.3f}".format,
            }
        )
    )

    print("\n── Einzelscheiben " + "─" * 55)
    for spalte in SCHEIBEN:
        tabelle = scheibe(frame, spalte, params)
        print(f"\n{spalte}")
        print(
            tabelle.to_string(
                formatters={
                    "Ausfallrate": _prozent,
                    "Recall": _prozent,
                    "Precision": _prozent,
                    "F1": _prozent,
                    "Kostenanteil": _prozent,
                    "Kosten EUR": "{:,.0f}".format,
                }
            )
        )

    print("\n── Zuerst ansehen " + "─" * 55)
    auffaellig = auffaellige_scheiben(frame, params=params)
    print(
        auffaellig.head(6).to_string(
            index=False,
            formatters={
                "Recall": _prozent,
                "Abstand zum Gesamt-Recall": "{:+.3f}".format,
                "Kostenanteil": _prozent,
                "Kosten EUR": "{:,.0f}".format,
            },
        )
    )

    bericht = {
        "berechnet_auf": "validierungsmenge",
        "schwelle": frame.attrs["schwelle"],
        "mindestgroesse": MINDESTGROESSE,
        "gesamt": insgesamt,
        "nach_ursache": ursachen.reset_index().to_dict(orient="records"),
        "scheiben": {
            spalte: scheibe(frame, spalte, params).reset_index().to_dict(orient="records")
            for spalte in SCHEIBEN
        },
        "zuerst_ansehen": auffaellig.head(10).to_dict(orient="records"),
    }
    pfad = PATHS.reports / BERICHT
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(
        json.dumps(bericht, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\ngeschrieben: {PATHS.relativ(pfad)}")


if __name__ == "__main__":
    _main()
