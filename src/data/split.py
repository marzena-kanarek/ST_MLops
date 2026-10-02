"""Leakage entfernen und Daten aufteilen (Etappe 6).

Zwei Aufgaben, die zusammengehören und in dieser Reihenfolge erledigt werden
müssen:

1. **Ursachenspalten entfernen.** ``TWF``, ``HDF``, ``PWF``, ``OSF`` und ``RNF``
   beschreiben die Ursache eines Ausfalls. Sie entstehen gemeinsam mit dem
   Ausfall und sind zum Vorhersagezeitpunkt nicht bekannt — Data Leakage.
   Ebenfalls raus: ``udi`` und ``product_id``, reine Kennungen ohne Information.

2. **Dreiteilig aufteilen.** Training zum Lernen, Validierung zum Auswählen von
   Modell, Hyperparametern und Schwellenwert, Test für genau eine Messung ganz
   am Ende.

Geschichtet wird nach der Zielgröße (``stratify``): bei 3,4 % positiven Fällen
könnte eine zufällige Aufteilung sonst eine Teilmenge mit kaum Ausfällen
erzeugen, und alle Kennzahlen darauf wären Rauschen.

Eine gruppenweise oder zeitliche Aufteilung ist hier **nicht** nötig: jede Zeile
trägt eine eigene ``product_id`` und es gibt keinen Zeitstempel (belegt in
Notebook 01, Abschnitt 9).

Aufruf von der Kommandozeile::

    python -m src.data.split
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATHS, load_params


def entferne_leakage_und_kennungen(
    frame: pd.DataFrame,
    params: dict | None = None,
) -> pd.DataFrame:
    """Entfernt Ursachenspalten und Kennungen.

    Args:
        frame: DataFrame mit technischen Spaltennamen.
        params: Inhalt von params.yaml. Standard: wird geladen.

    Returns:
        Neuer DataFrame ohne Ursachenspalten und ohne Kennungen.
    """
    params = params or load_params()
    zu_entfernen = [
        *params["data"]["leakage_columns"],
        *params["data"]["id_columns"],
    ]
    vorhanden = [s for s in zu_entfernen if s in frame.columns]
    return frame.drop(columns=vorhanden)


def split_data(
    frame: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Teilt einen DataFrame geschichtet in Training, Validierung und Test.

    Die Anteile stehen in params.yaml. Aufgeteilt wird in zwei Schritten: erst
    wird die Testmenge abgetrennt, dann aus dem Rest die Validierungsmenge.

    Returns:
        (train, val, test) — die Anteile beziehen sich jeweils auf den
        vollständigen Datensatz.
    """
    params = params or load_params()
    ziel = params["data"]["target"]
    seed = params["seed"]
    test_anteil = params["data"]["test_size"]
    val_anteil = params["data"]["validation_size"]

    train_val, test = train_test_split(
        frame,
        test_size=test_anteil,
        stratify=frame[ziel],
        random_state=seed,
    )
    # Anteil auf den verbleibenden Rest umrechnen, damit die Validierungsmenge
    # denselben Anteil am Gesamtdatensatz hat wie die Testmenge.
    val_anteil_vom_rest = val_anteil / (1 - test_anteil)
    train, val = train_test_split(
        train_val,
        test_size=val_anteil_vom_rest,
        stratify=train_val[ziel],
        random_state=seed,
    )
    return train, val, test


def uebersicht(
    teilmengen: dict[str, pd.DataFrame],
    ziel: str,
) -> pd.DataFrame:
    """Stellt Größe und Klassenverhältnis der Teilmengen gegenüber."""
    zeilen = []
    gesamt = sum(len(t) for t in teilmengen.values())
    for name, teil in teilmengen.items():
        zeilen.append(
            {
                "Teilmenge": name,
                "Zeilen": len(teil),
                "Anteil": len(teil) / gesamt,
                "Ausfälle": int(teil[ziel].sum()),
                "Ausfallrate": float(teil[ziel].mean()),
            }
        )
    return pd.DataFrame(zeilen).set_index("Teilmenge")


def schreibe_teilmengen(
    teilmengen: dict[str, pd.DataFrame],
    verzeichnis: Path | None = None,
) -> dict[str, Path]:
    """Legt die Teilmengen als Parquet unter data/processed/ ab.

    Parquet statt CSV, weil Datentypen erhalten bleiben — ein aus CSV gelesenes
    ``type`` wäre wieder ein beliebiger Text, und Ganzzahlen könnten zu
    Fließkomma werden.
    """
    verzeichnis = verzeichnis or PATHS.processed
    verzeichnis.mkdir(parents=True, exist_ok=True)
    pfade = {}
    for name, teil in teilmengen.items():
        pfad = verzeichnis / f"{name}.parquet"
        teil.to_parquet(pfad, index=False)
        pfade[name] = pfad
    return pfade


def lade_teilmengen(
    verzeichnis: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Liest die in Etappe 6 abgelegten Teilmengen wieder ein.

    Damit müssen spätere Notebooks nicht erneut aufteilen — sie arbeiten auf
    genau denselben Zeilen wie alle anderen.

    Raises:
        FileNotFoundError: wenn die Teilmengen fehlen; dann erst
            ``python -m src.data.split`` ausführen.
    """
    verzeichnis = verzeichnis or PATHS.processed
    teile = []
    for name in ("train", "val", "test"):
        pfad = verzeichnis / f"{name}.parquet"
        if not pfad.exists():
            raise FileNotFoundError(
                f"{pfad} fehlt. Bitte zuerst 'python -m src.data.split' ausführen."
            )
        teile.append(pd.read_parquet(pfad))
    return tuple(teile)  # type: ignore[return-value]


def trenne_merkmale_und_ziel(
    frame: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Zerlegt eine Teilmenge in Merkmale (X) und Zielgröße (y)."""
    params = params or load_params()
    ziel = params["data"]["target"]
    return frame.drop(columns=[ziel]), frame[ziel]


def _main() -> None:
    """Erzeugt die drei Teilmengen aus den Rohdaten und legt sie ab."""
    from src.data.load import load_raw_data
    from src.data.validate import validate_dataframe

    params = load_params()
    ziel = params["data"]["target"]

    roh = load_raw_data()
    validate_dataframe(roh).raise_if_invalid()

    merkmale = entferne_leakage_und_kennungen(roh, params)
    entfernt = sorted(set(roh.columns) - set(merkmale.columns))
    print("Entfernte Spalten:", ", ".join(entfernt))
    print("Verbleibende Spalten:", ", ".join(merkmale.columns))
    print()

    train, val, test = split_data(merkmale, params)
    tabelle = uebersicht({"train": train, "val": val, "test": test}, ziel)
    print(
        tabelle.to_string(
            formatters={
                "Anteil": "{:.1%}".format,
                "Ausfallrate": "{:.2%}".format,
            }
        )
    )
    print()

    pfade = schreibe_teilmengen({"train": train, "val": val, "test": test})
    for pfad in pfade.values():
        print(f"geschrieben: {PATHS.relativ(pfad)}")


if __name__ == "__main__":
    _main()
