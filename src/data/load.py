"""Laden und Einfrieren der Rohdaten.

Dies ist die **einzige** Stelle im Projekt, die die Rohdatei liest. Alle anderen
Module und Notebooks rufen ``load_raw_data()`` auf, statt selbst ein
``pd.read_csv`` zu schreiben.

Aufruf von der Kommandozeile, um den Hash der Rohdatei einmalig zu bestimmen::

    python -m src.data.load
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.config import PATHS, load_params

# Alle Werte dieses Moduls stammen aus params.yaml. Hier steht keine Zahl und
# kein Pfad mehr (Etappe 14).
_PARAMS = load_params()

#: Die eingefrorene Rohdatei. Pfad aus params.yaml, Abschnitt ``paths``.
RAW_DATA_PATH = PATHS.raw_file

# --- Eingefrorener Zustand der Rohdaten -------------------------------------
#: SHA-256 der eingefrorenen Rohdatei (params.yaml: ``data.raw_sha256``).
#: Einmalig mit dem Wert aus "python -m src.data.load" gefuellt und danach nicht
#: mehr verändert. Ändert er sich, ist die Rohdatei nicht mehr dieselbe - dann
#: ist keine Kennzahl mehr mit frueheren vergleichbar.
EXPECTED_SHA256 = _PARAMS["data"]["raw_sha256"]
EXPECTED_ROW_COUNT = _PARAMS["data_contract"]["expected_row_count"]

# --- Spaltennamen -----------------------------------------------------------
# Originalnamen enthalten Leerzeichen und Einheiten in Klammern. Das rächt sich
# in jeder Formel, in pandas-Query-Ausdruecken und in jedem JSON-Schema der
# späteren Schnittstelle, deshalb wird beim Laden umbenannt.
COLUMN_MAPPING: dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

#: Zielgröße der binaeren Klassifikation (params.yaml: ``data.target``).
TARGET_COLUMN = _PARAMS["data"]["target"]

#: Ursachenspalten des Ausfalls. Sie entstehen *mit* dem Ausfall und sind zum
#: Vorhersagezeitpunkt nicht bekannt -> Data Leakage. Entfernt werden sie erst
#: in Etappe 6, nicht hier: data/raw bleibt vollständig und unverändert.
LEAKAGE_COLUMNS: tuple[str, ...] = tuple(_PARAMS["data"]["leakage_columns"])


def datei_hash(pfad: Path, blockgroesse: int = 65_536) -> str:
    """Berechnet den SHA-256-Hash einer Datei blockweise.

    Blockweise, damit auch grosse Dateien nicht vollständig in den Arbeits-
    speicher geladen werden müssen.
    """
    digest = hashlib.sha256()
    with open(pfad, "rb") as handle:
        for block in iter(lambda: handle.read(blockgroesse), b""):
            digest.update(block)
    return digest.hexdigest()


def pruefe_integritaet(pfad: Path = RAW_DATA_PATH, erwartet: str = EXPECTED_SHA256) -> str:
    """Prueft, ob die Rohdatei noch die eingefrorene Datei ist.

    Gibt den tatsaechlichen Hash zurück. Ist ``erwartet`` leer, wird nur
    berechnet und nicht geprüft (Zustand vor dem ersten Eintragen).

    Raises:
        FileNotFoundError: wenn die Rohdatei fehlt.
        ValueError: wenn der Hash vom eingefrorenen Wert abweicht.
    """
    if not pfad.exists():
        raise FileNotFoundError(
            f"Rohdatei nicht gefunden: {pfad}\n"
            "Bitte den AI4I-2020-Datensatz herunterladen und unter data/raw/ ablegen "
            "(siehe references/datenbeschreibung.md)."
        )

    tatsaechlich = datei_hash(pfad)
    if erwartet and tatsaechlich != erwartet:
        raise ValueError(
            "Die Rohdatei stimmt nicht mehr mit dem eingefrorenen Stand überein.\n"
            f"  Datei:     {pfad}\n"
            f"  erwartet:  {erwartet}\n"
            f"  berechnet: {tatsaechlich}\n"
            "Kennzahlen aus frueheren Laeufen sind damit nicht mehr vergleichbar."
        )
    return tatsaechlich


def load_raw_data(
    pfad: Path = RAW_DATA_PATH,
    *,
    integritaet_pruefen: bool = True,
) -> pd.DataFrame:
    """Laedt die Rohdaten und benennt die Spalten auf technische Namen um.

    Args:
        pfad: Pfad zur Rohdatei. Standard ist die eingefrorene Datei unter data/raw/.
        integritaet_prüfen: Wenn True, wird vor dem Laden der SHA-256-Hash geprüft.

    Returns:
        DataFrame mit allen Originalspalten unter technischen Namen. Es werden
        weder Zeilen noch Spalten entfernt - das Aufbereiten beginnt in Etappe 6.
    """
    if integritaet_pruefen:
        pruefe_integritaet(pfad)

    # encoding ausdruecklich setzen; -sig entfernt ein eventuelles BOM.
    frame = pd.read_csv(pfad, encoding="utf-8-sig")

    fehlend = set(COLUMN_MAPPING) - set(frame.columns)
    if fehlend:
        raise ValueError("In der Rohdatei fehlen erwartete Spalten: " + ", ".join(sorted(fehlend)))

    frame = frame.rename(columns=COLUMN_MAPPING)

    if len(frame) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Unerwartete Zeilenzahl: {len(frame)} statt {EXPECTED_ROW_COUNT}.")

    return frame


def _main() -> None:
    """Gibt Hash und Eckdaten der Rohdatei aus - Grundlage für Etappe 3."""
    tatsaechlich = pruefe_integritaet(erwartet="")  # nur berechnen, nicht vergleichen
    frame = load_raw_data(integritaet_pruefen=False)
    anteil = frame[TARGET_COLUMN].mean()

    print(f"Datei:            {RAW_DATA_PATH}")
    print(f"Größe:            {RAW_DATA_PATH.stat().st_size:,} Byte")
    print(f"SHA-256:          {tatsaechlich}")
    print(f"Zeilen x Spalten: {frame.shape[0]} x {frame.shape[1]}")
    print(f"Positive Fälle:   {int(frame[TARGET_COLUMN].sum())} ({anteil:.2%})")
    print()
    print("Diesen Hash in params.yaml (data.raw_sha256) und in")
    print("references/datenbeschreibung.md eintragen.")


if __name__ == "__main__":
    _main()
