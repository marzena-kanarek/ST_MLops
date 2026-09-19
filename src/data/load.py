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

# Lage der Datei -> Projektwurzel. src/data/load.py: parents[0]=data, [1]=src, [2]=Wurzel.
# Ab Etappe 14 kommt das aus src/config.py (PATHS.raw); bis dahin steht es hier.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

# --- Eingefrorener Zustand der Rohdaten -------------------------------------
# EXPECTED_SHA256 wird einmalig mit dem Wert aus "python -m src.data.load"
# gefuellt und danach nicht mehr veraendert. Aendert er sich, ist die Rohdatei
# nicht mehr dieselbe - dann ist keine Kennzahl mehr mit frueheren vergleichbar.
EXPECTED_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"
EXPECTED_ROW_COUNT = 10_000

# --- Spaltennamen -----------------------------------------------------------
# Originalnamen enthalten Leerzeichen und Einheiten in Klammern. Das raecht sich
# in jeder Formel, in pandas-Query-Ausdruecken und in jedem JSON-Schema der
# spaeteren Schnittstelle, deshalb wird beim Laden umbenannt.
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

#: Zielgroesse der binaeren Klassifikation.
TARGET_COLUMN = "machine_failure"

#: Ursachenspalten des Ausfalls. Sie entstehen *mit* dem Ausfall und sind zum
#: Vorhersagezeitpunkt nicht bekannt -> Data Leakage. Entfernt werden sie erst
#: in Etappe 6, nicht hier: data/raw bleibt vollstaendig und unveraendert.
LEAKAGE_COLUMNS: tuple[str, ...] = ("twf", "hdf", "pwf", "osf", "rnf")


def datei_hash(pfad: Path, blockgroesse: int = 65_536) -> str:
    """Berechnet den SHA-256-Hash einer Datei blockweise.

    Blockweise, damit auch grosse Dateien nicht vollstaendig in den Arbeits-
    speicher geladen werden muessen.
    """
    digest = hashlib.sha256()
    with open(pfad, "rb") as handle:
        for block in iter(lambda: handle.read(blockgroesse), b""):
            digest.update(block)
    return digest.hexdigest()


def pruefe_integritaet(pfad: Path = RAW_DATA_PATH, erwartet: str = EXPECTED_SHA256) -> str:
    """Prueft, ob die Rohdatei noch die eingefrorene Datei ist.

    Gibt den tatsaechlichen Hash zurueck. Ist ``erwartet`` leer, wird nur
    berechnet und nicht geprueft (Zustand vor dem ersten Eintragen).

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
            "Die Rohdatei stimmt nicht mehr mit dem eingefrorenen Stand ueberein.\n"
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
        integritaet_pruefen: Wenn True, wird vor dem Laden der SHA-256-Hash geprueft.

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
    """Gibt Hash und Eckdaten der Rohdatei aus - Grundlage fuer Etappe 3."""
    tatsaechlich = pruefe_integritaet(erwartet="")  # nur berechnen, nicht vergleichen
    frame = load_raw_data(integritaet_pruefen=False)
    anteil = frame[TARGET_COLUMN].mean()

    print(f"Datei:            {RAW_DATA_PATH}")
    print(f"Groesse:          {RAW_DATA_PATH.stat().st_size:,} Byte")
    print(f"SHA-256:          {tatsaechlich}")
    print(f"Zeilen x Spalten: {frame.shape[0]} x {frame.shape[1]}")
    print(f"Positive Faelle:  {int(frame[TARGET_COLUMN].sum())} ({anteil:.2%})")
    print()
    print("Diesen Hash in EXPECTED_SHA256 und in references/datenbeschreibung.md eintragen.")


if __name__ == "__main__":
    _main()
