"""Datenvertrag und Validierung (Etappe 4).

Der Vertrag selbst steht in ``params.yaml``, nicht hier. Dieses Modul liest ihn
und prüft einen DataFrame dagegen.

Der Unterschied zwischen Fehler und Warnung ist der Kern:

* **Fehler** — eine Zusage ist gebrochen. Eine Spalte fehlt, ein Wert ist
  physikalisch unmöglich, eine unbekannte Kategorie taucht auf. Der Lauf bricht ab.
* **Warnung** — ein Wert liegt außerhalb des üblichen Bereichs, ist aber möglich.
  Im Training toleriert; an der Schnittstelle wird er als Hinweis mitgegeben.

Aufruf von der Kommandozeile::

    python -m src.data.validate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PARAMS_PATH, PROJECT_ROOT, load_params

# load_params liegt seit Etappe 14 in src/config.py. Hier bleibt der Name
# verfuegbar, damit bestehende Importe in Notebooks und Modulen weiter gelten.
__all__ = [
    "PARAMS_PATH",
    "PROJECT_ROOT",
    "ValidationResult",
    "load_contract",
    "load_params",
    "validate_dataframe",
]


def load_contract(pfad: Path = PARAMS_PATH) -> dict[str, Any]:
    """Liest nur den Abschnitt data_contract aus params.yaml."""
    return load_params(pfad)["data_contract"]


@dataclass
class ValidationResult:
    """Ergebnis einer Prüfung: was ist gebrochen, was ist nur auffällig."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_run: int = 0

    @property
    def is_valid(self) -> bool:
        """True, wenn keine harte Zusage verletzt wurde."""
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Bricht mit einer lesbaren Meldung ab, wenn Fehler vorliegen."""
        if self.errors:
            raise ValueError("Datenvertrag verletzt:\n- " + "\n- ".join(self.errors))

    def report(self) -> str:
        """Menschenlesbarer Bericht, für Notebook und Kommandozeile."""
        zeilen = [f"{self.checks_run} Prüfungen ausgeführt."]
        if self.errors:
            zeilen.append(f"\nFEHLER ({len(self.errors)}):")
            zeilen += [f"  - {e}" for e in self.errors]
        if self.warnings:
            zeilen.append(f"\nWARNUNGEN ({len(self.warnings)}):")
            zeilen += [f"  - {w}" for w in self.warnings]
        if not self.errors and not self.warnings:
            zeilen.append("Keine Beanstandungen — der Datenvertrag ist eingehalten.")
        elif not self.errors:
            zeilen.append("\nKeine Fehler — der Datenvertrag ist eingehalten.")
        return "\n".join(zeilen)

    def __str__(self) -> str:
        return self.report()


def _pruefe_numerisch(
    frame: pd.DataFrame,
    spalte: str,
    regeln: dict[str, Any],
    ergebnis: ValidationResult,
) -> None:
    """Typ, harte Grenzen und weiche Grenzen einer numerischen Spalte."""
    werte = frame[spalte]

    if not pd.api.types.is_numeric_dtype(werte):
        ergebnis.errors.append(
            f"'{spalte}': erwartet wurde ein numerischer Typ, vorhanden ist {werte.dtype}"
        )
        return

    einheit = regeln.get("einheit", "")
    suffix = f" {einheit}" if einheit else ""

    # Harte Grenzen -> Fehler
    if (untergrenze := regeln.get("min")) is not None:
        verstoesse = werte < untergrenze
        ergebnis.checks_run += 1
        if verstoesse.any():
            ergebnis.errors.append(
                f"'{spalte}': {int(verstoesse.sum())} Wert(e) unter der physikalischen "
                f"Untergrenze {untergrenze}{suffix} (kleinster: {werte.min()})"
            )
    if (obergrenze := regeln.get("max")) is not None:
        verstoesse = werte > obergrenze
        ergebnis.checks_run += 1
        if verstoesse.any():
            ergebnis.errors.append(
                f"'{spalte}': {int(verstoesse.sum())} Wert(e) über der physikalischen "
                f"Obergrenze {obergrenze}{suffix} (größter: {werte.max()})"
            )

    # Weiche Grenzen -> Warnung
    if (warn_unten := regeln.get("warn_min")) is not None:
        auffaellig = werte < warn_unten
        ergebnis.checks_run += 1
        if auffaellig.any():
            ergebnis.warnings.append(
                f"'{spalte}': {int(auffaellig.sum())} Wert(e) unter dem üblichen "
                f"Bereich (< {warn_unten}{suffix}, kleinster: {werte.min()})"
            )
    if (warn_oben := regeln.get("warn_max")) is not None:
        auffaellig = werte > warn_oben
        ergebnis.checks_run += 1
        if auffaellig.any():
            ergebnis.warnings.append(
                f"'{spalte}': {int(auffaellig.sum())} Wert(e) über dem üblichen "
                f"Bereich (> {warn_oben}{suffix}, größter: {werte.max()})"
            )


def _pruefe_kategorial(
    frame: pd.DataFrame,
    spalte: str,
    regeln: dict[str, Any],
    ergebnis: ValidationResult,
) -> None:
    """Erlaubte Ausprägungen einer kategorialen Spalte."""
    erlaubt = regeln.get("allowed")
    if erlaubt is None:
        return
    ergebnis.checks_run += 1
    vorhanden = set(frame[spalte].dropna().unique())
    unerwartet = vorhanden - set(erlaubt)
    if unerwartet:
        ergebnis.errors.append(
            f"'{spalte}': unerlaubte Ausprägung(en) {sorted(map(str, unerwartet))}, "
            f"erlaubt sind {erlaubt}"
        )


def validate_dataframe(
    frame: pd.DataFrame,
    contract: dict[str, Any] | None = None,
) -> ValidationResult:
    """Prueft einen DataFrame gegen den Datenvertrag.

    Args:
        frame: zu prüfender DataFrame mit technischen Spaltennamen.
        contract: Vertrag als dict. Standard: Abschnitt data_contract aus params.yaml.

    Returns:
        ValidationResult mit Fehlern, Warnungen und Anzahl der Prüfungen.
    """
    if contract is None:
        contract = load_contract()

    ergebnis = ValidationResult()
    spaltenregeln: dict[str, dict[str, Any]] = contract.get("columns", {})

    # 1) Pflichtspalten vorhanden?
    for spalte, regeln in spaltenregeln.items():
        ergebnis.checks_run += 1
        if spalte not in frame.columns:
            if regeln.get("required", False):
                ergebnis.errors.append(f"Pflichtspalte '{spalte}' fehlt")
            continue

        # 2) Fehlende Werte
        anteil_fehlend = float(frame[spalte].isna().mean())
        max_fehlend = float(contract.get("max_missing_ratio", 0.0))
        ergebnis.checks_run += 1
        if anteil_fehlend > max_fehlend:
            ergebnis.errors.append(
                f"'{spalte}': {anteil_fehlend:.1%} fehlende Werte, "
                f"erlaubt sind hoechstens {max_fehlend:.1%}"
            )

        # 3) Typ- und Wertebereichsprüfungen
        if regeln.get("kind") == "numeric":
            _pruefe_numerisch(frame, spalte, regeln, ergebnis)
        elif regeln.get("kind") == "categorical":
            _pruefe_kategorial(frame, spalte, regeln, ergebnis)

    # 4) Zeilenzahl (nur für die vollständige Rohdatei sinnvoll)
    if contract.get("check_row_count", False):
        erwartet = contract.get("expected_row_count")
        ergebnis.checks_run += 1
        if erwartet is not None and len(frame) != erwartet:
            ergebnis.errors.append(
                f"Zeilenzahl {len(frame)} weicht von den erwarteten {erwartet} ab"
            )

    # 5) Unbekannte Spalten -> nur Hinweis
    if contract.get("warn_on_unexpected_columns", False):
        bekannt = set(spaltenregeln) | set(load_params()["data"]["id_columns"])
        bekannt |= set(load_params()["data"]["leakage_columns"])
        unbekannt = sorted(set(frame.columns) - bekannt)
        ergebnis.checks_run += 1
        if unbekannt:
            ergebnis.warnings.append(f"Nicht im Vertrag beschriebene Spalten: {unbekannt}")

    return ergebnis


def _main() -> None:
    """Prueft die Rohdaten gegen den Vertrag und gibt den Bericht aus."""
    from src.data.load import load_raw_data

    frame = load_raw_data()
    vertrag = load_contract()
    vertrag["check_row_count"] = True  # bei der Rohdatei prüfen wir sie mit
    ergebnis = validate_dataframe(frame, vertrag)

    print(f"Geprüft: {len(frame)} Zeilen x {frame.shape[1]} Spalten")
    print(ergebnis.report())
    print()
    print("gültig:", ergebnis.is_valid)


if __name__ == "__main__":
    _main()
