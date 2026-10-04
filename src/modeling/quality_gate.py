"""Qualitätsschranke: Mindestanforderungen ausführbar machen.

**Warum das nötig ist:** Ohne Schranke überschreibt irgendwann ein schlechterer
Lauf das gute Modell — unbemerkt, weil niemand jedes Mal die Kennzahlen prüft.
Die Schranke macht aus einer besprochenen Anforderung eine ausgeführte.

Die Grenzwerte stehen in ``params.yaml`` und können dort geändert werden, ohne
Code anzufassen.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchrankenErgebnis:
    """Ergebnis der Prüfung: bestanden oder nicht, und warum nicht."""

    verstoesse: list[str] = field(default_factory=list)
    geprueft: list[str] = field(default_factory=list)

    @property
    def bestanden(self) -> bool:
        return not self.verstoesse

    def bericht(self) -> str:
        """Menschenlesbarer Bericht für Kommandozeile und Notebook."""
        zeilen = [f"{len(self.geprueft)} Anforderungen geprüft."]
        zeilen += [
            f"  erfüllt: {g}"
            for g in self.geprueft
            if g not in [v.split(" — ")[0] for v in self.verstoesse]
        ]
        if self.verstoesse:
            zeilen.append(f"\nNICHT ERFÜLLT ({len(self.verstoesse)}):")
            zeilen += [f"  - {v}" for v in self.verstoesse]
        else:
            zeilen.append("\nAlle Anforderungen erfüllt — das Modell darf gespeichert werden.")
        return "\n".join(zeilen)

    def raise_if_failed(self) -> None:
        if self.verstoesse:
            raise ValueError(
                "Qualitätsschranke nicht bestanden:\n- " + "\n- ".join(self.verstoesse)
            )


def pruefe_qualitaet(
    kennzahlen: dict[str, float],
    schranken: dict[str, float],
) -> SchrankenErgebnis:
    """Prüft Kennzahlen gegen die Mindestanforderungen aus params.yaml.

    Args:
        kennzahlen: Ergebnis eines Trainingslaufs (``pr_auc``, ``recall``,
            ``kosten_eur``).
        schranken: Abschnitt ``quality_gate`` aus params.yaml.

    Returns:
        SchrankenErgebnis mit allen Verstößen — nicht nur dem ersten, damit man
        in einem Durchgang sieht, was alles fehlt.
    """
    ergebnis = SchrankenErgebnis()

    mindestens = [
        ("pr_auc", "min_pr_auc", "PR-AUC"),
        ("recall", "min_recall", "Recall"),
    ]
    for kennzahl, schranke, bezeichnung in mindestens:
        if schranke not in schranken:
            continue
        grenze = schranken[schranke]
        ergebnis.geprueft.append(bezeichnung)
        wert = kennzahlen.get(kennzahl)
        if wert is None:
            ergebnis.verstoesse.append(f"{bezeichnung} — Kennzahl fehlt")
        elif wert < grenze:
            ergebnis.verstoesse.append(
                f"{bezeichnung} — {wert:.3f} liegt unter der Mindestanforderung {grenze}"
            )

    if "max_expected_cost_eur" in schranken:
        grenze = schranken["max_expected_cost_eur"]
        ergebnis.geprueft.append("erwartete Kosten")
        wert = kennzahlen.get("kosten_eur")
        if wert is None:
            ergebnis.verstoesse.append("erwartete Kosten — Kennzahl fehlt")
        elif wert > grenze:
            ergebnis.verstoesse.append(
                f"erwartete Kosten — {wert:,.0f} EUR übersteigen die Obergrenze "
                f"{grenze:,.0f} EUR".replace(",", ".")
            )

    return ergebnis
