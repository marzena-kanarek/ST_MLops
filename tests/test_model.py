"""Modelltests: Form, Wertebereich, Verhalten (Etappe 13, Testart 3)

Besonders wertvoll sind die **Verhaltenstests**: Sie prüfen nicht eine exakte
Zahl, sondern eine fachliche Erwartung — mehr Verschleiß soll das Risiko
erhöhen. Solche Tests überleben ein Neutraining, während
``assert pr_auc == 0.8604`` bei jedem Lauf bricht.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split import trenne_merkmale_und_ziel

# ── Form und Wertebereich ───────────────────────────────────────────────


def test_wahrscheinlichkeiten_liegen_zwischen_null_und_eins(
    trainiertes_kleinstmodell, teilmengen, params
):
    _, val = teilmengen
    X_val, _ = trenne_merkmale_und_ziel(val, params)
    p = trainiertes_kleinstmodell.predict_proba(X_val)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_vorhersage_hat_eine_zeile_je_eingabezeile(trainiertes_kleinstmodell, teilmengen, params):
    _, val = teilmengen
    X_val, _ = trenne_merkmale_und_ziel(val, params)
    assert len(trainiertes_kleinstmodell.predict(X_val)) == len(X_val)


def test_spaltenreihenfolge_ist_egal(trainiertes_kleinstmodell, basiszustand):
    """Die Pipeline spricht Spalten über Namen an, nicht über Positionen."""
    vertauscht = basiszustand[list(reversed(basiszustand.columns))]
    original = trainiertes_kleinstmodell.predict_proba(basiszustand)[0, 1]
    gedreht = trainiertes_kleinstmodell.predict_proba(vertauscht)[0, 1]
    assert original == pytest.approx(gedreht)


def test_unbekannte_kategorie_bricht_nicht_ab(trainiertes_kleinstmodell, basiszustand):
    """handle_unknown='ignore' im OneHotEncoder — eine neue Qualitätsvariante
    darf den Dienst nicht abstürzen lassen."""
    fremd = basiszustand.copy()
    fremd["type"] = "X"
    p = trainiertes_kleinstmodell.predict_proba(fremd)[0, 1]
    assert 0.0 <= p <= 1.0


# ── Verhalten ───────────────────────────────────────────────────────────


def test_hoeherer_verschleiss_erhoeht_das_risiko(risiko, basiszustand):
    hoch = basiszustand.copy()
    hoch["tool_wear_min"] = 240
    assert risiko(hoch) > risiko(basiszustand)


def test_hoeheres_drehmoment_bei_niedriger_drehzahl_erhoeht_das_risiko(risiko, basiszustand):
    """Der Betriebspunkt oben links im Streudiagramm aus Notebook 01."""
    ueberlastet = basiszustand.copy()
    ueberlastet["torque_nm"] = 68.0
    ueberlastet["rotational_speed_rpm"] = 1300
    assert risiko(ueberlastet) > risiko(basiszustand)


def test_geringe_waermeabfuhr_erhoeht_das_risiko(risiko, basiszustand):
    """Kleine Temperaturdifferenz bedeutet schlechte Wärmeabfuhr — die Logik
    hinter der Ausfallart HDF."""
    schlecht_gekuehlt = basiszustand.copy()
    schlecht_gekuehlt["process_temperature_k"] = 308.4  # Differenz 8,4 K statt 10
    assert risiko(schlecht_gekuehlt) > risiko(basiszustand)


def test_identische_eingabe_liefert_identische_ausgabe(risiko, basiszustand):
    """Reproduzierbarkeit: zweimal dieselbe Frage, zweimal dieselbe Antwort."""
    assert risiko(basiszustand) == risiko(basiszustand.copy())


# ── Qualitätsschranke ───────────────────────────────────────────────────


def test_schranke_laesst_gutes_modell_durch(params):
    from src.modeling.quality_gate import pruefe_qualitaet

    gut = {"pr_auc": 0.86, "recall": 0.86, "kosten_eur": 126_200}
    assert pruefe_qualitaet(gut, params["quality_gate"]).bestanden


def test_schranke_haelt_schlechtes_modell_auf(params):
    from src.modeling.quality_gate import pruefe_qualitaet

    schlecht = {"pr_auc": 0.61, "recall": 0.42, "kosten_eur": 310_000}
    ergebnis = pruefe_qualitaet(schlecht, params["quality_gate"])
    assert not ergebnis.bestanden
    assert len(ergebnis.verstoesse) == 3  # alle Verstöße, nicht nur der erste


def test_schranke_prueft_jede_anforderung_einzeln(params):
    """Ein Modell, das nur am Recall scheitert, muss genau einen Verstoß melden."""
    from src.modeling.quality_gate import pruefe_qualitaet

    knapp = {"pr_auc": 0.80, "recall": 0.70, "kosten_eur": 150_000}
    ergebnis = pruefe_qualitaet(knapp, params["quality_gate"])
    assert not ergebnis.bestanden
    assert len(ergebnis.verstoesse) == 1
    assert "Recall" in ergebnis.verstoesse[0]


# ── Entscheidungsschwelle ───────────────────────────────────────────────


def test_kosten_steigen_wenn_ausfaelle_uebersehen_werden(params):
    from src.modeling.threshold import kosten_je_schwelle

    y = pd.Series([0, 0, 1, 1])
    # Modell erkennt beide Ausfälle sicher.
    p = np.array([0.01, 0.02, 0.90, 0.95])
    kurve = kosten_je_schwelle(y, p, params["costs"], schwellen=np.array([0.5, 0.99]))
    assert kurve.loc[0, "Kosten EUR"] < kurve.loc[1, "Kosten EUR"]


def test_kostenformel_rechnet_nachvollziehbar(params):
    from src.modeling.threshold import kosten_je_schwelle

    y = pd.Series([0, 1])
    p = np.array([0.9, 0.1])  # genau falsch herum: ein FP und ein FN
    kurve = kosten_je_schwelle(y, p, params["costs"], schwellen=np.array([0.5]))
    erwartet = params["costs"]["false_negative_eur"] + params["costs"]["false_positive_eur"]
    assert kurve.loc[0, "Kosten EUR"] == pytest.approx(erwartet)
