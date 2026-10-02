"""Tests der Protokollierung (Etappe 17).

Alle Tests schreiben in ein Wegwerfverzeichnis (``tmp_path``) und nicht in
``reports/`` — eine Testsuite, die echte Betriebsprotokolle vollschreibt, ist
ihr eigenes Problem.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest
from fastapi.testclient import TestClient

from src.serving.prediction_log import Vorhersageprotokoll, aus_params
from src.utils import logging_setup

GUELTIG = {
    "type": "L",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}
RISKANT = {**GUELTIG, "tool_wear_min": 215, "torque_nm": 64.0, "rotational_speed_rpm": 1320}


@pytest.fixture
def protokoll(tmp_path) -> Vorhersageprotokoll:
    """Ein frisches Protokoll in einem Wegwerfverzeichnis."""
    return Vorhersageprotokoll(
        jsonl_pfad=tmp_path / "predictions.jsonl",
        sqlite_pfad=tmp_path / "predictions.db",
    )


def beispieleintrag(protokoll: Vorhersageprotokoll, **abweichend):
    vorgabe = {
        "request_id": "11111111-2222-3333-4444-555555555555",
        "zeile": 0,
        "eingabe": GUELTIG,
        "wahrscheinlichkeit": 0.42,
        "alarm": True,
        "schwelle": 0.06,
        "modell": {"name": "random_forest", "sha256": "a" * 64},
        "hinweise": [],
        "dauer_ms": 12.5,
    }
    vorgabe.update(abweichend)
    return protokoll.eintrag(**vorgabe)


# ── Der Eintrag ─────────────────────────────────────────────────────────


def test_eintrag_enthaelt_alles_was_verlangt_ist(protokoll) -> None:
    """Zeitstempel, Kennung, Eingangswerte, Wahrscheinlichkeit, Entscheidung,
    Schwellenwert, Modell mit Version, Antwortzeit, Warnungen."""
    eintrag = beispieleintrag(protokoll)
    for feld in (
        "timestamp",
        "request_id",
        "row",
        "probability",
        "alarm",
        "threshold",
        "model_name",
        "model_sha256",
        "latency_ms",
        "warnings",
        "inputs",
    ):
        assert feld in eintrag, f"Feld '{feld}' fehlt im Protokolleintrag"
    assert eintrag["inputs"] == GUELTIG
    assert eintrag["timestamp"].endswith("+00:00")  # Zeitzone mitgeschrieben


def test_eingaben_lassen_sich_abschalten(tmp_path) -> None:
    """Der Datenschutzschalter: ohne Eingangswerte protokollieren."""
    ohne = Vorhersageprotokoll(
        jsonl_pfad=tmp_path / "p.jsonl", sqlite_pfad=None, mit_eingaben=False
    )
    assert beispieleintrag(ohne)["inputs"] is None


# ── JSONL ───────────────────────────────────────────────────────────────


def test_jsonl_haengt_eine_zeile_je_vorhersage_an(protokoll) -> None:
    for nummer in range(3):
        protokoll.schreibe([beispieleintrag(protokoll, zeile=nummer)])

    zeilen = protokoll.jsonl_pfad.read_text(encoding="utf-8").strip().splitlines()
    assert len(zeilen) == 3
    for zeile in zeilen:
        json.loads(zeile)  # jede Zeile ist für sich gültiges JSON


# ── SQLite ──────────────────────────────────────────────────────────────


def test_sqlite_nimmt_die_eintraege_auf(protokoll) -> None:
    protokoll.schreibe([beispieleintrag(protokoll, zeile=n) for n in range(3)])
    assert protokoll.anzahl() == 3
    jüngster = protokoll.zeilen(1)[0]
    assert jüngster["model_sha256"] == "a" * 64
    assert json.loads(jüngster["inputs"])["type"] == "L"


def test_tagesstatistik_rechnet_richtig(protokoll) -> None:
    """Die Auswertung aus dem Leitfaden: Anfragen, Alarmquote, Antwortzeit je Tag."""
    protokoll.schreibe(
        [
            beispieleintrag(protokoll, zeile=0, alarm=True, dauer_ms=10.0),
            beispieleintrag(protokoll, zeile=1, alarm=False, dauer_ms=20.0),
            beispieleintrag(protokoll, zeile=2, alarm=False, dauer_ms=30.0),
        ]
    )
    statistik = protokoll.tagesstatistik()
    assert len(statistik) == 1
    assert statistik[0]["anfragen"] == 3
    assert statistik[0]["alarmquote"] == pytest.approx(1 / 3)
    assert statistik[0]["antwortzeit_ms"] == pytest.approx(20.0)


def test_unbrauchbare_datenbank_faellt_auf_jsonl_zurueck(tmp_path, monkeypatch) -> None:
    """Auf Netzlaufwerken kann SQLite nicht sperren — der Dienst darf nicht scheitern."""

    def streikt(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(sqlite3, "connect", streikt)
    p = Vorhersageprotokoll(jsonl_pfad=tmp_path / "p.jsonl", sqlite_pfad=tmp_path / "p.db")
    assert p.sqlite_pfad is None  # abgeschaltet, nicht abgestürzt
    assert p.schreibe([beispieleintrag(p)]) == 1  # JSONL schreibt weiter


def test_schreibfehler_wird_gemeldet_nicht_geworfen(protokoll, tmp_path) -> None:
    """Eine volle Platte ist kein Grund, eine Ausnahme nach oben zu geben."""
    protokoll.jsonl_pfad = tmp_path  # ein Verzeichnis, keine Datei
    assert protokoll.schreibe([beispieleintrag(protokoll)]) == 0


# ── Einrichtung des Anwendungsprotokolls ────────────────────────────────


def test_protokollstufe_kommt_aus_der_umgebung(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert logging_setup.stufe() == logging.WARNING
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert logging_setup.stufe() == logging.DEBUG


def test_logger_hat_handler(monkeypatch) -> None:
    """Nach der Einrichtung schreibt der Logger wirklich irgendwohin."""
    logging_setup.richte_ein(mit_datei=False, erneut=True)
    assert logging.getLogger().handlers
    logging_setup.richte_ein(mit_datei=False, erneut=True)  # zweimal ist harmlos
    assert len(logging.getLogger().handlers) == 1


# ── Zusammenspiel mit der Schnittstelle ─────────────────────────────────


@pytest.fixture
def klient_mit_protokoll(tmp_path):
    """Client, dessen Protokoll in ein Wegwerfverzeichnis schreibt."""
    import src.serving.api as api
    from src.serving.model_service import modelldatei

    if not modelldatei().exists():
        pytest.skip("Kein Modell vorhanden: python -m src.pipelines.run_pipeline")

    with TestClient(api.app) as c:
        api.STATE["protokoll"] = Vorhersageprotokoll(
            jsonl_pfad=tmp_path / "predictions.jsonl",
            sqlite_pfad=tmp_path / "predictions.db",
        )
        api.setze_kennzahlen_zurueck()
        yield c, api, api.STATE["protokoll"]


def test_drei_anfragen_drei_eintraege_und_zaehler_drei(klient_mit_protokoll) -> None:
    """Die Abschlussbedingung des Leitfadens, als Test festgehalten."""
    klient, api, protokoll = klient_mit_protokoll

    for _ in range(3):
        assert klient.post("/predict", json=GUELTIG).status_code == 200

    assert protokoll.anzahl() == 3
    assert len(protokoll.jsonl_pfad.read_text(encoding="utf-8").strip().splitlines()) == 3
    assert "pdm_predictions_total 3" in klient.get("/metrics").text
    assert "pdm_logged_total 3" in klient.get("/metrics").text


def test_stapel_erzeugt_einen_eintrag_je_zeile(klient_mit_protokoll) -> None:
    """Gleiche Anfrage-Kennung, eine Zeile je Betriebspunkt — für die Driftprüfung."""
    klient, _, protokoll = klient_mit_protokoll

    antwort = klient.post("/predict/batch", json={"items": [GUELTIG, RISKANT, GUELTIG]})
    kennung = antwort.json()["request_id"]

    assert protokoll.anzahl() == 3
    eintraege = protokoll.zeilen(3)
    assert {e["request_id"] for e in eintraege} == {kennung}
    assert sorted(e["row"] for e in eintraege) == [0, 1, 2]
    assert sum(e["alarm"] for e in eintraege) == 1


def test_protokollierte_werte_stimmen_mit_der_antwort_ueberein(klient_mit_protokoll) -> None:
    klient, _, protokoll = klient_mit_protokoll
    antwort = klient.post("/predict", json=RISKANT).json()
    eintrag = protokoll.zeilen(1)[0]

    assert eintrag["request_id"] == antwort["request_id"]
    assert eintrag["probability"] == antwort["failure_probability"]
    assert bool(eintrag["alarm"]) is antwort["alarm"]
    assert eintrag["threshold"] == antwort["threshold"]
    assert eintrag["model_sha256"] == antwort["model"]["sha256"]


def test_hinweise_landen_im_protokoll(klient_mit_protokoll) -> None:
    klient, _, protokoll = klient_mit_protokoll
    klient.post("/predict", json={**GUELTIG, "torque_nm": 140.0})
    hinweise = json.loads(protokoll.zeilen(1)[0]["warnings"])
    assert any("torque_nm" in h for h in hinweise)


def test_protokollfehler_verhindert_keine_antwort(klient_mit_protokoll, monkeypatch) -> None:
    """Der Aufrufer bekommt seine Vorhersage, auch wenn das Protokoll streikt."""
    klient, api, protokoll = klient_mit_protokoll

    def streikt(self, eintraege):
        raise OSError("Platte voll (Test)")

    monkeypatch.setattr(Vorhersageprotokoll, "schreibe", streikt)
    assert klient.post("/predict", json=GUELTIG).status_code == 200
    assert "pdm_log_failures_total 1" in klient.get("/metrics").text


def test_protokoll_aus_params_liest_die_konfiguration() -> None:
    """Die Pfade stehen in params.yaml, nicht im Code."""
    p = aus_params()
    assert p.jsonl_pfad is not None
    assert p.jsonl_pfad.name == "predictions.jsonl"
