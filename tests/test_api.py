"""Tests der HTTP-Schnittstelle (Etappe 16).

Die vierte Testsorte neben Daten-, Merkmals- und Modelltests: Hier wird nicht
geprüft, ob das Modell gut rechnet, sondern ob die **Schnittstelle sich richtig
verhält** — richtige Statuscodes, vollständige Antworten, verständliche Fehler.

``TestClient`` startet keinen Webserver und öffnet keinen Netzwerkport. Er ruft
die Anwendung direkt auf, inklusive ``lifespan`` — deshalb steht er in einem
``with``-Block, sonst wäre kein Modell geladen.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.config import load_params
from src.serving.model_service import modelldatei

GUELTIG = {
    "type": "L",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}

#: Ein Betriebspunkt, der im Training klar zu Ausfällen gehörte: hoher
#: Verschleiß, hohes Drehmoment, niedrige Drehzahl.
RISKANT = {
    "type": "L",
    "air_temperature_k": 302.0,
    "process_temperature_k": 310.5,
    "rotational_speed_rpm": 1320,
    "torque_nm": 64.0,
    "tool_wear_min": 215,
}


@pytest.fixture(scope="module")
def klient(tmp_path_factory):
    """Ein Client mit geladenem Modell. Überspringt, wenn kein Modell vorliegt.

    Das Vorhersageprotokoll wird in ein Wegwerfverzeichnis umgelenkt: Eine
    Testsuite darf nicht in die echten Betriebsprotokolle unter ``reports/``
    schreiben, sonst stehen dort Testdaten neben echten Vorhersagen.
    """
    if not modelldatei().exists():
        pytest.skip("Kein Modell vorhanden: python -m src.pipelines.run_pipeline")
    import src.serving.api as api
    from src.serving.prediction_log import Vorhersageprotokoll

    verzeichnis = tmp_path_factory.mktemp("api_protokoll")
    with TestClient(api.app) as c:
        api.STATE["protokoll"] = Vorhersageprotokoll(
            jsonl_pfad=verzeichnis / "predictions.jsonl",
            sqlite_pfad=verzeichnis / "predictions.db",
        )
        yield c


# ── Betriebsendpunkte ───────────────────────────────────────────────────


def test_health_meldet_geladenes_modell(klient) -> None:
    antwort = klient.get("/health")
    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["model_loaded"] is True
    assert daten["status"] == "ok"
    assert daten["uptime_s"] >= 0


def test_model_info_nennt_schwelle_und_herkunft(klient) -> None:
    daten = klient.get("/model-info").json()
    params = load_params()

    assert daten["threshold"] == params["decision"]["threshold"]
    assert daten["registry_name"] == params["model"]["registry_name"]
    assert re.fullmatch(r"[0-9a-f]{64}", daten["model_sha256"])
    assert "type" in daten["features"]
    assert "temp_difference_k" in daten["engineered_features"]
    assert daten["metrics"]["pr_auc"] > 0


def test_abgeleitete_merkmale_sind_keine_eingabefelder(klient) -> None:
    """Die Pipeline rechnet sie selbst — eine Anfrage darf sie nicht mitbringen."""
    daten = klient.get("/model-info").json()
    assert not set(daten["features"]) & set(daten["engineered_features"])


def test_ursachenspalten_sind_keine_eingabefelder(klient) -> None:
    """Zum Vorhersagezeitpunkt unbekannt — eine API, die danach fragt, ist unbedienbar."""
    felder = set(klient.get("/model-info").json()["features"])
    assert not felder & {"twf", "hdf", "pwf", "osf", "rnf", "machine_failure"}


# ── Einzelvorhersage ────────────────────────────────────────────────────


def test_vorhersage_ist_vollstaendig(klient) -> None:
    """Nicht nur die Wahrscheinlichkeit: Entscheidung, Schwelle, Modell, Kennung."""
    antwort = klient.post("/predict", json=GUELTIG)
    assert antwort.status_code == 200
    daten = antwort.json()

    assert 0.0 <= daten["failure_probability"] <= 1.0
    assert isinstance(daten["alarm"], bool)
    assert daten["alarm"] == (daten["failure_probability"] >= daten["threshold"])
    assert re.fullmatch(r"[0-9a-f-]{36}", daten["request_id"])
    assert daten["model"]["sha256"]
    assert daten["latency_ms"] >= 0


def test_riskanter_zustand_loest_alarm_aus(klient) -> None:
    """Verhaltenstest: hoher Verschleiß und hohe Last müssen Alarm ergeben."""
    daten = klient.post("/predict", json=RISKANT).json()
    assert daten["alarm"] is True
    assert daten["failure_probability"] > 0.5


def test_harmloser_zustand_loest_keinen_alarm_aus(klient) -> None:
    daten = klient.post("/predict", json=GUELTIG).json()
    assert daten["alarm"] is False


def test_gleiche_anfrage_gleiche_antwort(klient) -> None:
    """Determinismus: zwei gleiche Anfragen müssen dieselbe Zahl liefern."""
    erste = klient.post("/predict", json=RISKANT).json()
    zweite = klient.post("/predict", json=RISKANT).json()
    assert erste["failure_probability"] == zweite["failure_probability"]
    assert erste["request_id"] != zweite["request_id"]  # Kennung ist je Anfrage neu


# ── Ungültige Eingaben ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("feld", "wert", "was"),
    [
        ("type", "X", "unbekannte Qualitätsvariante"),
        ("torque_nm", -5, "negatives Drehmoment"),
        ("air_temperature_k", 100, "physikalisch unmögliche Temperatur"),
        ("rotational_speed_rpm", 99999, "Drehzahl über der harten Obergrenze"),
        ("tool_wear_min", "viel", "Text statt Zahl"),
    ],
)
def test_ungueltige_eingabe_ergibt_422(klient, feld, wert, was) -> None:
    """Harte Grenzen des Datenvertrags -> 422 mit Begründung, keine Vorhersage."""
    anfrage = {**GUELTIG, feld: wert}
    antwort = klient.post("/predict", json=anfrage)
    assert antwort.status_code == 422, was
    fehler = antwort.json()["detail"]
    assert any(feld in str(e["loc"]) for e in fehler)
    assert fehler[0]["msg"]  # eine lesbare Begründung ist vorhanden


def test_fehlendes_feld_ergibt_422(klient) -> None:
    ohne_drehmoment = {k: v for k, v in GUELTIG.items() if k != "torque_nm"}
    assert klient.post("/predict", json=ohne_drehmoment).status_code == 422


def test_tippfehler_im_feldnamen_ergibt_422(klient) -> None:
    """``extra="forbid"``: ein zusätzliches Feld rutscht nicht stillschweigend durch."""
    mit_tippfehler = {**GUELTIG, "torque_Nm": 42.8}
    assert klient.post("/predict", json=mit_tippfehler).status_code == 422


def test_weiche_grenze_wird_beantwortet_aber_gemeldet(klient) -> None:
    """Ungewöhnlich, aber möglich: Antwort mit Hinweis statt Ablehnung."""
    params = load_params()
    warn_max = params["data_contract"]["columns"]["torque_nm"]["warn_max"]
    hart_max = params["data_contract"]["columns"]["torque_nm"]["max"]
    dazwischen = (warn_max + hart_max) / 2

    antwort = klient.post("/predict", json={**GUELTIG, "torque_nm": dazwischen})
    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["warnings"], "Hinweis auf die weiche Grenze fehlt"
    assert any("torque_nm" in w for w in daten["warnings"])


# ── Stapelverarbeitung ──────────────────────────────────────────────────


def test_stapel_antwortet_je_zeile(klient) -> None:
    antwort = klient.post("/predict/batch", json={"items": [GUELTIG, RISKANT, GUELTIG]})
    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["count"] == 3
    assert len(daten["predictions"]) == 3
    assert len(daten["alarm_flags"]) == 3
    assert daten["alarms"] == sum(daten["alarm_flags"]) == 1


def test_stapel_und_einzelaufruf_stimmen_ueberein(klient) -> None:
    """Derselbe Zustand muss einzeln und im Stapel dieselbe Zahl ergeben."""
    einzeln = klient.post("/predict", json=RISKANT).json()["failure_probability"]
    im_stapel = klient.post("/predict/batch", json={"items": [GUELTIG, RISKANT]}).json()[
        "predictions"
    ][1]
    assert einzeln == im_stapel


def test_leerer_stapel_ergibt_422(klient) -> None:
    assert klient.post("/predict/batch", json={"items": []}).status_code == 422


# ── Kennzahlen und Dokumentation ────────────────────────────────────────


def test_metrics_im_prometheus_format(klient) -> None:
    antwort = klient.get("/metrics")
    assert antwort.status_code == 200
    assert antwort.headers["content-type"].startswith("text/plain")
    text = antwort.text
    assert "# TYPE pdm_predictions_total counter" in text
    assert "pdm_model_loaded 1" in text
    assert "pdm_decision_threshold" in text


def test_zaehler_wachsen_mit_den_vorhersagen(klient) -> None:
    def gelesen(name: str) -> float:
        for zeile in klient.get("/metrics").text.splitlines():
            if zeile.startswith(name + " "):
                return float(zeile.split()[1])
        raise AssertionError(f"{name} nicht in /metrics")

    vorher = gelesen("pdm_predictions_total")
    klient.post("/predict/batch", json={"items": [GUELTIG, RISKANT]})
    assert gelesen("pdm_predictions_total") == vorher + 2


def test_dokumentation_und_schema_sind_erreichbar(klient) -> None:
    assert klient.get("/docs").status_code == 200
    schema = klient.get("/openapi.json").json()
    for pfad in ("/health", "/model-info", "/predict", "/predict/batch", "/metrics"):
        assert pfad in schema["paths"], f"{pfad} fehlt im OpenAPI-Schema"


# ── Verhalten ohne Modell ───────────────────────────────────────────────


def test_ohne_modell_startet_der_dienst_und_meldet_es(monkeypatch) -> None:
    """Ein Dienst, der beim Start abstürzt, sagt niemandem, warum.

    Statt abzustürzen: 200 auf /health mit ``degraded``, 503 auf /predict.
    """
    import src.serving.api as api

    def kein_modell():
        raise FileNotFoundError("Keine Modelldatei (Test)")

    monkeypatch.setattr(api, "lade_modell", kein_modell)

    with TestClient(api.app) as c:
        gesundheit = c.get("/health")
        assert gesundheit.status_code == 200
        assert gesundheit.json()["status"] == "degraded"
        assert gesundheit.json()["model_loaded"] is False

        antwort = c.post("/predict", json=GUELTIG)
        assert antwort.status_code == 503
        assert "Test" in antwort.json()["detail"]

        assert "pdm_model_loaded 0" in c.get("/metrics").text
