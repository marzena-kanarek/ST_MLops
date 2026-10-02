"""HTTP-Schnittstelle des Modells (Etappe 16).

Starten::

    uvicorn src.serving.api:app --reload --port 8000

Danach ``http://localhost:8000/docs`` im Browser öffnen: FastAPI erzeugt aus den
Typangaben in ``schemas.py`` eine bedienbare Oberfläche, in der sich Anfragen
abschicken lassen, ohne curl zu schreiben.

**Die drei Dinge, die hier bewusst anders gemacht sind, als es naheliegt:**

1. **Das Modell wird einmal beim Start geladen**, in ``lifespan`` — nicht in der
   Endpunktfunktion. Sonst würde jede Anfrage die Modelldatei von der Platte
   lesen.
2. **Die Eingabe wird validiert**, und zwar gegen den Datenvertrag (siehe
   ``schemas.py``). Ein Tippfehler im Aufruf führt zu Statuscode 422 mit
   Begründung, nicht zu einer stillen Fehlvorhersage.
3. **Zurückgegeben wird mehr als die Wahrscheinlichkeit:** die Entscheidung, der
   verwendete Schwellenwert, das Modell samt Hashes, eine Anfrage-Kennung und
   etwaige Hinweise. Eine Antwort, die nur eine Zahl enthält, ist später nicht
   mehr nachvollziehbar.

Fehlt die Modelldatei, startet der Dienst trotzdem — ``/health`` meldet dann
``status: degraded`` und ``model_loaded: false``, die Vorhersage-Endpunkte
antworten mit 503. Ein Dienst, der beim Start abstürzt, sagt niemandem, warum.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response

from src.config import load_params
from src.serving.model_service import ModellDienst, lade_modell
from src.serving.prediction_log import Vorhersageprotokoll
from src.serving.prediction_log import aus_params as protokoll_aus_params
from src.serving.schemas import (
    Betriebszustand,
    Gesundheit,
    ModellInfo,
    Stapel,
    StapelAntwort,
    Vorhersage,
)
from src.utils.logging_setup import hole_logger

log = hole_logger(__name__)

PARAMS = load_params()
API = PARAMS["api"]

#: Zustand des laufenden Dienstes. Ein Wörterbuch statt globaler Variablen,
#: damit lifespan es beim Herunterfahren vollständig leeren kann.
STATE: dict[str, Any] = {}

#: Betriebskennzahlen im Arbeitsspeicher. Etappe 17 ergänzt die dauerhafte
#: Protokollierung; für /metrics im Prometheus-Format genügen Zähler.
ANFANGSWERTE: dict[str, float] = {
    "requests_total": 0,
    "requests_failed": 0,
    "predictions_total": 0,
    "alarms_total": 0,
    "warnings_total": 0,
    "logged_total": 0,
    "log_failures_total": 0,
    "latency_seconds_sum": 0.0,
}

KENNZAHLEN: dict[str, float] = dict(ANFANGSWERTE)


def setze_kennzahlen_zurueck() -> None:
    """Setzt alle Zähler auf null. Für Tests und für einen bewussten Neuanfang."""
    KENNZAHLEN.update(ANFANGSWERTE)


def protokolliere(protokoll: Vorhersageprotokoll | None, eintraege: list[dict[str, Any]]) -> None:
    """Schreibt Protokolleinträge und zählt mit.

    Läuft als Hintergrundaufgabe: Die Antwort ist beim Aufrufer, bevor
    geschrieben wird. Das Protokollieren verlängert damit nicht die Antwortzeit.
    """
    if protokoll is None:
        return
    try:
        geschrieben = protokoll.schreibe(eintraege)
    except Exception as fehler:  # noqa: BLE001 - zweite Absicherung
        log.error("Protokollieren fehlgeschlagen: %s", fehler)
        geschrieben = 0
    KENNZAHLEN["logged_total"] += geschrieben
    KENNZAHLEN["log_failures_total"] += len(eintraege) - geschrieben


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lädt das Modell beim Start und gibt es beim Herunterfahren frei."""
    STATE["started_at"] = time.time()
    STATE["protokoll"] = protokoll_aus_params(PARAMS)
    try:
        dienst = lade_modell()
        STATE["dienst"] = dienst
        log.info(
            "Modell geladen: %s (sha256 %s…), Schwellenwert %.2f, PR-AUC %.4f",
            dienst.pfad.name,
            dienst.sha256[:12],
            dienst.schwelle,
            float(dienst.kennzahlen.get("pr_auc", 0.0)),
        )
    except FileNotFoundError as fehler:
        STATE["dienst"] = None
        STATE["ladefehler"] = str(fehler)
        log.error("Kein Modell geladen: %s", fehler)
    log.info("Dienst bereit: %s %s", API["title"], API["version"])
    yield
    log.info("Dienst wird beendet.")
    STATE.clear()


app = FastAPI(
    title=API["title"],
    version=API["version"],
    summary="Vorausschauende Wartung: Ausfallwahrscheinlichkeit eines Betriebspunkts",
    lifespan=lifespan,
)


@app.middleware("http")
async def zaehle_anfragen(request: Request, call_next):
    """Zählt Anfragen und Antwortzeiten für ``/metrics``."""
    beginn = time.perf_counter()
    antwort = await call_next(request)
    if request.url.path != "/metrics":
        KENNZAHLEN["requests_total"] += 1
        KENNZAHLEN["latency_seconds_sum"] += time.perf_counter() - beginn
        if antwort.status_code >= 400:
            KENNZAHLEN["requests_failed"] += 1
    return antwort


def dienst() -> ModellDienst:
    """Gibt den geladenen Dienst zurück oder antwortet mit 503."""
    vorhanden = STATE.get("dienst")
    if vorhanden is None:
        raise HTTPException(
            status_code=503,
            detail=STATE.get("ladefehler", "Kein Modell geladen."),
        )
    return vorhanden


# ── Endpunkte ───────────────────────────────────────────────────────────


@app.get("/health", response_model=Gesundheit, tags=["Betrieb"])
def health() -> Gesundheit:
    """Läuft der Dienst, und ist ein Modell geladen?

    Für Startprüfungen (Container, CI) und für die Überwachung. Antwortet
    bewusst auch dann mit 200, wenn kein Modell geladen ist — die Aussage steckt
    in ``status`` und ``model_loaded``, nicht im Statuscode.
    """
    geladen = STATE.get("dienst")
    return Gesundheit(
        status="ok" if geladen else "degraded",
        model_loaded=geladen is not None,
        model_file=None if geladen is None else str(geladen.pfad.name),
        uptime_s=round(time.time() - STATE.get("started_at", time.time()), 1),
        api_version=API["version"],
    )


@app.get("/model-info", response_model=ModellInfo, tags=["Betrieb"])
def model_info() -> ModellInfo:
    """Welches Modell ist geladen, mit welchem Schwellenwert, aus welchen Daten?"""
    return ModellInfo(**dienst().auskunft())


@app.post("/predict", response_model=Vorhersage, tags=["Vorhersage"])
def predict(anfrage: Betriebszustand, hintergrund: BackgroundTasks) -> Vorhersage:
    """Einzelvorhersage für einen Betriebspunkt."""
    d = dienst()
    beginn = time.perf_counter()

    eingabe = anfrage.model_dump()
    frame = d.rahmen([eingabe])
    wahrscheinlichkeit = d.wahrscheinlichkeiten(frame)[0]
    alarm = d.alarm(wahrscheinlichkeit)
    hinweise = d.hinweise(frame)
    dauer_ms = round((time.perf_counter() - beginn) * 1000, 2)
    kennung = str(uuid.uuid4())

    KENNZAHLEN["predictions_total"] += 1
    KENNZAHLEN["alarms_total"] += int(alarm)
    KENNZAHLEN["warnings_total"] += len(hinweise)

    # Protokollstufen mit Absicht gewählt: Ein Alarm ist eine Nachricht, eine
    # ruhige Vorhersage nicht — sonst ertrinkt das Protokoll in Normalbetrieb.
    if alarm:
        log.info("Alarm: p=%.3f >= %.2f  request_id=%s", wahrscheinlichkeit, d.schwelle, kennung)
    else:
        log.debug("ruhig: p=%.3f  request_id=%s", wahrscheinlichkeit, kennung)
    for hinweis in hinweise:
        log.warning("Datenvertrag (weiche Grenze): %s  request_id=%s", hinweis, kennung)

    protokoll: Vorhersageprotokoll | None = STATE.get("protokoll")
    if protokoll is not None:
        hintergrund.add_task(
            protokolliere,
            protokoll,
            [
                protokoll.eintrag(
                    request_id=kennung,
                    zeile=0,
                    eingabe=eingabe,
                    wahrscheinlichkeit=wahrscheinlichkeit,
                    alarm=alarm,
                    schwelle=d.schwelle,
                    modell=d.modellangabe(),
                    hinweise=hinweise,
                    dauer_ms=dauer_ms,
                )
            ],
        )

    return Vorhersage(
        request_id=kennung,
        failure_probability=round(wahrscheinlichkeit, 6),
        alarm=alarm,
        threshold=d.schwelle,
        warnings=hinweise,
        model=d.modellangabe(),  # type: ignore[arg-type]
        latency_ms=dauer_ms,
    )


@app.post("/predict/batch", response_model=StapelAntwort, tags=["Vorhersage"])
def predict_batch(anfrage: Stapel, hintergrund: BackgroundTasks) -> StapelAntwort:
    """Mehrere Betriebspunkte in einer Anfrage.

    Ein Aufruf mit 1.000 Zeilen ist deutlich schneller als 1.000 Einzelaufrufe:
    Die Vorverarbeitung und der Wald laufen einmal über die ganze Matrix statt
    tausendmal über eine Zeile.
    """
    d = dienst()
    beginn = time.perf_counter()

    eingaben = [z.model_dump() for z in anfrage.items]
    frame = d.rahmen(eingaben)
    wahrscheinlichkeiten = d.wahrscheinlichkeiten(frame)
    flaggen = [d.alarm(p) for p in wahrscheinlichkeiten]
    hinweise = d.hinweise(frame)
    dauer_ms = round((time.perf_counter() - beginn) * 1000, 2)
    kennung = str(uuid.uuid4())

    KENNZAHLEN["predictions_total"] += len(wahrscheinlichkeiten)
    KENNZAHLEN["alarms_total"] += sum(flaggen)
    KENNZAHLEN["warnings_total"] += len(hinweise)

    log.info(
        "Stapel: %d Zeilen, %d Alarme, %.1f ms  request_id=%s",
        len(wahrscheinlichkeiten),
        sum(flaggen),
        dauer_ms,
        kennung,
    )
    for hinweis in hinweise:
        log.warning("Datenvertrag (weiche Grenze): %s  request_id=%s", hinweis, kennung)

    # Ein Eintrag je Zeile, alle mit derselben request_id: Die Driftprüfung in
    # Etappe 18 braucht einzelne Messwerte, nicht einen Mittelwert je Anfrage.
    protokoll: Vorhersageprotokoll | None = STATE.get("protokoll")
    if protokoll is not None:
        je_zeile = dauer_ms / max(len(wahrscheinlichkeiten), 1)
        hintergrund.add_task(
            protokolliere,
            protokoll,
            [
                protokoll.eintrag(
                    request_id=kennung,
                    zeile=nummer,
                    eingabe=eingabe,
                    wahrscheinlichkeit=p,
                    alarm=flagge,
                    schwelle=d.schwelle,
                    modell=d.modellangabe(),
                    hinweise=hinweise,
                    dauer_ms=je_zeile,
                )
                for nummer, (eingabe, p, flagge) in enumerate(
                    zip(eingaben, wahrscheinlichkeiten, flaggen, strict=True)
                )
            ],
        )

    return StapelAntwort(
        request_id=kennung,
        count=len(wahrscheinlichkeiten),
        alarms=sum(flaggen),
        threshold=d.schwelle,
        predictions=[round(p, 6) for p in wahrscheinlichkeiten],
        alarm_flags=flaggen,
        warnings=hinweise,
        model=d.modellangabe(),  # type: ignore[arg-type]
        latency_ms=dauer_ms,
    )


@app.get("/metrics", tags=["Betrieb"])
def metrics() -> Response:
    """Betriebskennzahlen im Prometheus-Textformat.

    Absichtlich dasselbe Format, das Prometheus abholt: ``# HELP``, ``# TYPE``,
    dann Name und Wert. Damit lässt sich der Dienst später ohne Änderung an eine
    Überwachung hängen. Die Zähler leben im Arbeitsspeicher und beginnen bei
    jedem Neustart bei null — das ist bei Prometheus-Zählern üblich und wird
    dort über die Zeitreihe aufgelöst.
    """
    geladen = 1 if STATE.get("dienst") else 0
    laufzeit = time.time() - STATE.get("started_at", time.time())

    zeilen = [
        "# HELP pdm_model_loaded Ist ein Modell geladen (1) oder nicht (0).",
        "# TYPE pdm_model_loaded gauge",
        f"pdm_model_loaded {geladen}",
        "# HELP pdm_uptime_seconds Laufzeit des Dienstes in Sekunden.",
        "# TYPE pdm_uptime_seconds gauge",
        f"pdm_uptime_seconds {laufzeit:.1f}",
        "# HELP pdm_requests_total Beantwortete Anfragen insgesamt.",
        "# TYPE pdm_requests_total counter",
        f"pdm_requests_total {int(KENNZAHLEN['requests_total'])}",
        "# HELP pdm_requests_failed_total Anfragen mit Statuscode >= 400.",
        "# TYPE pdm_requests_failed_total counter",
        f"pdm_requests_failed_total {int(KENNZAHLEN['requests_failed'])}",
        "# HELP pdm_predictions_total Berechnete Vorhersagen (Zeilen, nicht Anfragen).",
        "# TYPE pdm_predictions_total counter",
        f"pdm_predictions_total {int(KENNZAHLEN['predictions_total'])}",
        "# HELP pdm_alarms_total Vorhersagen oberhalb des Schwellenwerts.",
        "# TYPE pdm_alarms_total counter",
        f"pdm_alarms_total {int(KENNZAHLEN['alarms_total'])}",
        "# HELP pdm_contract_warnings_total Hinweise aus den weichen Grenzen des Datenvertrags.",
        "# TYPE pdm_contract_warnings_total counter",
        f"pdm_contract_warnings_total {int(KENNZAHLEN['warnings_total'])}",
        "# HELP pdm_logged_total Protokollierte Vorhersagen (Zeilen).",
        "# TYPE pdm_logged_total counter",
        f"pdm_logged_total {int(KENNZAHLEN['logged_total'])}",
        "# HELP pdm_log_failures_total Vorhersagen, die nicht protokolliert werden konnten.",
        "# TYPE pdm_log_failures_total counter",
        f"pdm_log_failures_total {int(KENNZAHLEN['log_failures_total'])}",
        "# HELP pdm_latency_seconds_sum Aufsummierte Antwortzeit aller Anfragen.",
        "# TYPE pdm_latency_seconds_sum counter",
        f"pdm_latency_seconds_sum {KENNZAHLEN['latency_seconds_sum']:.6f}",
    ]
    if geladen:
        d = STATE["dienst"]
        zeilen += [
            "# HELP pdm_decision_threshold Verwendeter Entscheidungsschwellenwert.",
            "# TYPE pdm_decision_threshold gauge",
            f"pdm_decision_threshold {d.schwelle}",
            "# HELP pdm_model_pr_auc PR-AUC des geladenen Modells auf der Validierungsmenge.",
            "# TYPE pdm_model_pr_auc gauge",
            f"pdm_model_pr_auc {float(d.kennzahlen.get('pr_auc', 0.0)):.6f}",
        ]

    return Response(
        content="\n".join(zeilen) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
