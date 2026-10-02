"""Protokoll der Vorhersagen (Etappe 17).

**Warum das nicht optional ist:** Ohne Protokoll ist Überwachung unmöglich — es
gäbe nichts, worauf man sie anwenden könnte. Und wenn sich jemand über eine
seltsame Warnung beschwert, will man nachsehen können, welche Werte damals
hereinkamen. Etappe 18 misst Drift genau auf diesen Zeilen.

Geschrieben wird in **zwei** Formate, weil sie verschiedene Zwecke haben:

* **JSONL** (``reports/predictions.jsonl``) — eine JSON-Zeile je Vorhersage.
  Anhängen ist ein Schreibvorgang, die Datei bleibt mit ``tail`` lesbar, und sie
  lässt sich ohne Werkzeuge woanders einspielen.
* **SQLite** (``reports/predictions.db``) — dieselben Angaben in einer Tabelle,
  damit man sie per SQL auswerten kann::

      SELECT date(timestamp) AS tag, COUNT(*) AS anfragen,
             AVG(alarm) AS alarmquote, AVG(latency_ms) AS antwortzeit_ms
      FROM predictions GROUP BY tag ORDER BY tag DESC;

Ein Eintrag je **Zeile**, nicht je Anfrage: Ein Stapel mit 100 Betriebspunkten
erzeugt 100 Einträge, die dieselbe ``request_id`` tragen. Nur so hat die
Driftprüfung einzelne Messwerte, und nur so lässt sich eine einzelne Vorhersage
wiederfinden.

Lässt sich die Datenbank nicht anlegen — auf Netzlaufwerken kann SQLite nicht
sperren —, läuft der Dienst mit der JSONL-Datei allein weiter und schreibt eine
deutliche Meldung ins Anwendungsprotokoll. Ein Protokollziel, das nicht
erreichbar ist, soll den Dienst nicht am Start hindern.

**Fehler beim Protokollieren dürfen die Antwort nicht verhindern.** Eine volle
Platte ist ein Betriebsproblem, kein Grund, dem Aufrufer eine Vorhersage zu
verweigern. Deshalb fängt ``schreibe`` Ausnahmen ab und meldet sie ins
Anwendungsprotokoll.

Auswertung von der Kommandozeile::

    python -m src.serving.prediction_log
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import absolut, load_params
from src.utils.logging_setup import hole_logger

log = hole_logger(__name__)

#: Die Spalten der SQLite-Tabelle. ``inputs`` steht als JSON-Text darin — eine
#: Spalte je Merkmal wäre schneller abzufragen, müsste aber bei jeder Änderung
#: des Datenvertrags mitgepflegt werden.
TABELLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    request_id    TEXT    NOT NULL,
    row           INTEGER NOT NULL,
    probability   REAL    NOT NULL,
    alarm         INTEGER NOT NULL,
    threshold     REAL    NOT NULL,
    model_name    TEXT    NOT NULL,
    model_sha256  TEXT    NOT NULL,
    latency_ms    REAL,
    warnings      TEXT,
    inputs        TEXT
)
"""

INDEX = "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)"

TAGESSTATISTIK = """
SELECT date(timestamp) AS tag,
       COUNT(*)        AS anfragen,
       AVG(alarm)      AS alarmquote,
       AVG(probability) AS mittlere_wahrscheinlichkeit,
       AVG(latency_ms) AS antwortzeit_ms
FROM predictions
GROUP BY tag
ORDER BY tag DESC
"""


def jetzt() -> str:
    """Zeitstempel in UTC, nach ISO 8601 — mit Zeitzone, sonst unbrauchbar."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Vorhersageprotokoll:
    """Schreibt Vorhersagen nach JSONL und SQLite.

    Attribute:
        jsonl_pfad: Ziel der JSON-Zeilen, ``None`` schaltet sie ab.
        sqlite_pfad: Ziel der Datenbank, ``None`` schaltet sie ab.
        mit_eingaben: Eingangswerte mitschreiben. Bei personenbezogenen Daten
            wäre hier ``False`` richtig — die Driftprüfung müsste dann mit
            Kennzahlen statt mit Rohwerten arbeiten.
    """

    jsonl_pfad: Path | None
    sqlite_pfad: Path | None
    mit_eingaben: bool = True

    def __post_init__(self) -> None:
        for pfad in (self.jsonl_pfad, self.sqlite_pfad):
            if pfad is not None:
                pfad.parent.mkdir(parents=True, exist_ok=True)
        if self.sqlite_pfad is not None:
            try:
                with self._verbindung() as verbindung:
                    verbindung.execute(TABELLE)
                    verbindung.execute(INDEX)
            except sqlite3.Error as fehler:
                # Kommt auf Netzlaufwerken vor, auf denen SQLite nicht sperren
                # kann (derselbe Grund, aus dem MLFLOW_TRACKING_URI umstellbar
                # ist). Der Dienst läuft dann mit JSONL weiter, statt beim Start
                # abzubrechen - aber die Meldung ist deutlich.
                log.error(
                    "SQLite-Protokoll unter %s nicht nutzbar (%s). "
                    "Es wird nur die JSONL-Datei geschrieben.",
                    self.sqlite_pfad,
                    fehler,
                )
                self.sqlite_pfad = None

    # ── Verbindung ──────────────────────────────────────────────────────

    def _verbindung(self) -> sqlite3.Connection:
        """Eine Verbindung je Schreibvorgang.

        Absichtlich nicht eine dauerhafte: SQLite-Verbindungen gehören dem
        Thread, der sie geöffnet hat, und ein Webdienst beantwortet Anfragen in
        wechselnden Threads.
        """
        verbindung = sqlite3.connect(self.sqlite_pfad, timeout=5.0)
        verbindung.row_factory = sqlite3.Row
        return verbindung

    # ── Schreiben ───────────────────────────────────────────────────────

    def eintrag(
        self,
        *,
        request_id: str,
        zeile: int,
        eingabe: dict[str, Any],
        wahrscheinlichkeit: float,
        alarm: bool,
        schwelle: float,
        modell: dict[str, Any],
        hinweise: list[str],
        dauer_ms: float,
    ) -> dict[str, Any]:
        """Baut einen vollständigen Protokolleintrag.

        Enthält alles, was der Leitfaden verlangt: Zeitstempel, Anfrage-ID,
        Eingangswerte, Wahrscheinlichkeit, Entscheidung, Schwellenwert, Modell
        mit Version, Antwortzeit und die Hinweise aus der Validierung.
        """
        return {
            "timestamp": jetzt(),
            "request_id": request_id,
            "row": zeile,
            "probability": round(float(wahrscheinlichkeit), 6),
            "alarm": bool(alarm),
            "threshold": float(schwelle),
            "model_name": str(modell.get("name", "unbekannt")),
            "model_sha256": str(modell.get("sha256", "unbekannt")),
            "latency_ms": round(float(dauer_ms), 3),
            "warnings": list(hinweise),
            "inputs": dict(eingabe) if self.mit_eingaben else None,
        }

    def schreibe(self, eintraege: list[dict[str, Any]]) -> int:
        """Schreibt Einträge in beide Ziele. Fehler werden gemeldet, nicht geworfen.

        Returns:
            Anzahl der geschriebenen Einträge, 0 bei einem Fehler.
        """
        if not eintraege:
            return 0
        try:
            if self.jsonl_pfad is not None:
                with open(self.jsonl_pfad, "a", encoding="utf-8") as datei:
                    for eintrag in eintraege:
                        datei.write(json.dumps(eintrag, ensure_ascii=False) + "\n")

            if self.sqlite_pfad is not None:
                with self._verbindung() as verbindung:
                    verbindung.executemany(
                        "INSERT INTO predictions (timestamp, request_id, row, "
                        "probability, alarm, threshold, model_name, model_sha256, "
                        "latency_ms, warnings, inputs) "
                        "VALUES (:timestamp, :request_id, :row, :probability, :alarm, "
                        ":threshold, :model_name, :model_sha256, :latency_ms, "
                        ":warnings, :inputs)",
                        [
                            {
                                **eintrag,
                                "alarm": int(eintrag["alarm"]),
                                "warnings": json.dumps(eintrag["warnings"], ensure_ascii=False),
                                "inputs": (
                                    json.dumps(eintrag["inputs"], ensure_ascii=False)
                                    if eintrag["inputs"] is not None
                                    else None
                                ),
                            }
                            for eintrag in eintraege
                        ],
                    )
        except Exception as fehler:  # noqa: BLE001 - Protokollfehler darf nicht durchschlagen
            log.error("Vorhersage konnte nicht protokolliert werden: %s", fehler)
            return 0
        return len(eintraege)

    # ── Lesen ───────────────────────────────────────────────────────────

    def anzahl(self) -> int:
        """Zahl der Einträge in der Datenbank."""
        if self.sqlite_pfad is None or not self.sqlite_pfad.exists():
            return 0
        with self._verbindung() as verbindung:
            return int(verbindung.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def zeilen(self, grenze: int = 10) -> list[dict[str, Any]]:
        """Die jüngsten Einträge, für die Fehlersuche."""
        if self.sqlite_pfad is None or not self.sqlite_pfad.exists():
            return []
        with self._verbindung() as verbindung:
            treffer = verbindung.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (grenze,)
            ).fetchall()
        return [dict(zeile) for zeile in treffer]

    def tagesstatistik(self) -> list[dict[str, Any]]:
        """Anfragen, Alarmquote und Antwortzeit je Tag."""
        if self.sqlite_pfad is None or not self.sqlite_pfad.exists():
            return []
        with self._verbindung() as verbindung:
            return [dict(z) for z in verbindung.execute(TAGESSTATISTIK).fetchall()]


def aus_params(params: dict[str, Any] | None = None) -> Vorhersageprotokoll:
    """Baut das Protokoll aus dem Abschnitt ``logging`` von params.yaml."""
    einstellungen = (params or load_params())["logging"]
    return Vorhersageprotokoll(
        jsonl_pfad=absolut(einstellungen["predictions_jsonl"])
        if einstellungen.get("predictions_jsonl")
        else None,
        sqlite_pfad=absolut(einstellungen["predictions_sqlite"])
        if einstellungen.get("predictions_sqlite")
        else None,
        mit_eingaben=bool(einstellungen.get("log_inputs", True)),
    )


def _main() -> None:
    """Zeigt die Auswertung des Protokolls."""
    protokoll = aus_params()
    print(f"Einträge insgesamt: {protokoll.anzahl()}\n")

    statistik = protokoll.tagesstatistik()
    if not statistik:
        print("Noch keine Vorhersagen protokolliert.")
        print("Dienst starten (uvicorn src.serving.api:app) und /predict aufrufen.")
        return

    print(f"{'Tag':12s} {'Anfragen':>9s} {'Alarmquote':>11s} {'mittl. p':>9s} {'Antwortzeit':>12s}")
    for zeile in statistik:
        print(
            f"{zeile['tag']:12s} {zeile['anfragen']:>9d} "
            f"{zeile['alarmquote']:>10.1%} "
            f"{zeile['mittlere_wahrscheinlichkeit']:>9.3f} "
            f"{zeile['antwortzeit_ms']:>9.2f} ms"
        )

    print("\nJüngste Einträge:")
    for zeile in protokoll.zeilen(3):
        print(
            f"  {zeile['timestamp']}  p={zeile['probability']:.3f}  "
            f"alarm={bool(zeile['alarm'])}  {zeile['latency_ms']:.1f} ms"
        )


if __name__ == "__main__":
    _main()
