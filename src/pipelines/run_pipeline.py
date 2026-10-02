"""Die ganze Kette als ein Befehl (Etappe 15).

Ein Aufruf führt vom Rohdatensatz zum bewerteten, gespeicherten Modell::

    python -m src.pipelines.run_pipeline

**Warum das die eigentliche Probe auf Reproduzierbarkeit ist:** Eine Kette, die
nur als Abfolge von Notebooks existiert, die jemand in der richtigen Reihenfolge
anklicken muss, ist nicht reproduzierbar — sie ist mündliche Überlieferung. Erst
wenn ein Befehl aus leeren Verzeichnissen dasselbe Ergebnis herstellt, ist der
Nachweis geführt.

Die fünf Schritte stehen in ``SCHRITTE`` und laufen in fester Reihenfolge. Jeder
ist eine Funktion mit klaren Ein- und Ausgaben; keiner greift auf
Zwischenzustände eines anderen zu, außer über das gemeinsame ``zustand``-Wörter-
buch. Deshalb ließe sich jeder Schritt später ohne Umbau als Aufgabe in einem
Orchestrator (Airflow, Prefect, Kubeflow) führen. Für einen Prototyp ist ein
solcher Orchestrator nicht nötig — die Struktur darauf auszulegen ist eine
Architekturentscheidung, keine Auslassung.

Am Ende entsteht ``reports/pipeline_run.json``, das **Manifest**. Es beantwortet
in einer Datei die Frage, die jede Prüfung stellt: *welche Daten, welcher Code,
welche Parameter?* Die Schlüssel dieser Datei sind englisch geschrieben — so
heißt ``provenance.raw_data_sha256`` genau so, wie es in
``references/datenbeschreibung.md`` zugesagt ist, und ein Werkzeug, das Manifeste
einliest, findet die erwarteten Namen.

Aufrufe::

    python -m src.pipelines.run_pipeline
    python -m src.pipelines.run_pipeline --no-mlflow
    python -m src.pipelines.run_pipeline --name "nachtlauf"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PARAMS_PATH, PATHS, load_params
from src.data.load import datei_hash, load_raw_data, pruefe_integritaet
from src.data.split import (
    entferne_leakage_und_kennungen,
    schreibe_teilmengen,
    split_data,
    uebersicht,
)
from src.data.validate import load_contract, validate_dataframe
from src.modeling.train import trainiere
from src.utils.tracking import bibliotheksversionen, git_commit

#: Die Schritte der Kette in fester Reihenfolge.
SCHRITTE: tuple[str, ...] = (
    "daten",
    "pruefen",
    "aufteilen",
    "trainieren",
    "bewerten",
    "referenz",
)

#: Dateiname des Manifests.
MANIFEST_NAME = "pipeline_run.json"


def set_seeds(seed: int) -> None:
    """Legt alle Zufallsquellen fest, die das Ergebnis beeinflussen.

    ``PYTHONHASHSEED`` wirkt nur auf Prozesse, die **nach** dieser Zeile
    gestartet werden — der eigene Interpreter hat seinen Hash-Startwert schon
    beim Start festgelegt. Gesetzt wird die Variable trotzdem: Unterprozesse
    (etwa die Arbeitsprozesse von ``n_jobs=-1``) erben sie.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def jetzt() -> str:
    """Zeitstempel in UTC, nach ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def text_hash(pfad: Path) -> str:
    """SHA-256 einer Datei — hier für params.yaml."""
    return hashlib.sha256(pfad.read_bytes()).hexdigest()


# ── Die einzelnen Schritte ──────────────────────────────────────────────
# Jeder Schritt bekommt denselben Satz Argumente und gibt zurück, was im
# Manifest über ihn stehen soll. Was ein späterer Schritt braucht, legt er in
# "zustand" ab — das ist die einzige Verbindung zwischen ihnen.


def schritt_daten(
    zustand: dict[str, Any], params: dict[str, Any], optionen: dict[str, Any]
) -> dict[str, Any]:
    """Rohdaten laden und den eingefrorenen Zustand prüfen."""
    hash_wert = pruefe_integritaet()
    frame = load_raw_data(integritaet_pruefen=False)
    zustand["roh"] = frame
    zustand["raw_data_sha256"] = hash_wert
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "positive_rate": round(float(frame[params["data"]["target"]].mean()), 5),
    }


def schritt_pruefen(
    zustand: dict[str, Any], params: dict[str, Any], optionen: dict[str, Any]
) -> dict[str, Any]:
    """Rohdaten gegen den Datenvertrag prüfen. Fehler brechen die Kette ab."""
    vertrag = load_contract()
    vertrag["check_row_count"] = True  # bei der vollständigen Rohdatei sinnvoll
    ergebnis = validate_dataframe(zustand["roh"], vertrag)
    ergebnis.raise_if_invalid()
    return {
        "checks_run": ergebnis.checks_run,
        "errors": len(ergebnis.errors),
        "warnings": ergebnis.warnings,
    }


def schritt_aufteilen(
    zustand: dict[str, Any], params: dict[str, Any], optionen: dict[str, Any]
) -> dict[str, Any]:
    """Ursachenspalten und Kennungen entfernen, geschichtet dreiteilig aufteilen."""
    merkmale = entferne_leakage_und_kennungen(zustand["roh"], params)
    entfernt = sorted(set(zustand["roh"].columns) - set(merkmale.columns))

    train, val, test = split_data(merkmale, params)
    teilmengen = {"train": train, "val": val, "test": test}
    pfade = schreibe_teilmengen(teilmengen)

    zustand["uebersicht"] = uebersicht(teilmengen, params["data"]["target"])
    return {
        "removed_columns": entfernt,
        "rows": {name: int(len(teil)) for name, teil in teilmengen.items()},
        "positives": {
            name: int(teil[params["data"]["target"]].sum()) for name, teil in teilmengen.items()
        },
        "files": [PATHS.relativ(p) for p in pfade.values()],
    }


def schritt_trainieren(
    zustand: dict[str, Any], params: dict[str, Any], optionen: dict[str, Any]
) -> dict[str, Any]:
    """Modell trainieren, bewerten, gegen die Qualitätsschranke prüfen.

    Der Schritt ruft genau dieselbe Funktion auf, die auch ``python -m
    src.modeling.train`` benutzt. Es gibt keinen zweiten Trainingspfad, der
    auseinanderlaufen könnte.
    """
    ergebnis = trainiere(
        lauf_name=optionen.get("lauf_name"),
        mit_mlflow=optionen.get("mit_mlflow", True),
    )
    zustand["ergebnis"] = ergebnis
    return {
        "model": params["model"]["name"],
        "hyperparameters": params["model"]["random_forest"],
        "mlflow": bool(optionen.get("mit_mlflow", True)),
        "pr_auc": round(ergebnis["pr_auc"], 4),
        "recall": round(ergebnis["recall"], 4),
    }


def schritt_bewerten(
    zustand: dict[str, Any], params: dict[str, Any], optionen: dict[str, Any]
) -> dict[str, Any]:
    """Prüfen, was der Lauf hinterlassen hat, und das Modell abhaken.

    Hier wird nicht neu gerechnet, sondern nachgesehen: Hat die Schranke
    gehalten, und liegt das Modell wirklich auf der Platte? Ein nicht
    bestandener Lauf bricht hier ab — mit dem ausdrücklichen Hinweis, dass das
    bisherige Modell unverändert geblieben ist.
    """
    ergebnis = zustand["ergebnis"]
    if not ergebnis["schranke_bestanden"]:
        raise RuntimeError(
            "Qualitätsschranke nicht bestanden: "
            + "; ".join(ergebnis["verstoesse"])
            + ". Das Modell wurde nicht gespeichert, das bisherige "
            "models/model.joblib bleibt unverändert."
        )

    modell_pfad = PATHS.models / "model.joblib"
    if not modell_pfad.exists():
        raise FileNotFoundError(f"Erwartete Modelldatei fehlt: {modell_pfad}")

    zustand["model_sha256"] = datei_hash(modell_pfad)
    zustand["model_file"] = PATHS.relativ(modell_pfad)
    return {
        "model_file": zustand["model_file"],
        "model_bytes": modell_pfad.stat().st_size,
        "quality_gate": "bestanden",
        "costs_eur": ergebnis["kosten_eur"],
    }


def schritt_referenz(
    zustand: dict[str, Any], params: dict[str, Any], optionen: dict[str, Any]
) -> dict[str, Any]:
    """Referenzstichprobe für die Überwachung ablegen (Etappe 18).

    Ohne Referenz kann man keine Drift messen. Abgelegt wird eine Stichprobe der
    **Validierungsmenge** samt den Wahrscheinlichkeiten, die das gerade
    trainierte Modell ihr gibt — Eingangsverteilungen *und* Vorhersageverteilung
    in einer Datei.

    Die Validierungsmenge und nicht die Trainingsmenge: Auf gelernten Zeilen
    sagt ein Random Forest beinahe 0 oder 1, diese Verteilung wäre als Maßstab
    für den Betrieb unbrauchbar.
    """
    from src.monitoring.drift import schreibe_referenz

    pfad, zeilen = schreibe_referenz(params=params)
    zustand["referenz_pfad"] = pfad
    return {
        "file": PATHS.relativ(pfad),
        "rows": zeilen,
        "source": "validierungsmenge",
    }


FUNKTIONEN: dict[str, Callable[..., dict[str, Any]]] = {
    "daten": schritt_daten,
    "pruefen": schritt_pruefen,
    "aufteilen": schritt_aufteilen,
    "trainieren": schritt_trainieren,
    "bewerten": schritt_bewerten,
    "referenz": schritt_referenz,
}


# ── Der Durchlauf ───────────────────────────────────────────────────────


def schreibe_manifest(manifest: dict[str, Any]) -> Path:
    """Legt das Manifest unter reports/ ab und gibt den Pfad zurück."""
    pfad = PATHS.reports / MANIFEST_NAME
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return pfad


def fuehre_aus(
    mit_mlflow: bool = True,
    lauf_name: str | None = None,
) -> dict[str, Any]:
    """Führt alle Schritte aus und schreibt das Manifest.

    Das Manifest wird **auch bei einem Abbruch** geschrieben: Ein fehlgeschlagener
    Lauf, über den nichts festgehalten ist, lässt sich nicht untersuchen.

    Returns:
        Das Manifest als Wörterbuch.
    """
    params = load_params()
    set_seeds(params["seed"])
    PATHS.ensure()

    optionen = {"mit_mlflow": mit_mlflow, "lauf_name": lauf_name}
    zustand: dict[str, Any] = {}
    beginn = time.perf_counter()

    manifest: dict[str, Any] = {
        "started_at": jetzt(),
        "finished_at": None,
        "duration_s": None,
        "status": "laufend",
        "command": "python -m src.pipelines.run_pipeline",
        "provenance": {
            "raw_data_sha256": params["data"]["raw_sha256"],
            "params_sha256": text_hash(PARAMS_PATH),
            "params_file": PATHS.relativ(PARAMS_PATH),
            "git_commit": git_commit(),
            "seed": params["seed"],
        },
        "environment": {
            **bibliotheksversionen(),
            "plattform": f"{platform.system()} {platform.machine()}",
            "python_executable": sys.executable,
        },
        "steps": {},
        "model_sha256": None,
        "metrics": None,
    }

    for name in SCHRITTE:
        schritt_beginn = time.perf_counter()
        print(f"[{name}] läuft …", flush=True)
        try:
            angaben = FUNKTIONEN[name](zustand, params, optionen)
        except Exception as fehler:
            manifest["steps"][name] = {
                "status": "fehlgeschlagen",
                "duration_s": round(time.perf_counter() - schritt_beginn, 2),
                "error": f"{type(fehler).__name__}: {fehler}",
            }
            manifest["status"] = "fehlgeschlagen"
            manifest["finished_at"] = jetzt()
            manifest["duration_s"] = round(time.perf_counter() - beginn, 2)
            pfad = schreibe_manifest(manifest)
            print(f"\n[{name}] ABGEBROCHEN: {fehler}")
            print(f"Manifest (mit Fehler): {PATHS.relativ(pfad)}")
            raise

        dauer = round(time.perf_counter() - schritt_beginn, 2)
        manifest["steps"][name] = {"status": "ok", "duration_s": dauer, **angaben}
        print(f"[{name}] fertig in {dauer:.2f} s")

    ergebnis = zustand["ergebnis"]
    manifest["model_sha256"] = zustand["model_sha256"]
    manifest["model_file"] = zustand["model_file"]
    manifest["metrics"] = {
        k: v for k, v in ergebnis.items() if k not in ("schranke_bestanden", "verstoesse")
    }
    manifest["quality_gate"] = {
        "passed": ergebnis["schranke_bestanden"],
        "violations": ergebnis["verstoesse"],
        "thresholds": params["quality_gate"],
    }
    manifest["status"] = "ok"
    manifest["finished_at"] = jetzt()
    manifest["duration_s"] = round(time.perf_counter() - beginn, 2)

    manifest["_manifest_path"] = PATHS.relativ(schreibe_manifest(manifest))
    return manifest


def _main() -> None:
    zerleger = argparse.ArgumentParser(
        description="Vollständiger Lauf: Rohdaten -> geprüftes, gespeichertes Modell"
    )
    zerleger.add_argument(
        "--no-mlflow", action="store_true", help="ohne Protokollierung in MLflow (Tests, CI)"
    )
    zerleger.add_argument("--name", default=None, help="Name des Laufs in MLflow")
    argumente = zerleger.parse_args()

    print(f"Konfiguration: {PATHS.relativ(PARAMS_PATH)}")
    print(f"Schritte:      {' -> '.join(SCHRITTE)}\n")

    try:
        manifest = fuehre_aus(
            mit_mlflow=not argumente.no_mlflow,
            lauf_name=argumente.name,
        )
    except Exception:
        sys.exit(1)

    print("\n── Zusammenfassung " + "─" * 42)
    for name, angaben in manifest["steps"].items():
        print(f"  {name:12s} {angaben['status']:16s} {angaben['duration_s']:>7.2f} s")
    print(f"  {'gesamt':12s} {'':16s} {manifest['duration_s']:>7.2f} s")

    kennzahlen = manifest["metrics"]
    print("\nKennzahlen auf der Validierungsmenge")
    print(f"  PR-AUC     {kennzahlen['pr_auc']:.4f}")
    print(f"  Recall     {kennzahlen['recall']:.4f}")
    print(f"  Precision  {kennzahlen['precision']:.4f}")
    print(f"  Kosten     {kennzahlen['kosten_eur']:,.0f}".replace(",", ".") + " EUR")
    print(f"  Schwelle   {kennzahlen['schwelle']}")

    print("\nQualitätsschranke: bestanden")
    print(f"Modell:   {manifest['model_file']}  (sha256 {manifest['model_sha256'][:12]}…)")
    print(f"Manifest: {manifest['_manifest_path']}")


if __name__ == "__main__":
    _main()
