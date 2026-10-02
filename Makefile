# Ausführbare Dokumentation (Etappe 22).
#
# "make" allein zeigt alle Ziele mit Erklärung. Ein Makefile ist die kürzeste
# Form von Dokumentation, weil man sie nicht lesen, sondern ausführen kann - und
# weil sie dadurch nicht veralten kann, ohne aufzufallen.
#
# Voraussetzung: eine aktive virtuelle Umgebung (siehe "make setup").

.DEFAULT_GOAL := help
.PHONY: help setup daten pipeline test test-schnell lint format api drift drift-probe\
	abschluss docker docker-stop slicing notebooks abgabe clean

PYTHON ?= python

help:  ## Diese Übersicht
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}'

# ── Einrichten ────────────────────────────────────────────────────────
setup:  ## Virtuelle Umgebung anlegen und Pakete installieren
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt
	@echo
	@echo "Fertig. Jetzt aktivieren:  source .venv/bin/activate"

daten:  ## Rohdaten holen und gegen den eingefrorenen SHA-256 prüfen
	$(PYTHON) scripts/fetch_data.py

# ── Rechnen ───────────────────────────────────────────────────────────
pipeline:  ## Ganze Kette: Rohdaten -> geprüftes, gespeichertes Modell
	$(PYTHON) -m src.pipelines.run_pipeline

abschluss:  ## Einmalige Messung auf der Testmenge (nur einmal!)
	$(PYTHON) -m src.modeling.final_evaluation

# ── Prüfen ────────────────────────────────────────────────────────────
test:  ## Alle Tests
	pytest

test-schnell:  ## Tests ohne die, die ein Modell trainieren
	pytest -m "not langsam"

lint:  ## Statische Prüfung und Formatierung kontrollieren
	ruff check src tests scripts
	ruff format --check src tests scripts

format:  ## Formatieren und automatisch Behebbares beheben
	ruff check src tests scripts --fix
	ruff format src tests scripts

# ── Betreiben ─────────────────────────────────────────────────────────
api:  ## Schnittstelle starten (http://localhost:8000/docs)
	uvicorn src.serving.api:app --reload --port 8000

drift:  ## Protokollierte Anfragen gegen die Referenzstichprobe prüfen
	$(PYTHON) -m src.monitoring.drift

drift-probe:  ## Nachweisen, dass die Überwachung anschlägt
	$(PYTHON) -m src.monitoring.simulate_drift --details

docker:  ## Abbild bauen und starten
	docker compose up --build -d
	@echo "warte auf den Dienst ..."
	@until curl -fsS http://localhost:8000/health >/dev/null 2>&1; do sleep 1; done
	@curl -s http://localhost:8000/health
	@echo

docker-stop:  ## Container stoppen und entfernen
	docker compose down

# ── Abgeben ───────────────────────────────────────────────────────────
slicing:  ## Fehler nach Segmenten zerlegen -> reports/error_slicing.json
	python -m src.modeling.error_slicing

notebooks:  ## Prüfen, ob die Notebooks ihre Ergebnisse enthalten
	python scripts/check_notebooks.py

abgabe:  ## ZIP-Archiv für die Abgabe bauen (inklusive Rohdaten)
	$(PYTHON) scripts/build_submission.py

clean:  ## Zwischendateien entfernen (Daten, Modelle und Berichte bleiben)
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "aufgeräumt"
