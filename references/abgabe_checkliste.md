# Abgabe-Checkliste

Geprüft am 2. Oktober 2026, gegen die Checkliste aus Teil 8 des Leitfadens.
Jeder Haken steht für eine **durchgeführte Prüfung**, nicht für eine Absicht; in
der Spalte „Beleg" steht, womit.

---

## Reproduzierbarkeit

| | Punkt | Beleg |
|---|---|---|
| ✅ | Frischer Klon in leerem Ordner läuft | Kopie mit genau den Dateien, die ein `git add -A` aufnimmt (88 Stück), frische virtuelle Umgebung, 169 Pakete aus `requirements.txt`, danach alle CI-Schritte in der Reihenfolge der Workflow-Datei. Einzige Abweichung: Die Rohdaten kamen aus einem lokalen Archiv, weil `archive.ics.uci.edu` aus dieser Umgebung gesperrt ist. |
| ✅ | Startwerte gesetzt, zwei Läufe dieselben Zahlen | `models/model.joblib` ist über Läufe **und über Umgebungen hinweg** bitgleich; `reference_sample.csv` ebenfalls, nachdem die Rundung ergänzt wurde (ohne sie schwankte die letzte Stelle um ein Maschinenepsilon, weil der Wald mit `n_jobs=-1` summiert). `tests/test_pipeline.py` und `tests/test_drift.py` prüfen beides. |
| ✅ | `requirements.txt` vorhanden, Versionen gepinnt | `requirements.txt` nennt Mindestversionen, `requirements-lock.txt` die exakt installierten, `requirements-api.txt` den schlanken Laufzeitsatz. |
| ✅ | Rohdaten unverändert, Herkunft und Hash dokumentiert | `references/datenbeschreibung.md`; der SHA-256 steht in `params.yaml` und wird bei **jedem** Laden geprüft, beim Holen (`scripts/fetch_data.py`), in der CI und im Manifest jedes Laufs. |

## Korrektheit

| | Punkt | Beleg |
|---|---|---|
| ✅ | Leakage geprüft und schriftlich begründet | Gemessen: PR-AUC 0,963 mit Ursachenspalten gegen 0,738 ohne. Notebook 01, `references/datenbeschreibung.md` Abschnitt 5, README Abschnitt 8. |
| ✅ | Testmenge genau einmal verwendet | `src/modeling/final_evaluation.py` verweigert einen zweiten Lauf; `--force` verlangt `--grund`, der in den Bericht geschrieben wird. Der Bericht hält einen zweiten Lauf fest: Er diente dem Vertrauensintervall, mit demselben Modell und denselben Daten — nach dem ersten Ergebnis wurde **keine** Entscheidung geändert. |
| ✅ | Metrik passt zum Klassenverhältnis | PR-AUC (Nulllinie 0,034) und erwartete Kosten in Euro statt Accuracy. README Abschnitt 1. |
| ✅ | Schwellenwert begründet, nicht 0,5 | 0,06 aus der Kostenkurve auf der Validierungsmenge; 126.200 € statt 170.100 €. Notebook 06, `params.yaml`. |
| ✅ | Baseline vorhanden, Modell schlägt sie | 0,037 (Zufall) → 0,860 (gewähltes Modell) auf derselben Menge. |

## Handwerk

| | Punkt | Beleg |
|---|---|---|
| ✅ | Tests grün, decken Daten / Merkmale / Modell / Schnittstelle ab | 130 Tests in 1,3 s über neun Dateien: Daten (9), Merkmale (8), Modell (13), Konfiguration (10), Kette (7), Schnittstelle (23), Protokollierung (15), Drift (23), Container (22). |
| ✅ | Linting grün | `ruff check src tests scripts` und `ruff format --check` ohne Beanstandung; `.pre-commit-config.yaml` hält das vor jedem Commit. |
| ⬜ | CI auf GitHub grün | **Offen — braucht einen Push.** Alle Schritte wurden lokal in einer frischen Umgebung nachgespielt und sind grün; der grüne Haken selbst erscheint erst nach `git push`. |
| ✅ | Keine Daten, Modelle oder Geheimnisse im Repository | `.gitignore` schließt `data/raw`, `data/processed`, `models/*.joblib`, `mlflow.db`, Betriebsprotokolle und `.env` aus. Eine bewusste Ausnahme: `data/processed/reference_sample.csv` (38 KB) ist der Maßstab der Driftprüfung und gehört zum Nachweis. Geheimnisse gibt es keine; `.env.example` nennt nur Schlüsselnamen. |
| ✅ | Alle Parameter in der Konfiguration | Suche nach magischen Zahlen in `src/` findet nur noch Indizes, Klassenlabels, physikalische Konstanten (2π/60) und technische Einstellungen. Begründete Ausnahme: die Suchräume der Zufallssuche (sie beschreiben, was einmalig durchsucht wurde, nicht womit gerechnet wird). |

## Betrieb

| | Punkt | Beleg |
|---|---|---|
| ✅ | Schnittstelle startet, `/docs` bedienbar, `/health` grün | Mit laufendem uvicorn geprüft: `/docs` 200 text/html, `/health` meldet `model_loaded: true`. Zusätzlich im nachgebauten Container-Inhalt mit dem schlanken Paketsatz. |
| ✅ | Ungültige Eingaben werden abgewiesen (422) | Fünf parametrisierte Fälle plus fehlendes Feld und Tippfehler im Feldnamen. Die Grenzen stammen aus dem Datenvertrag, nicht aus doppelt gepflegten Zahlen. |
| ✅ | Vorhersagen werden protokolliert | JSONL und SQLite, ein Eintrag je Zeile, im Hintergrund geschrieben. Nachgewiesen: drei Anfragen → drei Einträge, `pdm_predictions_total 3`. |
| ✅ | Überwachung erkennt simulierte Drift | `python -m src.monitoring.simulate_drift` prüft sich selbst und endet mit 1, wenn der Kontrollfall anschlägt oder ein Szenario unerkannt bleibt. |
| ✅ | Überwachungskonzept schriftlich, inklusive Neutrainings-Auslöser | `monitoring/monitoring_concept.md`, 246 Zeilen, mit den gemessenen Zahlen begründet. |

## Dokumentation

| | Punkt | Beleg |
|---|---|---|
| ✅ | README vollständig und getestet | Beantwortet die acht Fragen des Leitfadens; die Befehle wurden in der frischen Umgebung ausgeführt. |
| ❌ | Notebooks laufen mit „Restart & Run All" durch | **Offen und wichtig.** `python scripts/check_notebooks.py`: Die Notebooks 05 bis 08 enthalten **keine einzige Ausgabe** — als Beleg sind sie leer. In 01 bis 04 sind die Ausführungsnummern nicht in Folge. Beides lässt sich nur am eigenen Rechner beheben (05 rechnet eine Zufallssuche, 07 und 08 schreiben in MLflow). |
| ✅ | Zielarchitektur als Diagramm | README Abschnitt 6, als Mermaid-Diagramm — auf GitHub gerendert, Syntax und Darstellung geprüft. |
| ✅ | Bewusste Abweichungen und offene Punkte benannt | README Abschnitt 8 (dreizehn Entscheidungen mit ihrem Preis) und Abschnitt 10 (sechs offene Punkte und Grenzen). |

---

## Was noch zu tun ist

1. **Notebooks 05 bis 08 einmal durchlaufen lassen** und speichern, danach 01 bis
   04 mit „Restart & Run All" neu erzeugen. Prüfen mit
   `python scripts/check_notebooks.py` (Rückgabewert 0, wenn alles sitzt).
2. **Committen und pushen.** Danach läuft die CI an; der grüne Haken ist der
   letzte offene Punkt dieser Liste.
3. **Abgabearchiv neu bauen**, damit die durchgelaufenen Notebooks darin sind:
   `python scripts/build_submission.py`. Das Skript prüft das fertige Archiv
   selbst und gibt seinen SHA-256 aus.
