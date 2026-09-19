# Predictive Maintenance — MLOps-Prototyp

Prototyp eines vollständigen ML-Systems zur vorausschauenden Wartung: von den
eingefrorenen Rohdaten über Training und Bewertung bis zur lokalen
Schnittstelle, mit Experiment-Tracking, Tests, CI/CD und Überwachung.

Entstanden als Prüfungsleistung im Modul MLOps (Digital Business University of
Applied Sciences).

---

## 1. Fachliche Fragestellung

### Frage

> Fällt diese Maschine in ihrem aktuellen Betriebszustand aus?

Vorhergesagt wird aus Sensor- und Prozessgrößen eines einzelnen Betriebspunkts
— Lufttemperatur, Prozesstemperatur, Drehzahl, Drehmoment, Werkzeugverschleiß
und Qualitätsvariante des Werkstücks.

### Zielgröße

`machine_failure` (0/1) aus dem AI4I-2020-Datensatz. Binäre Klassifikation.

Die Ursachenspalten `TWF`, `HDF`, `PWF`, `OSF`, `RNF` werden **nicht** als
Merkmale verwendet: sie entstehen gemeinsam mit dem Ausfall und sind zum
Vorhersagezeitpunkt nicht bekannt. Ihre Verwendung wäre Data Leakage.

### Klassenverhältnis

339 von 10.000 Betriebszuständen sind Ausfälle — **3,39 % positive Fälle**.

Daraus folgt unmittelbar: **Accuracy scheidet als Metrik aus.** Ein Modell, das
pauschal „kein Ausfall" vorhersagt, erreicht 96,61 % Accuracy und ist
vollkommen wertlos.

### Fehlerkosten

| Fehlerart | Bedeutung | Angenommene Kosten |
|---|---|---|
| Falsch negativ (FN) | Ausfall übersehen — ungeplanter Stillstand, Folgeschäden | 10.000 € |
| Falsch positiv (FP) | Fehlalarm — unnötige Prüfung, kurzer Produktionsstopp | 500 € |
| Richtig positiv (TP) | Ausfall erkannt — geplante Wartung | 800 € |

> **Annahme, nicht Messung.** Die Werte sind geschätzt und stammen nicht aus
> betrieblichen Daten. Entscheidend ist nicht ihre absolute Höhe, sondern das
> Verhältnis von **20 zu 1** zwischen übersehenem Ausfall und Fehlalarm. Es
> bestimmt die Wahl der Metrik und den Entscheidungsschwellenwert.

### Bewertungsmetrik

**Primär: PR-AUC (Average Precision).** Bei 3,39 % positiven Fällen bewertet
die Precision-Recall-Kurve genau das, worauf es ankommt — wie gut das Modell
die seltene Klasse findet, ohne von der übergroßen Mehrheit der Normalfälle
geschmeichelt zu werden.

**Sekundär: erwartete Kosten in Euro.** Sie übersetzen die Modellgüte in die
Größe, über die im Betrieb tatsächlich entschieden wird, und machen das
Kostenverhältnis von 20 zu 1 explizit.

Begleitend werden Recall und Precision am gewählten Schwellenwert berichtet.
ROC-AUC wird nur nachrichtlich geführt: bei stark unausgeglichenen Klassen
fällt sie optimistisch aus.

### Erfolgsschwelle

Vor dem ersten Trainingslauf festgelegt:

| Kriterium | Mindestwert |
|---|---|
| PR-AUC auf der Validierungsmenge | ≥ 0,75 |
| Recall am gewählten Schwellenwert | ≥ 0,80 |

Ein Modell, das diese Werte nicht erreicht, wird nicht als Champion
registriert. Die Schwellen wandern in Etappe 12 als ausführbare
Qualitätsschranke nach `params.yaml`.

---

## 2. Datengrundlage

AI4I 2020 Predictive Maintenance Dataset (Matzka 2020), UCI Machine Learning
Repository, CC BY 4.0. 10.000 Zeilen, 14 Spalten, synthetisch erzeugt.

Die Rohdatei liegt unverändert unter `data/raw/ai4i2020.csv` und ist über
ihren SHA-256-Hash eingefroren. Herkunft, Lizenz, Zitation, Spaltenbedeutungen
und Hash sind in [`references/datenbeschreibung.md`](references/datenbeschreibung.md)
dokumentiert.

---

## 3. Projektstruktur

```
data/
├── raw/              Rohdaten, unverändert (nicht im Git)
├── processed/        abgeleitete Daten, jederzeit neu erzeugbar
└── external/         Fremdquellen
notebooks/            Werkstatt: Erkundung, Belege, Erklärungen
src/                  Fabrik: importierbarer, testbarer Code
├── data/             laden, prüfen, aufbereiten, aufteilen
├── features/         Merkmalskonstruktion
├── modeling/         Training, Abstimmung, Bewertung, Tracking
├── serving/          Schnittstelle, Schemas, Modell-Service
├── monitoring/       Drift, Betriebskennzahlen
├── pipelines/        Orchestrierung
└── utils/            Logging, Reproduzierbarkeit
tests/                pytest
models/               trainierte Artefakte (nicht im Git)
reports/              Kennzahlen, Abbildungen, Laufmanifeste
references/           Datenbeschreibung und Quellen
scripts/              einmalige Werkzeuge
```

Die Trennung ist keine Kosmetik: Notebooks sind die Werkstatt, `src/` ist die
Fabrik. Sobald etwas im Notebook funktioniert, wandert es als Funktion nach
`src/`, und das Notebook importiert sie zurück.

---

## 4. Einrichtung

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` nennt die benötigten Pakete mit Mindestversionen,
`requirements-lock.txt` die exakt installierten. Für einen reproduzierbaren
Nachbau die Sperrdatei verwenden:

```bash
python -m pip install -r requirements-lock.txt
```

Rohdaten wiederherstellen: siehe Abschnitt 6 in
`references/datenbeschreibung.md`.

---

## 5. Nutzung

```bash
# Rohdaten prüfen und Eckdaten ausgeben
python -m src.data.load
```

Weitere Befehle kommen mit den folgenden Etappen hinzu (Trainingslauf,
Pipeline, Schnittstelle, Überwachung).

---

## 6. Stand der Umsetzung

| | Etappe | Status |
|---|---|---|
| 1 | Problem und Metrik festlegen | erledigt |
| 2 | Werkzeuge und Repository einrichten | erledigt |
| 3 | Daten holen und einfrieren | erledigt |
| 4 | Datenvertrag und Validierung | offen |
| 5 | Explorative Analyse | offen |
| 6 | Leakage prüfen, Daten aufteilen | offen |
| 7 | Baseline bauen | offen |
| 8 | Merkmale konstruieren | offen |
| 9 | Modelle vergleichen und abstimmen | offen |
| 10 | Entscheidungsregel festlegen | offen |
| 11 | Experimente nachvollziehbar machen | offen |
| 12 | Qualitätsschranke und Modellregister | offen |
| 13 | Tests schreiben | offen |
| 14 | Konfiguration zentralisieren | offen |
| 15 | Pipeline als ein Befehl | offen |
| 16 | Modell als Schnittstelle bereitstellen | offen |
| 17 | Protokollierung und Kennzahlen | offen |
| 18 | Überwachung und Drift | offen |
| 19 | Formatierung und statische Prüfung | offen |
| 20 | CI/CD mit GitHub Actions | offen |
| 21 | Container | offen |
| 22 | Dokumentation und Abgabe | offen |

---

## 7. Hinweise

Der Datensatz ist synthetisch. Er bildet eine Fräsmaschine nach festen Regeln
nach und ist nicht an einer realen Anlage gemessen. Reale Sensordaten wären
verrauschter, lückenhafter und würden zusätzliche Schritte zur Datenbereinigung
erfordern. Die Ergebnisse dieses Prototyps sind deshalb nicht unbesehen auf
einen Produktionsbetrieb übertragbar.

Dokumentation, Code-Kommentare und Notebooks sind durchgehend deutschsprachig.
