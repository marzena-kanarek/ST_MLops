# Überwachungskonzept

Vorausschauende Wartung, AI4I-2020-Prototyp. 

Dieses Dokument beantwortet fünf Fragen: **was** gemessen wird, **in welchem
Takt**, **welcher Schwellenwert welche Reaktion** auslöst, **wer** die Meldung
bekommt und **wann neu trainiert** wird. 


---

## 1. Was wird gemessen

Drei Ebenen, bewusst alle drei. Jede allein führt in die Irre.

### 1.1 Datendrift — verschieben sich die Eingangsverteilungen?

| Maß | Für | Was es sagt |
|---|---|---|
| **PSI** (Population Stability Index) | alle Merkmale | *wie stark* sich eine Verteilung verschoben hat |
| **Kolmogorow-Smirnow** | numerische Merkmale | ob die Verschiebung bei dieser Stichprobengröße von Zufall zu unterscheiden ist |
| **Chi²** | `type` | dasselbe für kategoriale Merkmale |

PSI und Test ergänzen sich: Der Test allein wird bei großen Stichproben schon
bei fachlich bedeutungslosen Verschiebungen signifikant, der PSI allein sagt
nichts über die Verlässlichkeit der Messung.

Maßstab ist die **Referenzstichprobe** `data/processed/reference_sample.csv`:
1.000 Zeilen aus der Validierungsmenge, bei jedem Pipelinelauf neu geschrieben
und mit im Git. Bewusst die Validierungs- und nicht die Trainingsmenge — auf
gelernten Zeilen sagt ein Random Forest beinahe 0 oder 1, diese Verteilung wäre
als Betriebsmaßstab unbrauchbar.

### 1.2 Vorhersagedrift — verschiebt sich die Verteilung der Wahrscheinlichkeiten?

PSI über die ausgegebenen Wahrscheinlichkeiten, dazu die Alarmquote. Der
entscheidende Vorzug: **das sieht man sofort**, ohne auf wahre Labels zu warten.

### 1.3 Betriebskennzahlen

Aus `/metrics` (Prometheus-Format): Alarmquote, Antwortzeit, Fehlerquote
(Statuscodes ≥ 400), Zahl der Vertragswarnungen, fehlgeschlagene
Protokollschreibvorgänge.

### 1.4 Modellgüte

PR-AUC, Recall, Precision und erwartete Kosten — messbar **erst**, wenn wahre
Labels vorliegen.

---

## 2. In welchem Takt

| Was | Takt | Wie |
|---|---|---|
| Betriebskennzahlen | laufend | `/metrics`, von einer Prometheus-Instanz abgeholt |
| Vorhersagedrift | täglich | `python -m src.monitoring.drift` gegen das Protokoll |
| Datendrift je Merkmal | täglich | derselbe Lauf |
| Modellgüte | monatlich, sobald Labels da sind |
| Referenzstichprobe erneuern | bei jedem Neutraining | Pipelineschritt `referenz` |

Unter **100 aktuellen Zeilen** (`monitoring.min_rows`) wird nicht geurteilt. Bei
20 Zeilen schlägt jeder Test irgendwann zufällig an; ein Fehlalarm kostet
Vertrauen in die Überwachung.

---

## 3. Welcher Schwellenwert löst welche Reaktion aus

### Die Schwellen

| Kennzahl | stabil | beobachten | handeln |
|---|---|---|---|
| PSI je Merkmal | < 0,10 | 0,10 – 0,25 | > 0,25 |
| PSI der Vorhersagen | < 0,10 | 0,10 – 0,25 | > 0,25 |
| KS / Chi² | p ≥ 0,05 | — | p < 0,05 **zusammen mit** PSI ≥ 0,10 |
| Alarmquote | 3 – 10 % | Abweichung > ×2 | > 25 % oder < 1 % |
| Antwortzeit (Median) | < 50 ms | 50 – 200 ms | > 200 ms |
| Fehlerquote | < 1 % | 1 – 5 % | > 5 % |

Die Alarmquote der Referenz liegt bei **6,1 %**. Die untere Grenze ist so
wichtig wie die obere: Ein Dienst, der fast nie Alarm gibt, ist verdächtig und
nicht beruhigend.

### Die Reaktionen

- **stabil** — Eintrag in `reports/drift_report.json`, nichts weiter.
- **beobachten** — Vermerk im Wochenbericht, Beobachtung über drei aufeinander
  folgende Läufe. Reagiert wird erst, wenn die Verschiebung bleibt. Einmalige
  Ausschläge sind meist Stichprobenrauschen.
- **handeln** — Meldung an die Modellverantwortliche innerhalb eines Werktags.
  Erster Schritt ist **immer** die Ursachenprüfung, nicht das Neutraining:
  Sensortausch, Kalibrierung, Produktionsumstellung, Datenpipelinefehler.
  Ein Modell auf fehlerhafte Daten neu zu trainieren macht den Fehler dauerhaft.

### Warum alle drei Ebenen — mit gemessenen Zahlen

Belegt durch `python -m src.monitoring.simulate_drift` auf der
Validierungsmenge (1.500 Zeilen), Schwellenwert 0,06:

| Szenario | größtes PSI | Alarmquote | Recall | Kosten | Urteil |
|---|---|---|---|---|---|
| keine Drift (Kontrolle) | 0,002 | 5,7 % | 0,863 | 126.200 € | stabil |
| Prozesstemperatur 3 K zu hoch | 4,689 (`process_temperature_k`) | **6,0 %** | **0,627** | 244.600 € | handeln |
| Werkzeugverschleiß +60 min | 3,772 (`tool_wear_min`) | 32,3 % | 0,961 | 277.200 € | handeln |
| 80 % Variante H | 2,950 (`type`) | 5,7 % | **0,863** | 126.200 € | handeln |

**Zeile 2 ist der Grund für dieses Konzept.** Ein neu kalibrierter Sensor meldet
3 K zu viel. Die Temperaturdifferenz wächst dadurch, und eine große Differenz
bedeutet für das Modell gute Wärmeabfuhr — Entwarnung. Die Alarmquote bleibt bei
6,0 % statt 5,7 % praktisch unverändert; auf jedem Betriebsschaubild sieht das
nach Normalbetrieb aus. Dabei findet das Modell nur noch **63 % statt 86 %** der
Ausfälle, und die erwarteten Kosten verdoppeln sich fast. Wer nur
Betriebskennzahlen überwacht, sieht einen ruhigen Dienst und übersieht, dass das
Modell blind geworden ist. Die Datendrift zeigt es sofort: PSI 4,689.

**Zeile 4 ist die Gegenrichtung** und genauso wichtig. Die Typenverteilung
verschiebt sich massiv (PSI 2,950), Recall und Kosten bleiben auf die vierte
Stelle gleich: Das Merkmal trägt kaum zur Vorhersage bei. Ein PSI-Alarm heißt
*„die Eingangsdaten haben sich verändert"*, nicht *„das Modell ist schlechter
geworden"*. Deshalb löst er eine **Prüfung** aus und kein automatisches
Neutraining — sonst trainiert man gegen Rauschen.

Die Selbstprüfung des Simulationsskripts hält beides fest: Der Kontrollfall muss
ruhig bleiben, alle drei Szenarien müssen erkannt werden. Andernfalls endet es
mit Rückgabewert 1.

### Was die Simulation nicht zeigt

Die wahren Labels werden nicht mitsimuliert — sie gehören zu den ursprünglichen
Betriebspunkten. Der Recall nach einer Verschiebung sagt also *„so empfindlich
reagiert das Modell auf diese Verschiebung"* und nicht *„so viele Ausfälle gäbe
es in einem so veränderten Prozess"*. Für den Nachweis, dass die Überwachung
anschlägt, genügt das; für eine Aussage über den veränderten Prozess bräuchte es
ein Simulationsmodell der Maschine.

---

## 4. Wer bekommt die Meldung

| Rolle | Zuständig für | Erreicht über |
|---|---|---|
| Betrieb / Bereitschaft | Dienst erreichbar, Antwortzeit, Fehlerquote | Prometheus-Alarm |
| Modellverantwortliche | Daten- und Vorhersagedrift, Güte, Neutraining | Driftbericht, Meldung bei „handeln" |
| Instandhaltung | Alarme je Maschine, Rückmeldung zum Befund | Fachanwendung |
| Datenverantwortliche | Sensoren, Kalibrierung, Datenpipeline | Meldung bei Vertragsverletzung |

**Offen und bewusst so vermerkt:** Dieses Projekt ist ein Prototyp ohne
betriebliche Einbettung. Die Rollen sind benannt, aber keiner Person zugeordnet,
und es gibt keinen Bereitschaftsplan. Die Zuordnung ist eine organisatorische
Entscheidung, die der Betreiber treffen muss — sie hier zu erfinden wäre eine
Scheingenauigkeit.

---

## 5. Wann wird neu trainiert

Drei Auslöser, in dieser Rangfolge:

1. **Nach Schwellenwertverletzung** — PSI > 0,25 auf einem Merkmal, das
   nachweislich zur Vorhersage beiträgt, über drei Läufe hinweg, **und** nach
   abgeschlossener Ursachenprüfung. Ein Sensorfehler wird repariert, nicht
   antrainiert.
2. **Nach Datenmenge** — sobald mindestens 2.000 neue Betriebspunkte mit
   bestätigtem Befund vorliegen (etwa 20 % des ursprünglichen Datensatzes).
3. **Nach Zeitplan** — halbjährlich, auch ohne Auffälligkeit. Ein Lauf, der
   nichts verändert, ist der Beweis, dass die Kette noch funktioniert; dieser
   Nachweis verfällt, wenn man ihn nie führt.

Jedes Neutraining läuft über `python -m src.pipelines.run_pipeline` und damit
durch die **Qualitätsschranke** aus Etappe 12: PR-AUC ≥ 0,75, Recall ≥ 0,80,
erwartete Kosten ≤ 200.000 €. Ein schlechteres Modell überschreibt das bisherige
nicht — es wird weder gespeichert noch als `champion` registriert. Ein Rückfall
auf die Vorgängerversion ist über das MLflow-Modellregister ein Handgriff.

Nach jedem Neutraining wird die Referenzstichprobe erneuert, und der
Entscheidungsschwellenwert wird auf den neuen Kosten neu bestimmt — er ist nicht
dauerhaft gültig, sondern ein Ergebnis der jeweiligen Datenlage.

---

## 6. Woher kommen die wahren Labels — die größte Lücke

In diesem Prototyp liegen die Labels im Datensatz. Im Betrieb nicht, und das ist
die ehrlichste Aussage dieses Dokuments:

- Ein bestätigter Ausfall entsteht aus der **Instandhaltungsrückmeldung**:
  Maschine stand, Ursache aufgenommen, Auftrag abgeschlossen. Das dauert je nach
  Organisation **Tage bis Wochen**.
- Ein **verhinderter** Ausfall liefert gar kein eindeutiges Label. Hat der Alarm
  einen Schaden abgewendet, oder war er falsch? Beides sieht hinterher gleich
  aus. Das ist kein Messproblem, sondern eine Eigenart vorausschauender
  Wartung — nur ein Teil der Vorhersagen wird je überprüfbar.
- Fehlalarme werden systematisch **unter**berichtet: Wer nachsieht und nichts
  findet, dokumentiert das selten.

Folgen, die man aushalten muss:

1. **PR-AUC und Recall im Betrieb hinken wochenlang nachher.** Deshalb sind
   Daten- und Vorhersagedrift die **Frühindikatoren** — sie brauchen keine
   Labels. Die Güte bestätigt erst später, was die Drift vermutet hat.
2. **Die Rückmeldung muss organisatorisch erzwungen werden.** Ohne einen Prozess,
   der jeden Alarm mit Befund „bestätigt / unbestätigt" zurückschreibt, gibt es
   kein Neutraining auf Betriebsdaten — nur noch Neutraining auf den alten Daten.
   Das ist eine organisatorische Lücke, keine technische.
3. **`RNF` setzt eine Obergrenze.** Die Zufallsausfälle im Datensatz sind
   definitionsgemäß nicht vorhersagbar. Kein Recall von 1,0 ist erreichbar, und
   ein Modell, das in diese Richtung getrieben wird, lernt Rauschen.

Zusätzlich gilt für die Übertragbarkeit: Der AI4I-2020-Datensatz ist
**synthetisch** und hat keine Zeitachse. Saisonale Drift, Alterung über Monate
und Wechselwirkungen zwischen Maschinen lassen sich daran nicht untersuchen. Die
hier festgelegten Schwellen sind damit plausibel begründet, aber nicht an einer
realen Anlage kalibriert — die erste Betriebsphase dient genau dazu.

---

## 7. Werkzeuge

```bash
# Protokollierte Anfragen gegen die Referenz prüfen
python -m src.monitoring.drift

# Eine Datei gegen die Referenz prüfen
python -m src.monitoring.drift --aktuell data/processed/val.parquet

# Nachweis, dass die Überwachung anschlägt (endet mit 1, wenn nicht)
python -m src.monitoring.simulate_drift --details

# Betriebskennzahlen
curl http://localhost:8000/metrics

# Protokoll auswerten
python -m src.serving.prediction_log
```

Berichte: `reports/drift_report.json`, `reports/drift_simulation.json`.
Schwellen und Pfade: `params.yaml`, Abschnitt `monitoring`.
