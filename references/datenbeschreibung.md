# Datenbeschreibung — AI4I 2020 Predictive Maintenance Dataset


## 1. Herkunft

| Quelle | UCI Machine Learning Repository, Datensatz 601 |
| URL (Datensatzseite) | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| URL (Direktdownload) | https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip |
| DOI | 10.24432/C5HS5C |
| Urheber | Stephan Matzka, HTW Berlin |
| Veröffentlichung | 2020 |
| Lizenz | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Abrufdatum | 19.09.2026 |


**Veröffentlichung:**

| Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications.
| In 2020 Third International Conference on Artificial Intelligence for Industries (AI4I). IEEE.


**Art der Daten:**

Der syntetische Datensatz spiegelt die realen Daten zur vorausschauenden Instandhaltung aus der Industrie wider, ist aber nicht an einer Maschine gemessen worden. Verteilungen und Ausfallmechanismen wurden nach festen Regeln erzeugt.

---

## 2. Eingefrorene Datei

| Pfad im Projekt | `data/raw/ai4i2020.csv` |
| Herkunft der Datei | entpackt aus dem oben verlinkten ZIP-Archiv (509,9 KB) |
| Dateigröße | 522.048 Byte |
| SHA-256 | dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e |
| Zeilen × Spalten | 10.000 × 14 |
| Kodierung | UTF-8, Trennzeichen `,`, Dezimaltrennzeichen `.` |

Hash und Eckdaten werden erzeugt mit:

```bash
python -m src.data.load
```

Derselbe Wert steht in `params.yaml` unter `data.raw_sha256` und wird beim Laden der Rohdaten geprüft (`src/data/load.py` liest ihn von dort als `EXPECTED_SHA256`). Er wird in jedem MLflow-Lauf protokolliert und als `provenance.raw_data_sha256` in das Pipeline-Manifest `reports/pipeline_run.json` geschrieben.

**Die Rohdatei wird nicht verändert.**

---

## 3. Spalten

Die Originalnamen enthalten Leerzeichen und Einheiten in Klammern. Beim Laden
werden sie über `COLUMN_MAPPING` in `src/data/load.py` auf technische Namen
abgebildet.

| Originalspalte | Technischer Name | Typ | Einheit | Beschreibung |
|---|---|---|---|---|
| `UDI` | `udi` | int | – | Laufende Nummer, 1–10.000 |
| `Product ID` | `product_id` | str | – | Produktkennung: Buchstabe der Qualitätsvariante + Seriennummer |
| `Type` | `type` | kategorial | – | Qualitätsvariante: `L` (low, ca. 50 %), `M` (medium, ca. 30 %), `H` (high, ca. 20 %) |
| `Air temperature [K]` | `air_temperature_k` | float | K | Umgebungstemperatur, normalisierter Random Walk um ca. 300 K |
| `Process temperature [K]` | `process_temperature_k` | float | K | Prozesstemperatur, ca. Lufttemperatur + 10 K |
| `Rotational speed [rpm]` | `rotational_speed_rpm` | int | 1/min | Drehzahl, aus einer Leistung von 2.860 W abgeleitet, mit Rauschen überlagert |
| `Torque [Nm]` | `torque_nm` | float | Nm | Drehmoment, normalverteilt um ca. 40 Nm, keine negativen Werte |
| `Tool wear [min]` | `tool_wear_min` | int | min | Werkzeugverschleiß; Varianten H/M/L addieren 5/3/2 Minuten je bearbeitetem Teil |
| `Machine failure` | `machine_failure` | int (0/1) | – | **Zielgröße:** Ist die Maschine in diesem Betriebszustand ausgefallen? |
| `TWF` | `twf` | int (0/1) | – | Tool Wear Failure — Verschleißausfall |
| `HDF` | `hdf` | int (0/1) | – | Heat Dissipation Failure — Ausfall durch mangelnde Wärmeabfuhr |
| `PWF` | `pwf` | int (0/1) | – | Power Failure — Leistung außerhalb des zulässigen Bereichs |
| `OSF` | `osf` | int (0/1) | – | Overstrain Failure — Überlastung (Verschleiß × Drehmoment) |
| `RNF` | `rnf` | int (0/1) | – | Random Failure — zufälliger Ausfall, unabhängig von den Prozessgrößen |

Fehlende Werte: keine.

---

## 4. Zielgröße und Klassenverhältnis

Zielgröße ist `machine_failure` (binäre Klassifikation). Der Anteil positiver
Fälle liegt bei ca. 3,4 % (339 von 10.000).

Folge für die Bewertung: Ein Modell, das immer „kein Ausfall" vorhersagt,
erreicht eine Accuracy von ca. 96,6 % und ist wertlos. Bewertet wird deshalb
über PR-AUC / Average Precision sowie über die erwarteten Kosten.

---

## 5. Hinweis zu Data Leakage

Die Spalten `TWF`, `HDF`, `PWF`, `OSF`, `RNF` beschreiben die Ursachen eines
Ausfalls. Sie entstehen gemeinsam mit dem Ausfall und sind zum
Vorhersagezeitpunkt nicht bekannt. Werden sie als Merkmale verwendet, entsteht
Data Leakage: die Kennzahlen steigen, das Modell ist im Betrieb nutzlos.

---

## 6. Wiederherstellung

Die Rohdatei liegt nicht im Git-Repository (`.gitignore`: `data/raw/*`). Um sie
wiederherzustellen:

Am einfachsten mit `python scripts/fetch_data.py` — das Skript erledigt die drei
Schritte unten und prüft den Hash selbst. Von Hand:

1. ZIP-Archiv unter der oben genannten Download-URL laden
2. entpacken, `ai4i2020.csv` nach `data/raw/` legen
3. `python -m src.data.load` ausführen

Für die Abgabe als ZIP über das LMS wird `data/raw/ai4i2020.csv` mit
eingepackt, damit die Pipeline ohne Internetzugang lauffähig ist.
