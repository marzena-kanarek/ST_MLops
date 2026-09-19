## Business Context

Ungeplante Maschinenausfälle können für Unternehmen erhebliche Kosten verursachen. Neben direkten Reparaturkosten können Produktionsunterbrechungen, Verzögerungen bei der Auftragsabwicklung und zusätzliche Wartungsaufwände entstehen.

**Predictive Maintenance** verfolgt das Ziel, mögliche Maschinenausfälle frühzeitig zu erkennen. Anstatt Wartungsarbeiten ausschließlich nach festen Zeitintervallen oder erst nach einem Ausfall durchzuführen, werden vorhandene Maschinen- und Sensordaten genutzt, um das Ausfallrisiko vorherzusagen.

In diesem Projekt wird ein Machine-Learning-Modell auf Basis des **AI4I 2020 Predictive Maintenance Dataset** entwickelt. Der Datensatz enthält verschiedene Merkmale einer Maschine, darunter Lufttemperatur, Prozesstemperatur, Rotationsgeschwindigkeit, Drehmoment und Werkzeugverschleiß. Zusätzlich enthält er Informationen darüber, ob ein Maschinenausfall aufgetreten ist.

Das Business-Ziel besteht darin, anhand dieser Daten frühzeitig Maschinen mit einem erhöhten Ausfallrisiko zu identifizieren. Dadurch könnten Wartungsmaßnahmen gezielter geplant und ungeplante Produktionsstillstände reduziert werden.

Das Projekt betrachtet Predictive Maintenance damit aus einer **Business- und Machine-Learning-Perspektive**:

* frühzeitige Erkennung potenzieller Maschinenausfälle
* Reduzierung ungeplanter Stillstandszeiten
* gezieltere Planung von Wartungsmaßnahmen
* bessere Nutzung vorhandener Maschinen- und Sensordaten
* Unterstützung von Wartungsentscheidungen durch Machine Learning

Ein besonderer Fokus liegt auf der **Erkennung von Ausfällen**, da in einem realen Wartungsszenario das Übersehen eines bevorstehenden Ausfalls erhebliche Auswirkungen haben kann. Deshalb werden neben Accuracy auch Metriken wie **ROC-AUC, PR-AUC / Average Precision, Precision, Recall und F1-Score** betrachtet.


######## Bei Predictive Maintenance ist insbesondere der Recall wichtig, da das Übersehen eines potenziellen Maschinenausfalls zu ungeplanten Produktionsstillständen und zusätzlichen Kosten führen kann. Der F1-Score ermöglicht hingegen eine ausgewogene Bewertung des Verhältnisses zwischen Precision und Recall.
