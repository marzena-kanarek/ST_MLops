"""Einheitliche Protokollierung für den ganzen Dienst.

``print`` ist für Skripte in Ordnung und für einen laufenden Dienst falsch: Man
kann den Detailgrad nicht steuern, die Ausgabe trägt keinen Zeitstempel, und
umleiten lässt sie sich nur als Ganzes. Das ``logging``-Modul löst genau das —
eine Stelle legt Format, Stufe und Ziele fest, alle Module holen sich nur noch
einen Logger.

Zwei Ziele sind eingerichtet:

* **Bildschirm** — damit beim Entwickeln sichtbar ist, was passiert.
* **Datei** (``reports/api.log``) — damit es auch noch da ist, wenn das Terminal
  geschlossen ist.

Die Stufe kommt aus ``params.yaml`` (``logging.level``) und lässt sich über die
Umgebungsvariable ``LOG_LEVEL`` überschreiben, ohne die Konfiguration zu ändern.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.config import absolut, load_params

FORMAT = "%(asctime)s  %(levelname)-8s %(name)-24s %(message)s"
DATUMSFORMAT = "%Y-%m-%d %H:%M:%S"

#: Merker, damit mehrfaches Aufrufen nicht mehrfach dieselben Handler anhängt
#: (sonst erscheint jede Zeile zwei-, drei-, viermal).
_eingerichtet = False


def stufe(params: dict[str, Any] | None = None) -> int:
    """Die Protokollstufe: Umgebungsvariable hat Vorrang vor params.yaml."""
    aus_umgebung = os.environ.get("LOG_LEVEL")
    name = aus_umgebung or (params or load_params())["logging"]["level"]
    return getattr(logging, str(name).upper(), logging.INFO)


def protokolldatei(params: dict[str, Any] | None = None) -> Path:
    """Pfad der Anwendungsprotokolldatei aus params.yaml."""
    return absolut((params or load_params())["logging"]["app_log"])


def richte_ein(
    params: dict[str, Any] | None = None,
    mit_datei: bool = True,
    erneut: bool = False,
) -> None:
    """Richtet Format, Stufe und Ziele ein. Mehrfachaufruf ist harmlos.

    Args:
        params: Inhalt von params.yaml.
        mit_datei: False schreibt nur auf den Bildschirm (Tests).
        erneut: True erzwingt eine neue Einrichtung (Tests).
    """
    global _eingerichtet
    if _eingerichtet and not erneut:
        return

    params = params or load_params()
    wurzel = logging.getLogger()
    if erneut:
        for handler in list(wurzel.handlers):
            wurzel.removeHandler(handler)

    wurzel.setLevel(stufe(params))
    formatierer = logging.Formatter(FORMAT, datefmt=DATUMSFORMAT)

    bildschirm = logging.StreamHandler()
    bildschirm.setFormatter(formatierer)
    wurzel.addHandler(bildschirm)

    if mit_datei:
        pfad = protokolldatei(params)
        pfad.parent.mkdir(parents=True, exist_ok=True)
        datei = logging.FileHandler(pfad, encoding="utf-8")
        datei.setFormatter(formatierer)
        wurzel.addHandler(datei)

    _eingerichtet = True


def hole_logger(name: str) -> logging.Logger:
    """Gibt den Logger eines Moduls zurück, nach Bedarf nach Einrichtung."""
    richte_ein()
    return logging.getLogger(name)
