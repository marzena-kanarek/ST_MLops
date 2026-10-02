"""Tests der Container-Dateien (Etappe 21).

Ein Docker-Abbild kann man ohne Docker nicht bauen. Prüfen lässt sich aber das,
was beim Bauen schiefgeht — und der häufigste Fehler ist nicht die
Dockerfile-Syntax, sondern ein **fehlendes Paket** in der schlanken
Anforderungsdatei. Der Dienst startet dann im Container nicht, obwohl er auf dem
Entwicklungsrechner läuft, wo alles installiert ist.

Genau das war in Etappe 21 der Fall: ``requirements-api.txt`` aus der Vorlage
nannte kein PyYAML, obwohl ``src/config.py`` ``params.yaml`` liest. Der folgende
Test hätte das sofort gezeigt, ohne einen einzigen Build.
"""

from __future__ import annotations

import ast
import fnmatch
import re
import sys
from pathlib import Path

import pytest
import yaml

from src.config import PROJECT_ROOT

DOCKERFILE = PROJECT_ROOT / "Dockerfile.api"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
COMPOSE = PROJECT_ROOT / "docker-compose.yml"
ANFORDERUNGEN = PROJECT_ROOT / "requirements-api.txt"

#: Paketname auf Importname. Nicht jedes Paket heißt wie sein Modul.
IMPORTNAME = {
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv",
    "uvicorn[standard]": "uvicorn",
}

#: Module, die das Abbild nicht braucht, obwohl sie importiert werden: Sie kommen
#: als Abhängigkeit eines genannten Pakets mit.
MITGEBRACHT = {"pydantic", "starlette", "scipy", "annotated_types"}


def module_der_schnittstelle() -> set[str]:
    """Sammelt alle Importe der Dateien, die beim Start der API geladen werden.

    Ausgegangen wird von ``src/serving/api.py``; von dort werden die eigenen
    ``src.``-Importe weiterverfolgt. Das ergibt genau die Dateien, die im
    Container ausgeführt werden.
    """
    offen = [PROJECT_ROOT / "src/serving/api.py"]
    gesehen: set[Path] = set()
    fremd: set[str] = set()

    while offen:
        pfad = offen.pop()
        if pfad in gesehen or not pfad.exists():
            continue
        gesehen.add(pfad)

        baum = ast.parse(pfad.read_text(encoding="utf-8"))
        for knoten in ast.walk(baum):
            namen: list[str] = []
            if isinstance(knoten, ast.Import):
                namen = [a.name for a in knoten.names]
            elif isinstance(knoten, ast.ImportFrom) and knoten.module:
                namen = [knoten.module]
            for name in namen:
                wurzel = name.split(".")[0]
                if wurzel == "src":
                    teile = name.split(".")
                    offen.append(PROJECT_ROOT / Path(*teile).with_suffix(".py"))
                    offen.append(PROJECT_ROOT / Path(*teile) / "__init__.py")
                elif wurzel not in sys.stdlib_module_names and wurzel != "__future__":
                    fremd.add(wurzel)
    return fremd


def genannte_pakete() -> set[str]:
    """Die Importnamen der in requirements-api.txt genannten Pakete."""
    namen = set()
    for zeile in ANFORDERUNGEN.read_text(encoding="utf-8").splitlines():
        zeile = zeile.split("#")[0].strip()
        if not zeile:
            continue
        paket = re.split(r"[<>=!\[]", zeile)[0].strip().lower()
        namen.add(IMPORTNAME.get(paket, paket.replace("-", "_")))
    return namen


# ── Der Paketsatz ───────────────────────────────────────────────────────


def test_requirements_api_deckt_alle_importe_der_schnittstelle() -> None:
    """Jedes Modul, das die API lädt, muss im Abbild installiert sein.

    Fehlt eines, startet der Container nicht — und man merkt es erst nach dem
    Bauen, Hochladen und Starten.
    """
    gebraucht = module_der_schnittstelle() - MITGEBRACHT
    genannt = genannte_pakete()
    fehlend = sorted(gebraucht - genannt)
    assert not fehlend, (
        f"In requirements-api.txt fehlen: {fehlend}. "
        "Der Dienst würde im Container beim Start abbrechen."
    )


def test_abbild_enthaelt_keine_entwicklungswerkzeuge() -> None:
    """Das Laufzeitabbild soll schlank bleiben."""
    genannt = genannte_pakete()
    unerwuenscht = {
        "mlflow",
        "matplotlib",
        "seaborn",
        "pytest",
        "ruff",
        "jupyterlab",
        "notebook",
        "shap",
        "lime",
        "xgboost",
        "mkdocs",
    }
    assert not (genannt & unerwuenscht), (
        f"Im Laufzeitabbild haben diese Pakete nichts zu suchen: {sorted(genannt & unerwuenscht)}"
    )


# ── Dockerfile ──────────────────────────────────────────────────────────


def test_dockerfile_kopiert_nur_vorhandene_dateien() -> None:
    """Ein COPY auf eine fehlende Datei bricht den Bau ab."""
    quellen = re.findall(
        r"^COPY (?:--[\w=:-]+ )*(\S+)", DOCKERFILE.read_text(encoding="utf-8"), re.M
    )
    for quelle in quellen:
        if quelle.startswith("/") or quelle.startswith("--"):
            continue  # aus der ersten Stufe kopiert, nicht aus dem Kontext
        pfad = PROJECT_ROOT / quelle
        if quelle == "models/model.joblib" and not pfad.exists():
            pytest.skip("Kein Modell vorhanden: python -m src.pipelines.run_pipeline")
        assert pfad.exists(), f"COPY-Quelle fehlt: {quelle}"


def test_dockerfile_laeuft_nicht_als_root() -> None:
    """Ein Dienst, der nur Vorhersagen liefert, braucht keine Rootrechte."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    benutzer = re.findall(r"^USER (\S+)", text, re.M)
    assert benutzer, "Kein USER-Befehl — der Container würde als root laufen"
    assert benutzer[-1] != "root"


def test_healthcheck_fragt_den_health_endpunkt() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text
    assert "/health" in text.split("HEALTHCHECK", 1)[1]


def test_abhaengigkeiten_werden_vor_dem_code_kopiert() -> None:
    """Sonst installiert Docker bei jeder Codeänderung alles neu."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert text.index("requirements-api.txt") < text.index("COPY --chown=dienst:dienst src/")


# ── .dockerignore ───────────────────────────────────────────────────────


def _ausgeschlossen(pfad: str, muster: list[str]) -> bool:
    """Bildet die Auswertung von .dockerignore nach: das letzte Treffen gilt."""
    zustand = False
    for m in muster:
        negiert = m.startswith("!")
        roh = (m[1:] if negiert else m).rstrip("/")
        kandidaten = [pfad] + [str(p) for p in Path(pfad).parents if str(p) != "."]
        if any(fnmatch.fnmatch(k, roh) or fnmatch.fnmatch(k, roh + "/*") for k in kandidaten):
            zustand = not negiert
    return zustand


@pytest.fixture
def muster() -> list[str]:
    return [
        z.strip()
        for z in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if z.strip() and not z.strip().startswith("#")
    ]


@pytest.mark.parametrize(
    "pfad",
    [
        "requirements-api.txt",
        "params.yaml",
        "models/model.joblib",
        "src/serving/api.py",
        "src/config.py",
    ],
)
def test_dockerignore_laesst_die_copy_quellen_durch(pfad, muster) -> None:
    """Ein zu breites Ausschlussmuster ist der häufigste Grund für einen fehlgeschlagenen Bau."""
    assert not _ausgeschlossen(pfad, muster), f"{pfad} wird von .dockerignore ausgeschlossen"


@pytest.mark.parametrize(
    "pfad",
    [
        "data/raw/ai4i2020.csv",
        "notebooks/01_data_exploration.ipynb",
        "tests/test_api.py",
        ".venv/bin/python",
        "mlflow.db",
        "reports/api.log",
        "models/anderes_modell.joblib",
    ],
)
def test_dockerignore_haelt_ballast_draussen(pfad, muster) -> None:
    assert _ausgeschlossen(pfad, muster), f"{pfad} würde unnötig in den Bau-Kontext wandern"


# ── docker-compose.yml ──────────────────────────────────────────────────


def test_compose_baut_das_richtige_abbild_und_oeffnet_den_port() -> None:
    inhalt = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    api = inhalt["services"]["api"]
    assert api["build"]["dockerfile"] == DOCKERFILE.name
    assert "8000:8000" in api["ports"]


def test_compose_sichert_die_protokolle_ueber_neustarts() -> None:
    """Ohne Datenträger wären die protokollierten Vorhersagen mit dem Container weg."""
    inhalt = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    eingebunden = inhalt["services"]["api"]["volumes"]
    assert any("/app/reports" in str(e) for e in eingebunden)
    assert "protokolle" in inhalt["volumes"]


def test_compose_haelt_mlflow_hinter_einem_profil() -> None:
    """Die Oberfläche ist ein Zusatz und soll nicht bei jedem "up" mitstarten."""
    inhalt = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert inhalt["services"]["mlflow"]["profiles"] == ["tracking"]


def test_compose_nennt_keine_veraltete_version() -> None:
    """Der version-Eintrag ist in Compose V2 wirkungslos und erzeugt eine Warnung."""
    assert "version" not in yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
