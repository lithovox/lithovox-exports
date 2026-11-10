"""De parameters van de gekoppelde grondsoorten in een LithoVox-model (.nc).

In LithoVox koppel je de grondsoortklassen van het model aan de grondsoorten
van een voorbeeldbestand van D-Stability (.stix) of D-Settlement (.sli). Het
model (formaatversie 3) bewaart dan bij de variabele `soil` het attribuut
`soil_parameters`, JSON:

    {
      "format": "lithovox-soil-parameters", "version": 1,
      "mappings": [
        {"software": "d_stability", "label": "D-Stability",
         "file": "dijk.stix", "sha256": "…", "loaded": "2026-11-08T12:00:00Z",
         "soils": [
           {"soil": "hollandveen", "link": "soil", "code": "VEEN", "name": "Veen",
            "parameters": {…de grondsoort zoals in soils.json van de .stix…}},
           {"soil": "basalton", "link": "none", "code": null}
         ]}
      ]
    }

Er is één item in `soils` voor elke grondsoort van het model (in de volgorde
van `soil_names`). `code` null betekent: geen grondsoort gekoppeld (`link`
zegt waarom: "none" met opzet, "suggested" een voorstel dat niet bevestigd
is, "undecided" nog niet gekozen). De parameters staan erin zoals in het
voorbeeldbestand: een D-Stability-grondsoort als zijn JSON-object, een
D-Settlement-grondsoort als lijst van [sleutel, waarde]-paren (waarden als
tekst). Het bedrijf ondertekent ze samen met de voxels.

Sinds november 2026 koppel je de D-Stability-grondsoorten niet meer in
LithoVox: het `.nc` heeft dan alleen nog D-Settlement. Voor de exports geef
je een eigen `.stix` of `.flox` met een koppeling (zie soil_library.py)."""

import json
from pathlib import Path

import h5py

ATTRIBUTE = "soil_parameters"
FORMAT = "lithovox-soil-parameters"
VERSION = 1


def read_soil_parameters(f: h5py.File) -> dict | None:
    """Het attribuut `soil_parameters` van een geopend model, gelezen en
    gecontroleerd; None als het model er geen heeft (er is in LithoVox geen
    voorbeeldbestand geladen). ValueError als het attribuut niet te lezen is."""
    if "soil" not in f or ATTRIBUTE not in f["soil"].attrs:
        return None
    raw = f["soil"].attrs[ATTRIBUTE]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"De grondsoortparameters van het model zijn niet te lezen: {e}") from e
    return _check(data, "De grondsoortparameters van het model")


def _check(data, what: str) -> dict:
    """`data` als grondsoortparameters gecontroleerd (formaat, versie,
    koppelingen); ValueError met wat er mis is."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{what} hebben een onbekend formaat")
    if data.get("version") != VERSION:
        raise ValueError(f"{what} hebben versie {data.get('version')}; dit pakket leest versie {VERSION}")
    if not isinstance(data.get("mappings"), list):
        raise ValueError(f"{what} hebben geen koppelingen")
    return data


def mapping_of(data: dict | None, software: str) -> dict | None:
    """De koppeling van één programma ("d_stability" of "d_settlement") uit
    `data` (zoals read_soil_parameters het geeft), of None."""
    for mapping in (data or {}).get("mappings") or []:
        if isinstance(mapping, dict) and mapping.get("software") == software:
            return mapping
    return None
