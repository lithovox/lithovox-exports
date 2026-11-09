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
tekst). Het bedrijf ondertekent ze samen met de voxels."""

import json

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
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError("De grondsoortparameters van het model hebben een onbekend formaat")
    if data.get("version") != VERSION:
        raise ValueError(
            f"De grondsoortparameters van het model hebben versie {data.get('version')}; "
            f"dit pakket leest versie {VERSION}"
        )
    if not isinstance(data.get("mappings"), list):
        raise ValueError("De grondsoortparameters van het model hebben geen koppelingen")
    return data


def mapping_of(data: dict | None, software: str) -> dict | None:
    """De koppeling van één programma ("d_stability" of "d_settlement") uit
    `data` (zoals read_soil_parameters het geeft), of None."""
    for mapping in (data or {}).get("mappings") or []:
        if isinstance(mapping, dict) and mapping.get("software") == software:
            return mapping
    return None


def dstability_library(data: dict | None) -> dict | None:
    """De D-Stability-grondsoorten van het model als bibliotheek voor de
    exports: `soils` (elke gekoppelde grondsoort één keer, zoals in
    soils.json van een .stix) en `mapping` (modelgrondsoort → code, None voor
    geen). None als het model geen D-Stability-koppeling heeft."""
    mapping = mapping_of(data, "d_stability")
    if mapping is None:
        return None
    soils: dict[str, dict] = {}
    links: dict[str, str | None] = {}
    for entry in mapping.get("soils") or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("soil"), str):
            continue
        code = entry.get("code") or None
        parameters = entry.get("parameters")
        if code and not isinstance(parameters, dict):
            code = None
        links[entry["soil"]] = code
        # Meerdere modelgrondsoorten kunnen dezelfde grondsoort hebben: die komt er één keer in.
        if code and code not in soils:
            soils[code] = parameters
    return {"soils": list(soils.values()), "mapping": links}
