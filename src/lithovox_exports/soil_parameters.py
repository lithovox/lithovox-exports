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

Sinds november 2026 koppel je de D-Stability-grondsoorten in LithoVox
Stability, niet meer in LithoVox: het `.nc` heeft dan alleen nog
D-Settlement. LithoVox Stability bewaart de koppelingen als een open
**koppelbestand** (`*.soil-links.json`): dezelfde JSON met één koppeling
("d_stability") en het model waar ze bij horen:

    {"format": "lithovox-soil-parameters", "version": 1,
     "model": {"file": "dijk.nc", "sha256": "…"},
     "mappings": [{"software": "d_stability", …, "soils": [...]}]}

`to_dstability` en `to_xslope` nemen het als `soil_links`
(read_soil_links leest en controleert het)."""

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


def read_soil_links(source: "str | Path | dict") -> dict:
    """Een koppelbestand van LithoVox Stability (`*.soil-links.json`, een pad)
    of de inhoud ervan (een dict), gelezen en gecontroleerd: formaat, versie,
    een D-Stability-koppeling en het model waar het bij hoort. ValueError
    als het niet te lezen is."""
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Het koppelbestand {path.name} is niet te lezen: {e}") from e
    _check(data, "De koppelingen")
    if mapping_of(data, "d_stability") is None:
        raise ValueError("Het koppelbestand heeft geen D-Stability-koppeling")
    model = data.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("sha256"), str):
        raise ValueError("Het koppelbestand zegt niet bij welk model het hoort")
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
