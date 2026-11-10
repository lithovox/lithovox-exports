"""Grondsoorten met hun parameters uit een eigen bestand: een D-Stability-
(`.stix`) of D-Geo Flow-bestand (`.flox`) van de gebruiker, en een koppeling
`{grondsoort in het model: code van de grondsoort in dat bestand}`.

    doorsnede.to_dstability("km0.250.stix", soils="grondsoorten.stix",
                            mapping={"klei": "K_Ozw", "zand": "Z_Pleist", "veen": None})

De extensie van het bestand zegt wat het is; het moet hetzelfde programma
zijn als de export (een `.stix` voor D-Stability, een `.flox` voor D-Geo
Flow): de parameters van het ene programma zijn niet die van het andere.
Een grondsoort van het model die de koppeling niet noemt, of die ze aan
None koppelt, krijgt alleen zijn naam (de standaardparameters van het
programma). Alleen de gebruikte grondsoorten gaan in de export, niet de
hele bibliotheek. De koppeling kan ook een JSON-bestand zijn met hetzelfde
object."""

import json
from pathlib import Path

# Een grondsoortenbestand groter dan dit lezen we niet (het is een zip van JSON).
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_MAPPING_BYTES = 1024 * 1024

PROGRAMS = {".stix": "D-Stability", ".flox": "D-Geo Flow"}


def read_soil_file(path: str | Path, suffix: str) -> dict[str, dict]:
    """De grondsoorten van het bestand `path`, per code, elk met al zijn
    parameters (zoals in soils.json van het bestand). `suffix` is de
    extensie die de export vraagt (".stix" of ".flox"). ValueError als het
    bestand niet van dat programma is of niet te lezen."""
    path = Path(path)
    program = PROGRAMS[suffix]
    if path.suffix.lower() != suffix:
        raise ValueError(
            f"De grondsoorten voor {program} komen uit een {suffix}-bestand, niet uit "
            f"{path.name}: de parameters van het ene programma passen niet in het andere"
        )
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ValueError(f"Het grondsoortenbestand {path.name} is niet te lezen: {e}") from e
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Het grondsoortenbestand {path.name} is te groot ({size // 2**20} MB)")
    if suffix == ".stix":
        from geolib.models.dstability import DStabilityModel as Program
    else:
        from geolib.models.dgeoflow import DGeoFlowModel as Program
    model = Program()
    try:
        model.parse(path)
    except Exception as e:  # GEOLib geeft allerlei fouten op een slecht bestand
        raise ValueError(
            f"{path.name} is geen {program}-bestand dat GEOLib kan lezen "
            f"(een oud bestand: open en bewaar het eerst in {program}): {e}"
        ) from e
    return {soil.Code: soil.model_dump(mode="json") for soil in model.datastructure.soils.Soils}


def read_mapping(source: "str | Path | dict") -> dict[str, str | None]:
    """De koppeling als dict, of uit een JSON-bestand met hetzelfde object;
    gecontroleerd: namen en codes zijn tekst, een code mag ook None (geen)
    zijn. ValueError als het niet klopt."""
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        try:
            if path.stat().st_size > MAX_MAPPING_BYTES:
                raise ValueError(f"Het koppelbestand {path.name} is te groot")
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Het koppelbestand {path.name} is niet te lezen: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            'De koppeling is geen object {"grondsoort in het model": "code in het bestand"}'
        )
    for soil, code in data.items():
        if not isinstance(soil, str) or not (code is None or isinstance(code, str)):
            raise ValueError(
                f"De koppeling van {soil!r} is {code!r}: verwacht de code van een grondsoort (tekst) of null"
            )
    return {soil: (code or None) for soil, code in data.items()}


def linked_soils(
    model_soils: list[str],
    soils: "str | Path | None",
    mapping: "str | Path | dict | None",
    suffix: str,
) -> dict[str, dict]:
    """Per grondsoort van het model die gekoppeld is: de parameters van zijn
    grondsoort uit het bestand. Leeg zonder bestand en koppeling. ValueError
    bij een van de twee zonder de ander, een naam die geen grondsoort van het
    model is of een code die het bestand niet heeft."""
    if soils is None and mapping is None:
        return {}
    if soils is None or mapping is None:
        raise ValueError(
            "Geef het grondsoortenbestand (soils) en de koppeling (mapping) allebei, of geen van beide"
        )
    links = read_mapping(mapping)
    unknown = sorted(set(links) - set(model_soils))
    if unknown:
        raise ValueError(
            f"De koppeling noemt {', '.join(unknown)}, maar dat is geen grondsoort van het model "
            f"(die zijn: {', '.join(model_soils)})"
        )
    library = read_soil_file(soils, suffix)
    missing = sorted({code for code in links.values() if code and code not in library})
    if missing:
        raise ValueError(
            f"{Path(soils).name} heeft geen grondsoort met de code {', '.join(missing)}"
        )
    return {soil: library[code] for soil, code in links.items() if code}


def layer_soils(names: list[str], linked: dict[str, dict]) -> tuple[list[dict], dict[str, str]]:
    """De grondsoorten voor een export met de modelgrondsoorten `names` (de
    gebruikte, in de volgorde van het model): elk `{"code", "name", "parameters" (of None:
    alleen de naam), "color_of" (de modelgrondsoort wiens kleur hij krijgt)}`,
    en de code per modelgrondsoort. Twee modelgrondsoorten aan dezelfde
    grondsoort delen hem (met de kleur van de eerste); een niet gekoppelde
    grondsoort heeft zijn naam als code, met " (2)" enzovoort als een
    gekoppelde grondsoort die code al heeft (codes zijn uniek in een bestand)."""
    soils: list[dict] = []
    codes: dict[str, str] = {}
    for name in names:
        parameters = linked.get(name)
        if parameters is not None:
            code = parameters["Code"]
            if code not in {s["code"] for s in soils}:
                soils.append({"code": code, "name": parameters.get("Name") or code,
                              "parameters": parameters, "color_of": name})
            codes[name] = code
    taken = {s["code"] for s in soils}
    for name in names:
        if name in codes:
            continue
        code, n = name, 1
        while code in taken:
            n += 1
            code = f"{name} ({n})"
        taken.add(code)
        soils.append({"code": code, "name": name, "parameters": None, "color_of": name})
        codes[name] = code
    return soils, codes
