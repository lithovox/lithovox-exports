"""Een dwarsdoorsnede als D-Stability-model (.stix), gemaakt met d-geolib:
de grondpolygonen van de doorsnede (zie profile.py) als lagen, elk met zijn
grondsoort, gekleurd zoals in het model (zonder arcering). Zonder
grondsoorten uit een .stix (zie read_stix_soils) hebben de grondsoorten
alleen een naam (de standaardparameters van D-Stability); met grondsoorten
gaan die allemaal met hun parameters in het bestand, en krijgt een laag de
grondsoort die de gebruiker aan zijn grondsoort koppelde (of, zonder
koppeling, die met dezelfde code of naam). Water, belastingen en de
rekeninstellingen laten we aan de gebruiker."""

import re
import tempfile
from pathlib import Path

from geolib.geometry.one import Point
from geolib.models.dstability import DStabilityModel
from geolib.models.dstability.internal import PersistableShadingTypeEnum, PersistableSoil, PersistableSoilVisualization
from geolib.soils.soil import Soil

from .profile import ProfileSlice, SoilPolygon

# D-Stability-coördinaten worden op zoveel decimalen afgerond (mm).
DECIMALS = 3


def soil_key(name: str | None) -> str:
    """Namen en codes van grondsoorten worden vergeleken zonder spaties aan de
    randen, ongeacht hoofdletters, en met spaties, underscores en streepjes
    als gelijk ("Zand antropogeen" = "zand_antropogeen")."""
    return re.sub(r"[\s_-]+", " ", (name or "").strip().lower())


def argb(color: str | None) -> str | None:
    """`#rrggbb` als `#AARRGGBB` van D-Stability (ondoorzichtig); None als het geen kleur is."""
    if not color or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return None
    return "#FF" + color[1:].upper()


def read_stix_soils(data: bytes) -> dict:
    """De grondsoorten van een D-Stability-bestand (.stix): `soils`, elke
    grondsoort met al zijn parameters (zoals in soils.json van het bestand;
    zonder kleuren: een export kleurt de grondsoorten zoals het model).
    ValueError als het bestand niet te lezen is."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "soils.stix"
        path.write_bytes(data)
        model = DStabilityModel()
        try:
            model.parse(path)
        except Exception as e:
            raise ValueError(f"Dit is geen D-Stability-bestand dat d-geolib kan lezen: {e}") from e
    return {"soils": [soil.model_dump(mode="json") for soil in model.datastructure.soils.Soils]}


def _add_soil_library(model: DStabilityModel, library: dict) -> list[PersistableSoil]:
    """Zet de grondsoorten van `library` (zoals read_stix_soils ze geeft) in `model`, met nieuwe ids."""
    added = []
    for data in library.get("soils") or []:
        soil = PersistableSoil(**data)
        soil.Id = str(model._get_next_id())
        model.datastructure.soils.Soils.append(soil)
        added.append(soil)
    return added


def library_soils(soils: list[PersistableSoil], mapping: dict[str, str | None] | None):
    """Zoekt de bibliotheekgrondsoort bij de naam van een modelgrondsoort: de
    grondsoort die `mapping` op code noemt (None of "": geen), of anders die
    waarvan de code of naam volgens soil_key gelijk is (een code wint). Ook
    gebruikt door de XSLOPE-export."""
    by_code = {soil.Code: soil for soil in reversed(soils) if soil.Code}
    codes = {soil_key(soil.Code): soil for soil in reversed(soils)}
    names = {soil_key(soil.Name): soil for soil in reversed(soils)}
    known = {**{k: v for k, v in names.items() if k}, **{k: v for k, v in codes.items() if k}}

    def find(name: str) -> PersistableSoil | None:
        if mapping is not None and name in mapping:
            code = mapping[name]
            if code and code not in by_code:
                raise ValueError(f'grondsoort "{name}" is gekoppeld aan "{code}", maar die staat niet in de grondsoorten')
            return by_code[code] if code else None
        return known.get(soil_key(name))

    return find


def write_stix(section: ProfileSlice, polygons: list[SoilPolygon], path: Path, label: str = "", library: dict | None = None) -> None:
    """Schrijft `polygons` (x = afstand tot de referentielijn, positief
    binnenwaarts; z = hoogte) als D-Stability-model naar `path`; met `library`
    (grondsoorten zoals read_stix_soils ze geeft) gaan al zijn grondsoorten
    erin, en gebruiken de lagen ze: de grondsoort die zijn `mapping`
    (naam van de modelgrondsoort → code in de bibliotheek, None voor geen)
    geeft, of anders die met dezelfde code of naam."""
    model = DStabilityModel()
    # Alleen de grondsoorten van de doorsnede (en de bibliotheek): niet de standaardgrondsoorten van D-Stability.
    model.datastructure.soils.Soils = []
    model.datastructure.soilvisualizations.SoilVisualizations = []
    find = library_soils(_add_soil_library(model, library) if library else [], (library or {}).get("mapping"))
    codes: dict[int, str] = {}
    coloured: set[str] = set()
    for soil in sorted({p.soil for p in polygons}):
        name = section.soils[soil]
        stix_soil = find(name)
        if stix_soil is not None:
            codes[soil] = stix_soil.Code
            soil_id = stix_soil.Id
        else:
            # Codes zijn uniek in een bestand: de bibliotheek kan deze naam al hebben (gekoppeld aan geen).
            taken = {s.Code for s in model.datastructure.soils.Soils}
            code, n = name, 1
            while code in taken:
                n += 1
                code = f"{name} ({n})"
            codes[soil] = code
            soil_id = model.add_soil(Soil(name=name, code=code))
        # Een grondsoort die meerdere modelgrondsoorten gebruiken krijgt de kleur van de eerste.
        color = argb(section.colors.get(name))
        if color and soil_id not in coloured:
            coloured.add(soil_id)
            model.datastructure.soilvisualizations.SoilVisualizations.append(
                PersistableSoilVisualization(Color=color, PersistableShadingType=PersistableShadingTypeEnum.NONE, SoilId=soil_id)
            )
    for polygon in polygons:
        points = [Point(x=round(x, DECIMALS), z=round(z, DECIMALS)) for x, z in polygon.points]
        model.add_layer(points, codes[polygon.soil], label=section.soils[polygon.soil])
    if label:
        model.datastructure.scenarios[0].Label = label
    model.serialize(path)
