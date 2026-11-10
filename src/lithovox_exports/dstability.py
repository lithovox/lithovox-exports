"""Een dwarsdoorsnede als D-Stability-model (.stix), gemaakt met GEOLib
(Deltares, d-geolib op PyPI): de grondpolygonen van de doorsnede (zie
profile.py) als lagen, elk met zijn grondsoort, gekleurd zoals in het model
(zonder arcering). Een gekoppelde grondsoort (zie soil_library.py) krijgt de
grondsoort uit het bestand van de gebruiker met al zijn parameters; de
andere hebben alleen een naam (de standaardparameters van D-Stability).
Water, belastingen en de rekeninstellingen laten we aan de gebruiker."""

import re
from pathlib import Path

from geolib.geometry.one import Point
from geolib.models.dstability import DStabilityModel
from geolib.models.dstability.internal import (
    PersistableShadingTypeEnum,
    PersistableSoil,
    PersistableSoilVisualization,
)
from geolib.soils.soil import Soil

from .profile import ProfileSlice, SoilPolygon
from .soil_library import layer_soils

# Coördinaten worden op zoveel decimalen afgerond (mm).
DECIMALS = 3


def argb(color: str | None) -> str | None:
    """`#rrggbb` als `#AARRGGBB` van D-Stability en D-Geo Flow (ondoorzichtig); None als het geen kleur is."""
    if not color or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return None
    return "#FF" + color[1:].upper()


def write_stix(
    section: ProfileSlice,
    polygons: list[SoilPolygon],
    path: Path,
    label: str = "",
    linked: dict[str, dict] | None = None,
) -> None:
    """Schrijft `polygons` (x = afstand tot de referentielijn, positief
    binnenwaarts; z = hoogte) als D-Stability-model naar `path`. `linked`:
    per modelgrondsoort de D-Stability-grondsoort (zoals in soils.json) die
    hij krijgt (zie soil_library.linked_soils)."""
    model = DStabilityModel()
    # Alleen de grondsoorten van de doorsnede: niet de standaardgrondsoorten van D-Stability.
    model.datastructure.soils.Soils = []
    model.datastructure.soilvisualizations.SoilVisualizations = []
    # In de volgorde van het model (de codes in het .nc), zoals altijd.
    names = [section.soils[i] for i in sorted({p.soil for p in polygons})]
    soils, codes = layer_soils(names, linked or {})
    for soil in soils:
        if soil["parameters"] is not None:
            persistable = PersistableSoil(**soil["parameters"])
            persistable.Id = str(model._get_next_id())
            model.datastructure.soils.Soils.append(persistable)
            soil_id = persistable.Id
        else:
            soil_id = model.add_soil(Soil(name=soil["name"], code=soil["code"]))
        color = argb(section.colors.get(soil["color_of"]))
        if color:
            model.datastructure.soilvisualizations.SoilVisualizations.append(
                PersistableSoilVisualization(
                    Color=color,
                    PersistableShadingType=PersistableShadingTypeEnum.NONE,
                    SoilId=soil_id,
                )
            )
    for polygon in polygons:
        name = section.soils[polygon.soil]
        points = [Point(x=round(x, DECIMALS), z=round(z, DECIMALS)) for x, z in polygon.points]
        model.add_layer(points, codes[name], label=name)
    if label:
        model.datastructure.scenarios[0].Label = label
    model.serialize(path)
