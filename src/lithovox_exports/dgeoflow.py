"""Een dwarsdoorsnede als D-Geo Flow-model (.flox), gemaakt met GEOLib,
net als de D-Stability-export (dstability.py): de grondpolygonen van de
doorsnede (zie profile.py) als lagen, elk met zijn grondsoort en kleur. Een
gekoppelde grondsoort (zie soil_library.py) krijgt de grondsoort uit het
`.flox` van de gebruiker met zijn doorlatendheden; de andere hebben alleen
een naam (de standaarddoorlatendheden van D-Geo Flow). Randvoorwaarden
(waterstanden), mesh en rekeninstellingen laten we aan de gebruiker."""

from pathlib import Path

from geolib.geometry.one import Point
from geolib.models.dgeoflow import DGeoFlowModel
from geolib.models.dgeoflow.internal import (
    PersistableShadingTypeEnum,
    PersistableSoil,
    PersistableSoilVisualization,
)
from geolib.soils.soil import Soil

from .dstability import DECIMALS, argb
from .profile import ProfileSlice, SoilPolygon
from .soil_library import layer_soils


def write_flox(
    section: ProfileSlice,
    polygons: list[SoilPolygon],
    path: Path,
    label: str = "",
    linked: dict[str, dict] | None = None,
) -> None:
    """Schrijft `polygons` (x = afstand tot de referentielijn, positief
    binnenwaarts; z = hoogte) als D-Geo Flow-model naar `path`. `linked`:
    per modelgrondsoort de D-Geo Flow-grondsoort (zoals in soils.json) die
    hij krijgt (zie soil_library.linked_soils)."""
    model = DGeoFlowModel()
    # Alleen de grondsoorten van de doorsnede: niet de standaardgrondsoorten van D-Geo Flow.
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
