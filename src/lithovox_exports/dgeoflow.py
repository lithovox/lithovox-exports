"""Een dwarsdoorsnede als D-Geo Flow-model (.flox), gemaakt met d-geolib,
net als de D-Stability-export (dstability.py): de grondpolygonen van de
doorsnede (zie profile.py) als lagen, elk met zijn grondsoort en kleur. De
grondsoorten hebben alleen een naam (de standaarddoorlatendheden van D-Geo
Flow); randvoorwaarden (waterstanden), mesh en rekeninstellingen laten we
aan de gebruiker."""

from pathlib import Path

from geolib.geometry.one import Point
from geolib.models.dgeoflow import DGeoFlowModel
from geolib.models.dgeoflow.internal import PersistableShadingTypeEnum, PersistableSoilVisualization
from geolib.soils.soil import Soil

from .dstability import DECIMALS, argb
from .profile import ProfileSlice, SoilPolygon


def write_flox(section: ProfileSlice, polygons: list[SoilPolygon], path: Path, label: str = "") -> None:
    """Schrijft `polygons` (x = afstand tot de referentielijn, positief
    binnenwaarts; z = hoogte) als D-Geo Flow-model naar `path`."""
    model = DGeoFlowModel()
    # Alleen de grondsoorten van de doorsnede: niet de standaardgrondsoorten van D-Geo Flow.
    model.datastructure.soils.Soils = []
    for soil in sorted({p.soil for p in polygons}):
        name = section.soils[soil]
        soil_id = model.add_soil(Soil(name=name, code=name))
        color = argb(section.colors.get(name))
        if color:
            model.datastructure.soilvisualizations.SoilVisualizations.append(
                PersistableSoilVisualization(Color=color, PersistableShadingType=PersistableShadingTypeEnum.NONE, SoilId=soil_id)
            )
    for polygon in polygons:
        points = [Point(x=round(x, DECIMALS), z=round(z, DECIMALS)) for x, z in polygon.points]
        model.add_layer(points, section.soils[polygon.soil], label=section.soils[polygon.soil])
    if label:
        model.datastructure.scenarios[0].Label = label
    model.serialize(path)
