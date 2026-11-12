"""De parameterzones van een LithoVox-model (.nc).

In LithoVox zet je bij een sondering of boring eigen waarden voor een
grondsoort (nu: de volumieke gewichten): ze gelden in een cirkel met een
straal rond dat object, de parameterzone. Het model bewaart ze in de
metadata (`lithovox_metadata`, sleutel `localSoilParameters`, versie 1):

    {"version": 1, "comment": "…", "units": {"gammaDry": "kN/m3", …},
     "points": [{"kind": "cpt", "object": "CPT000000074503",
                 "x": 120594.0, "y": 474114.0, "radius": 50.0,
                 "soil": "Peat", "soilClass": "Peat",
                 "gammaDry": 10.0, "gammaSat": 10.0, …}]}

`soilClass` is de grondsoort van het model waarvoor de waarden gelden (de
kriging-klasse van de code `soil` van het object), `x`, `y` het midden in
het coördinatenstelsel van het model. Geen voxel draagt de waarden en de
zones zijn niet ondertekend: ze komen apart naast de grondsoorten van het
model. `Model.parameter_areas` geeft ze; een dwarsdoorsnede gebruikt ze met
`use_parameter_area_values=True`."""

import math
from dataclasses import dataclass

import numpy as np

KEY = "localSoilParameters"
VERSION = 1


@dataclass(frozen=True)
class ParameterArea:
    """Eén parameterzone: een cirkel rond een sondering of boring waarin de
    grondsoort `soil_class` van het model de gewichten van de zone krijgt."""

    # Bijvoorbeeld "cpt" of "borehole", en de naam van het object.
    kind: str
    object: str
    # Het midden (m, coördinatenstelsel van het model) en de straal (m).
    x: float
    y: float
    radius: float
    # De grondsoort van het model waarvoor de zone geldt, en de code van het object.
    soil_class: str
    soil: str
    # Volumiek gewicht boven en onder de freatische lijn (kN/m3).
    gamma_dry: float
    gamma_sat: float

    @property
    def name(self) -> str:
        """Bijvoorbeeld "Peat (CPT000000074503)": de naam van de grondsoort in de zone."""
        return f"{self.soil_class} ({self.object})"

    def contains(self, x, y):
        """Of het punt (of de punten, arrays) x, y in de zone ligt (de rand erbij)."""
        return np.hypot(np.asarray(x) - self.x, np.asarray(y) - self.y) <= self.radius

    def distance(self, x, y):
        """De afstand (m) van het punt (of de punten) x, y tot het midden."""
        return np.hypot(np.asarray(x) - self.x, np.asarray(y) - self.y)

    def crosses(self, start: tuple[float, float], end: tuple[float, float]) -> bool:
        """Of het lijnstuk van `start` naar `end` (x, y) de zone raakt."""
        (x0, y0), (x1, y1) = start, end
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else min(1.0, max(0.0, ((self.x - x0) * dx + (self.y - y0) * dy) / length2))
        return bool(self.distance(x0 + t * dx, y0 + t * dy) <= self.radius)


def _number(point: dict, key: str, positive: bool = False) -> float:
    value = point.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"De parameterzone {point.get('object')!r} heeft geen getal bij {key!r}")
    if positive and value <= 0:
        raise ValueError(f"De parameterzone {point.get('object')!r} heeft {key!r} {value:g}: verwacht meer dan 0")
    return float(value)


def read_parameter_areas(metadata: dict, soils: list[str]) -> list[ParameterArea]:
    """De parameterzones uit de metadata van een model; leeg als het model er
    geen heeft (ook de oudere modellen). `soils`: de grondsoorten van het
    model. ValueError als ze niet te lezen zijn of een zone een grondsoort
    noemt die het model niet heeft."""
    data = metadata.get(KEY)
    if data is None:
        return []
    if not isinstance(data, dict):
        raise ValueError("De parameterzones van het model zijn niet te lezen")
    if data.get("version") != VERSION:
        raise ValueError(
            f"De parameterzones van het model hebben versie {data.get('version')}; dit pakket leest versie {VERSION}"
        )
    points = data.get("points")
    if not isinstance(points, list):
        raise ValueError("De parameterzones van het model hebben geen 'points'")
    areas = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Een parameterzone van het model is geen object")
        for key in ("object", "soilClass"):
            if not isinstance(point.get(key), str) or not point[key]:
                raise ValueError(f"Een parameterzone van het model heeft geen tekst bij {key!r}")
        if point["soilClass"] not in soils:
            raise ValueError(
                f"De parameterzone {point['object']!r} geldt voor {point['soilClass']!r}, "
                f"maar dat is geen grondsoort van het model (die zijn: {', '.join(soils)})"
            )
        areas.append(
            ParameterArea(
                kind=str(point.get("kind") or ""),
                object=point["object"],
                x=_number(point, "x"),
                y=_number(point, "y"),
                radius=_number(point, "radius", positive=True),
                soil_class=point["soilClass"],
                soil=str(point.get("soil") or point["soilClass"]),
                gamma_dry=_number(point, "gammaDry", positive=True),
                gamma_sat=_number(point, "gammaSat", positive=True),
            )
        )
    return areas
