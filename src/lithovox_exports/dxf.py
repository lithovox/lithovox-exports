"""Een dwarsdoorsnede als 2D-DXF-tekening (AutoCAD 2004, gemaakt met ezdxf;
alleen 2D POLYLINE-entiteiten, zodat CAD-programma's en de CAD-import van
GEO5 hem lezen; 2004 is de oudste DXF met exacte kleuren): de
grondpolygonen van de doorsnede (zie profile.py) als

- INTERFACES: de grenzen tussen grondsoorten en het bovenvlak, open
  polylijnen die elkaar in precies dezelfde punten raken en niet overlappen
  (GEO5: File > Import > DXF to interfaces);
- OUTLINE: de zijkanten en onderkant van het model;
- SOIL_<naam>: per grondsoort zijn vlakken, gesloten polylijnen (GEO5: DXF
  to template, om te zien welk vlak welke grondsoort is bij het toekennen),
  in de kleur van de grondsoort in het model (en de dichtstbijzijnde AutoCAD
  Color Index, voor programma's die alleen die lezen).

x is de afstand tot de referentielijn (m, positief binnenwaarts), y de
hoogte (m). Alleen geometrie: grondsoorten en hun parameters vul je in het
ontvangende programma in."""

import re
from pathlib import Path

import ezdxf
from ezdxf import colors, units
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union

from .dstability import DECIMALS
from .profile import ProfileSlice, SoilPolygon

# Coördinaten die zo dicht (m) bij een zijkant of de onderkant liggen, liggen erop.
EPS = 1e-6
DXF_VERSION = "R2004"
# DXF-kleuren (AutoCAD Color Index): grenzen rood, omtrek grijs; de laag van
# een grondsoort zijn modelkleur en de index die er het dichtst bij ligt, of
# voor een grondsoort zonder kleur om de beurt een van deze.
INTERFACES_COLOR = 1
OUTLINE_COLOR = 8
SOIL_COLORS = [3, 4, 5, 6, 2, 30, 140, 200, 50, 90]
# De kleuren van AutoCAD Color Index 0-255 (r, g, b), zoals AutoCAD ze in model space tekent.
ACI_RGB = [colors.int2rgb(value) for value in colors.DXF_DEFAULT_COLORS]
# 0 is "by block" en 7 is zwart of wit afhankelijk van de achtergrond: geen kleur voor een grondsoort.
_SOIL_INDEXES = [i for i in range(1, 256) if i != 7]


def _rgb(color: str | None) -> tuple[int, int, int] | None:
    if not color or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        return None
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def nearest_aci(color: str | None) -> int | None:
    """De AutoCAD Color Index die het dichtst bij `#rrggbb` ligt (volgens
    "redmean", een goedkope benadering van hoe ver kleuren uit elkaar lijken
    te liggen); None als het geen kleur is."""
    rgb = _rgb(color)
    if rgb is None:
        return None
    r, g, b = rgb

    def distance(i: int) -> float:
        r2, g2, b2 = ACI_RGB[i]
        mean = (r + r2) / 2
        return (2 + mean / 256) * (r - r2) ** 2 + 4 * (g - g2) ** 2 + (2 + (255 - mean) / 256) * (b - b2) ** 2

    return min(_SOIL_INDEXES, key=distance)


def layer_names(soils: list[str]) -> dict[str, str]:
    """Een laagnaam per grondsoort: SOIL_ en de naam met alleen de tekens die
    R12 toestaat (letters, cijfers, $, - en _), hooguit 31 tekens, uniek."""
    names: dict[str, str] = {}
    for soil in soils:
        base = ("SOIL_" + re.sub(r"[^A-Za-z0-9$_-]", "_", soil))[:31]
        name, n = base, 2
        while name.upper() in {v.upper() for v in names.values()}:
            suffix = f"_{n}"
            name, n = base[: 31 - len(suffix)] + suffix, n + 1
        names[soil] = name
    return names


def polylines(polygons: list[SoilPolygon]) -> list[tuple[str, list[tuple[float, float]]]]:
    """De grenzen en de omtrek: elke rand van een polygoon één keer, als
    polylijnen gesplitst waar drie of meer grondsoorten samenkomen en waar ze
    een hoek van het model omgaan; (INTERFACES of OUTLINE, punten)."""
    points = [(x, z) for p in polygons for x, z in p.points]
    u_min, u_max = min(x for x, _ in points), max(x for x, _ in points)
    z_min = min(z for _, z in points)

    def on_outline(a: tuple[float, float], b: tuple[float, float]) -> bool:
        # Op een zijkant of de onderkant; het bovenvlak telt als grens.
        return (
            (abs(a[0] - u_min) < EPS and abs(b[0] - u_min) < EPS)
            or (abs(a[0] - u_max) < EPS and abs(b[0] - u_max) < EPS)
            or (abs(a[1] - z_min) < EPS and abs(b[1] - z_min) < EPS)
        )

    network = linemerge(unary_union([LineString([*p.points, p.points[0]]) for p in polygons]))
    lines = list(network.geoms) if isinstance(network, MultiLineString) else [network]
    result = []
    for line in lines:
        coords = [(round(x, DECIMALS), round(z, DECIMALS)) for x, z in line.coords]
        run, kind = [coords[0]], None
        for a, b in zip(coords, coords[1:]):
            k = "OUTLINE" if on_outline(a, b) else "INTERFACES"
            if kind is not None and k != kind:
                result.append((kind, run))
                run = [a]
            run.append(b)
            kind = k
        result.append((kind, run))
    return result


def write_dxf(section: ProfileSlice, polygons: list[SoilPolygon], path: Path, label: str = "") -> None:
    """Schrijft `polygons` (x = afstand tot de referentielijn, positief
    binnenwaarts; z = hoogte) als DXF-tekening naar `path`. `label` wordt
    niet gebruikt (het staat in de bestandsnaam)."""
    doc = ezdxf.new(DXF_VERSION, setup=False)
    doc.units = units.M
    doc.layers.add("INTERFACES", color=INTERFACES_COLOR)
    doc.layers.add("OUTLINE", color=OUTLINE_COLOR)
    names = layer_names(sorted({section.soils[p.soil] for p in polygons}))
    for i, (soil, name) in enumerate(names.items()):
        color = section.colors.get(soil)
        layer = doc.layers.add(name, color=nearest_aci(color) or SOIL_COLORS[i % len(SOIL_COLORS)])
        # De exacte kleur; de index blijft voor programma's die alleen die lezen.
        if _rgb(color):
            layer.rgb = _rgb(color)
    msp = doc.modelspace()
    for kind, points in polylines(polygons):
        msp.add_polyline2d(points, dxfattribs={"layer": kind})
    for p in polygons:
        points = [(round(x, DECIMALS), round(z, DECIMALS)) for x, z in p.points]
        msp.add_polyline2d(points, close=True, dxfattribs={"layer": names[section.soils[p.soil]]})
    doc.saveas(path)
