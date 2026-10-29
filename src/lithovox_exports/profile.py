"""Een dwarsdoorsnede van een corridormodel (.nc) als grondpolygonen, voor
rekensoftware zoals D-Stability."""

import json
from dataclasses import dataclass

import h5py
import numpy as np
from scipy import ndimage
from shapely import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union

from .grid import _number

# Gebieden van één grondsoort met minder voxels dan dit zijn ruis van de
# kriging: ze krijgen de grondsoort die de meeste van hun buren hebben.
MIN_REGION_VOXELS = 6
# Hoe ver (in voxels) een gladgemaakte grens van de trap van de voxels mag liggen.
SMOOTHING = 1.5
# Lenzen losknippen: hooguit zoveel rondes.
MAX_CUT_ROUNDS = 8


@dataclass
class ProfileSlice:
    """De voxels van een corridormodel in de dwarsdoorsnede bij één kolom
    langs de lijn: kolommen op afstand u van de referentielijn (positief
    binnenwaarts, van A naar B), rijen op hoogte z."""

    index: int
    chainage: float
    # x, y van het punt op de referentielijn, en de eenheidsvector haaks erop (binnenwaarts).
    station: tuple[float, float]
    normal: tuple[float, float]
    u: np.ndarray  # (nd,) kolommiddens
    z: np.ndarray  # (nz,) rijmiddens
    du: float
    dz: float
    labels: np.ndarray  # (nd, nz) int8 code van de grondsoort, -1 = geen voxel
    ground: np.ndarray  # (nd,) maaiveld (m), NaN waar onbekend
    soils: list[str]
    colors: dict[str, str]

    def to_world(self, u: float) -> tuple[float, float]:
        """x, y (in het coördinatenstelsel van het model) van het punt `u` meter van de referentielijn."""
        return (
            self.station[0] + u * self.normal[0],
            self.station[1] + u * self.normal[1],
        )


def _fill_left_out(f: h5py.File, section: ProfileSlice) -> None:
    """Vult de kolommen van `section` die het model weggelaten heeft (aan de
    binnenkant van een bocht, waar ze dichter bij de lijn op een andere
    metrering liggen) met de modelkolom op hun plek: van de kolommen
    waarvan de lijn haaks op de referentielijn binnen een halve voxellengte
    langs de plek gaat, de dichtstbijzijnde die niet weggelaten is."""
    source = np.asarray(f["source"][section.index], dtype=np.int32)
    empty = np.nonzero(source < 0)[0]
    if len(empty) == 0:
        return
    ds = _number(f.attrs["ds"])
    px = section.station[0] + section.u[empty] * section.normal[0]
    py = section.station[1] + section.u[empty] * section.normal[1]
    # Een kolom reikt alleen tot plekken binnen de breedte van het model rond zijn station.
    all_x = np.asarray(f["station_x"][()], dtype=np.float64)
    all_y = np.asarray(f["station_y"][()], dtype=np.float64)
    reach = np.abs(section.u).max() + section.du + ds
    near = np.hypot(all_x[:, None] - px[None, :], all_y[:, None] - py[None, :]).min(axis=1) <= reach
    candidates = np.nonzero(near)[0]
    if len(candidates) == 0:
        return
    lo, hi = int(candidates[0]), int(candidates[-1]) + 1
    sx, sy = all_x[lo:hi], all_y[lo:hi]
    nx = np.asarray(f["normal_x"][lo:hi])
    ny = np.asarray(f["normal_y"][lo:hi])
    window_source = np.asarray(f["source"][lo:hi], dtype=np.int32)
    # Per plek (rijen) en kandidaatkolom (kolommen): afstand tot de lijn en afstand erlangs.
    dx = px[:, None] - sx[None, :]
    dy = py[:, None] - sy[None, :]
    offset = dx * nx[None, :] + dy * ny[None, :]
    along = np.abs(dx * ny[None, :] - dy * nx[None, :])
    j = np.rint((offset - section.u[0]) / section.du).astype(np.int64)
    inside = (j >= 0) & (j < len(section.u))
    usable = inside & (along <= ds / 2 + 1e-9)
    usable[inside] &= window_source[np.nonzero(inside)[1], j[inside]] >= 0
    along = np.where(usable, along, np.inf)
    best = np.argmin(along, axis=1)
    found = np.isfinite(along[np.arange(len(empty)), best])
    if not found.any():
        return
    soil, ground = f["soil"], f["ground"]
    for k, column in zip(np.nonzero(found)[0], empty[found]):
        ii, jj = lo + best[k], j[k, best[k]]
        section.labels[column] = soil[ii, jj]
        section.ground[column] = ground[ii, jj]


def read_slice(f: h5py.File, chainage: float) -> ProfileSlice:
    """De dwarsdoorsnede van een geopend model bij de kolom die het dichtst bij
    `chainage` ligt; de kolommen die het model bij een bocht weggelaten heeft
    gevuld met het model op hun plek (zodat de doorsnede over de hele breedte
    van het model loopt, zoals de doorsnede waarvan hij gemaakt is)."""
    s = np.asarray(f["s"][()], dtype=np.float64)
    i = int(np.clip(np.argmin(np.abs(s - chainage)), 0, len(s) - 1))
    soil = f["soil"]
    section = ProfileSlice(
        index=i,
        chainage=float(s[i]),
        station=(float(f["station_x"][i]), float(f["station_y"][i])),
        normal=(float(f["normal_x"][i]), float(f["normal_y"][i])),
        u=np.asarray(f["d"][()], dtype=np.float64),
        z=np.asarray(f["z"][()], dtype=np.float64),
        du=_number(f.attrs["dd"]),
        dz=_number(f.attrs["dz"]),
        labels=np.asarray(soil[i], dtype=np.int8),
        ground=(np.asarray(f["ground"][i], dtype=np.float64) if "ground" in f else np.full(soil.shape[1], np.nan)),
        soils=list(json.loads(soil.attrs["soil_names"])),
        colors=dict(json.loads(soil.attrs.get("soil_colors", "{}"))),
    )
    if "source" in f:
        _fill_left_out(f, section)
    return section


def remove_specks(labels: np.ndarray, min_voxels: int = MIN_REGION_VOXELS) -> np.ndarray:
    """`labels` waarin elk gebied (4-samenhangend, van één grondsoort of van
    geen voxel) kleiner dan `min_voxels` de grondsoort krijgt die de meeste
    van zijn buren hebben."""
    result = labels.copy()
    for _ in range(3):
        changed = False
        for value in np.unique(result):
            regions, count = ndimage.label(result == value)
            if count == 0:
                continue
            sizes = np.bincount(regions.ravel())
            small = np.nonzero(sizes < min_voxels)[0]
            small = small[small != 0]
            if len(small) == 0:
                continue
            # Alleen het stuk rond elk klein gebied bekijken: de doorsnede kan groot zijn.
            boxes = ndimage.find_objects(regions)
            for region in small:
                box = boxes[region - 1]
                box = tuple(slice(max(0, b.start - 1), b.stop + 1) for b in box)
                mask = regions[box] == region
                around = ndimage.binary_dilation(mask) & ~mask
                neighbours = result[box][around]
                neighbours = neighbours[neighbours != value]
                if len(neighbours) == 0:
                    continue
                values, counts = np.unique(neighbours, return_counts=True)
                result[box][mask] = values[np.argmax(counts)]
                changed = True
        if not changed:
            break
    return result


def boundary_lines(labels: np.ndarray, u_edges: np.ndarray, z_edges: np.ndarray) -> list[LineString]:
    """De grenzen tussen verschillende grondsoorten (en tussen grond en geen
    voxel: de omtrek van het model) als lijnen langs de randen van de voxels,
    samengevoegd tussen de punten waar drie of meer grondsoorten samenkomen:
    elke lijn scheidt steeds dezelfde twee."""
    padded = np.pad(labels, 1, constant_values=-1)
    segments = []
    # Verticale randen: tussen kolom j - 1 en j (in padded), op u_edges[j].
    vertical = padded[:-1, 1:-1] != padded[1:, 1:-1]
    j, k = np.nonzero(vertical)
    segments += [((u_edges[a], z_edges[b]), (u_edges[a], z_edges[b + 1])) for a, b in zip(j, k)]
    # Horizontale randen: tussen rij k - 1 en k (in padded), op z_edges[k].
    horizontal = padded[1:-1, :-1] != padded[1:-1, 1:]
    j, k = np.nonzero(horizontal)
    segments += [((u_edges[a], z_edges[b]), (u_edges[a + 1], z_edges[b])) for a, b in zip(j, k)]
    if not segments:
        return []
    merged = linemerge(MultiLineString(segments))
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


def _split_at_sides(line: LineString, sides: tuple[float, float]) -> list[LineString]:
    """`line` gesplitst waar hij bij de linker- of rechterkant van het model
    (x in `sides`) afbuigt, zodat het gladmaken de hoeken en rechte zijkanten
    van het model houdt."""
    coords = list(line.coords)
    on_side = [any(abs(x - side) < 1e-9 for side in sides) for x, _ in coords]
    pieces, start = [], 0
    for i in range(1, len(coords) - 1):
        if on_side[i] and not (on_side[i - 1] and on_side[i + 1]):
            pieces.append(LineString(coords[start : i + 1]))
            start = i
    pieces.append(LineString(coords[start:]))
    return pieces


def _midpoints(line: LineString) -> LineString:
    """`line` door de middens van zijn voxelranden, met de eindpunten: een trap
    wordt de lijn halverwege zijn binnen- en buitenhoeken, zodat vereenvoudigen
    de grens niet een halve trede verschuift."""
    coords = np.asarray(line.coords)
    if len(coords) < 3:
        return line
    middles = (coords[:-1] + coords[1:]) / 2
    if line.is_closed:
        return LineString(np.r_[middles, middles[:1]])
    return LineString(np.r_[coords[:1], middles, coords[-1:]])


def _label_at(section: ProfileSlice, labels: np.ndarray, point: Point) -> int:
    j = int(np.clip(np.floor((point.x - (section.u[0] - section.du / 2)) / section.du), 0, len(section.u) - 1))
    k = int(np.clip(np.floor((point.y - (section.z[0] - section.dz / 2)) / section.dz), 0, len(section.z) - 1))
    return int(labels[j, k])


def _cut(face: Polygon) -> list[LineString]:
    """Lijnen die de gaten van `face` losknippen: een verticale lijn door het
    midden van elk gat, van het gat omhoog en omlaag tot de omtrek van het
    vlak (of het volgende gat), die het vlak in tweeën deelt."""
    cuts = []
    _, bottom, _, top = face.bounds
    for ring in face.interiors:
        x0, _, x1, _ = ring.bounds
        x = (x0 + x1) / 2
        part = LineString([(x, bottom - 1), (x, top + 1)]).intersection(face)
        pieces = list(part.geoms) if hasattr(part, "geoms") else [part]
        cuts += [p for p in pieces if isinstance(p, LineString) and p.length > 0 and p.distance(ring) < 1e-9]
    return cuts


@dataclass
class SoilPolygon:
    """Een vlak van één grondsoort in een dwarsdoorsnede."""

    # Index in de grondsoorten van het model (Model.soils).
    soil: int
    # (x, z): tegen de klok in, niet gesloten (het laatste punt is niet gelijk aan het eerste).
    points: list[tuple[float, float]]


def section_polygons(
    section: ProfileSlice,
    smoothing: float = SMOOTHING,
    min_voxels: int = MIN_REGION_VOXELS,
) -> list[SoilPolygon]:
    """De grondpolygonen van een doorsnede: ze delen hun punten precies met
    hun buren en hebben geen gaten (zoals D-Stability vereist)."""
    labels = remove_specks(section.labels, min_voxels)
    u_edges = section.u[0] - section.du / 2 + np.arange(len(section.u) + 1) * section.du
    z_edges = section.z[0] - section.dz / 2 + np.arange(len(section.z) + 1) * section.dz
    lines = boundary_lines(labels, u_edges, z_edges)
    if smoothing > 0:
        tolerance = smoothing * min(section.du, section.dz)
        sides = (u_edges[0], u_edges[-1])
        lines = [
            _midpoints(piece).simplify(tolerance, preserve_topology=False)
            for line in lines
            for piece in _split_at_sides(line, sides)
        ]
    lines = [line for line in lines if line.length > 0]
    if not lines:
        return []

    cuts: list[LineString] = []
    faces: list[tuple[int, Polygon]] = []
    for _ in range(MAX_CUT_ROUNDS):
        faces = []
        holes = False
        for face in polygonize(unary_union(lines + cuts)):
            if face.area <= 0:
                continue
            soil = _label_at(section, labels, face.representative_point())
            if soil < 0:
                continue
            faces.append((soil, face))
            if face.interiors:
                holes = True
                cuts += _cut(face)
        if not holes:
            break
    result = []
    for soil, face in faces:
        exterior = face.exterior
        if not exterior.is_ccw:
            exterior = LineString(list(exterior.coords)[::-1])
        coords = [(float(x), float(y)) for x, y in list(exterior.coords)[:-1]]
        result.append(SoilPolygon(soil=soil, points=coords))
    return result
