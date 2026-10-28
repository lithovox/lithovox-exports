"""Een corridormodel als OGC 3D Tiles 1.1 (bijvoorbeeld voor Cesium, ArcGIS
of QGIS): een tileset.json en een GLB-tegel per TILE_LENGTH m van de lijn,
op de wereldbol geplaatst (ECEF)."""

import json
from pathlib import Path

import numpy as np
import pyproj
from pyproj import Transformer

from .glb import CorridorGlbBuilder
from .grid import CorridorChunk, CorridorGrid, ModelCrs

# Lengte van een tegel (m langs de referentielijn).
TILE_LENGTH = 250.0
# WGS 84 geocentrisch (ECEF), waarin 3D Tiles zijn coördinaten verwacht.
ECEF = "EPSG:4978"


class EcefTransform:
    """Modelcoördinaten (x, y, hoogte) naar ECEF, voor 3D Tiles.

    Heeft het model een samengesteld stelsel met een verticaal datum
    (bijvoorbeeld EPSG:7415, RD New + NAP), dan rekent PROJ de hoogte om,
    zo nodig met correctiegrids van het PROJ-netwerk (`accurate`). Anders
    wordt de hoogte als ellipsoïdische hoogte genomen: de ligging klopt dan
    horizontaal, maar kan verticaal tientallen meters afwijken (de
    geoïdehoogte) en `accurate` is False."""

    def __init__(self, crs: ModelCrs):
        self.accurate = False
        self.transformer = None
        if crs.crs.is_compound:
            # De correctiegrids voor de hoogte komen van het PROJ-netwerk (cdn.proj.org).
            pyproj.network.set_network_enabled(True)
            try:
                self.transformer = Transformer.from_crs(crs.crs, ECEF, always_xy=True, only_best=True)
                self.accurate = True
            except pyproj.exceptions.ProjError:
                self.transformer = None
        if self.transformer is None:
            self.transformer = Transformer.from_crs(crs.crs, ECEF, always_xy=True)
        how = "" if self.accurate else " (hoogte als ellipsoïdische hoogte)"
        self.description = f"{crs.name} naar {ECEF}{how}"

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """(n, 3) x, y, hoogte naar (n, 3) ECEF X, Y, Z."""
        x, y, z = self.transformer.transform(points[:, 0], points[:, 1], points[:, 2])
        return np.stack([x, y, z], axis=1)


def _box(low: np.ndarray, high: np.ndarray, centre: np.ndarray) -> dict:
    """Een 3D Tiles-box om low..high (ECEF), ten opzichte van `centre`."""
    half = (high - low) / 2
    middle = (low + high) / 2 - centre
    return {"box": [*middle.tolist(), half[0], 0, 0, 0, half[1], 0, 0, 0, half[2]]}


class TilesetWriter:
    """Schrijft een corridormodel als OGC 3D Tiles 1.1 in `directory`: voeg de
    brokken op volgorde toe en roep daarna finish aan."""

    def __init__(
        self,
        directory: Path,
        grid: CorridorGrid,
        transform: EcefTransform,
        soil_boundaries: bool = False,
        voxel_edges: bool = True,
    ):
        self.directory = directory
        (directory / "tiles").mkdir(parents=True, exist_ok=True)
        self.grid = grid
        self.transform = transform
        self.soil_boundaries = soil_boundaries
        self.voxel_edges = voxel_edges
        self.builder = CorridorGlbBuilder(grid, soil_boundaries, voxel_edges)
        self.tile_start = 0
        self.children: list[dict] = []
        self.low = np.full(3, np.inf)
        self.high = np.full(3, -np.inf)

    def add(self, chunk: CorridorChunk) -> None:
        self.builder.add(chunk)
        if (chunk.stop - self.tile_start) * self.grid.ds >= TILE_LENGTH:
            self._flush(chunk.stop)

    def _flush(self, stop: int) -> None:
        builder = self.builder
        self.builder = CorridorGlbBuilder(self.grid, self.soil_boundaries, self.voxel_edges)
        start, self.tile_start = self.tile_start, stop
        if builder.empty:
            return
        ecef = self.transform(builder.world_points())
        low, high = ecef.min(axis=0), ecef.max(axis=0)
        centre = (low + high) / 2
        name = f"tiles/tile_{len(self.children):04d}.glb"
        (self.directory / name).write_bytes(builder.ecef_glb(self.transform, centre))
        self.low, self.high = np.minimum(self.low, low), np.maximum(self.high, high)
        s0 = float(self.grid.chainage[start] - self.grid.ds / 2)
        self.children.append(
            {
                # Kolomsgewijze 4x4: een verschuiving naar het midden van de tegel.
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, *centre.tolist(), 1],
                "boundingVolume": _box(low, high, centre),
                "geometricError": 0,
                "content": {"uri": name},
                "extras": {"chainage": [s0, s0 + (stop - start) * self.grid.ds]},
            }
        )

    def finish(self) -> Path:
        """Schrijft de laatste tegel en tileset.json; geeft het pad van tileset.json."""
        self._flush(self.grid.shape[0])
        if not self.children:
            self.low = self.high = np.zeros(3)
        size = float(np.linalg.norm(self.high - self.low))
        tileset = {
            "asset": {
                "version": "1.1",
                "generator": "lithovox-exports",
                "extras": {
                    "crs": self.grid.crs.name,
                    "verticalDatum": self.grid.crs.vertical_datum,
                    "transformation": self.transform.description,
                    "accurate": self.transform.accurate,
                },
            },
            "geometricError": size,
            "root": {
                "boundingVolume": _box(self.low, self.high, np.zeros(3)),
                "geometricError": TILE_LENGTH,
                "refine": "ADD",
                "children": self.children,
            },
        }
        path = self.directory / "tileset.json"
        path.write_text(json.dumps(tileset, indent=1), encoding="utf-8")
        return path
