"""Het hele model in andere formaten, voor weergave en GIS: GLB, OGC 3D Tiles
en een LAZ-puntenwolk. Het bestand wordt een brok tegelijk gelezen, dus de
grootte van het model maakt niet uit; alle drie gebruiken dezelfde grovere
weergavevoxels (elk krijgt de grondsoort die de meeste van zijn modelvoxels
hebben)."""

import shutil
import zipfile
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .glb import CorridorGlbBuilder, model_origin
from .grid import CorridorChunk, CorridorGrid, coarsen, read_grid
from .laz import PointCloudWriter
from .tiles import EcefTransform, TilesetWriter


def display_chunks(f: h5py.File, voxel_width: float, voxel_height: float) -> tuple[CorridorGrid, Iterator[CorridorChunk]]:
    """Het weergavegrid van een geopend model, met voxels van ongeveer
    `voxel_width` x `voxel_height` m dwars en omhoog (hele veelvouden van die
    van het model, even lang langs de lijn), en zijn brokken op volgorde,
    een voor een uit het bestand gelezen."""
    grid = read_grid(f)
    fd = max(1, int(round(voxel_width / grid.dd)))
    fz = max(1, int(round(voxel_height / grid.dz)))
    display, pad = grid.coarsened(fd, fz)
    soil = f["soil"]
    ns = grid.shape[0]

    def chunks() -> Iterator[CorridorChunk]:
        for start, stop in grid.chunks():
            # Met de kolom aan weerszijden, zodat de vlakken aan de randen van een brok kloppen.
            lo, hi = max(0, start - 1), min(ns, stop + 1)
            matrix = np.asarray(soil[lo:hi], dtype=np.int8)
            chunk = CorridorChunk(start=start, stop=stop, lo=lo, hi=hi, matrix=matrix)
            yield coarsen(chunk, fd, fz, pad, len(grid.soils))

    return display, chunks()


def _extras(f: h5py.File, display: CorridorGrid) -> dict:
    return {
        "name": str(f.attrs.get("name", "")),
        "project": str(f.attrs.get("project", "")),
        "voxel": {"length": display.ds, "width": display.dd, "height": display.dz},
    }


def model_to_glb(
    file: str | Path,
    voxel_width: float = 1.0,
    voxel_height: float = 0.2,
    soil_boundaries: bool = False,
    voxel_edges: bool = False,
) -> bytes:
    """Het model in `file` als GLB: één mesh per grondsoort, alleen de
    buitenkant (met `soil_boundaries` ook de vlakken tussen grondsoorten; met
    `voxel_edges` met normalen en de randen van elke voxel donkerder, ongeveer
    vier keer zo groot); posities ten opzichte van een oorsprong in de extras
    van de GLB (zie glb.py)."""
    with h5py.File(file, "r") as f:
        display, chunks = display_chunks(f, voxel_width, voxel_height)
        builder = CorridorGlbBuilder(display, soil_boundaries, voxel_edges)
        for chunk in chunks:
            builder.add(chunk)
        extras = _extras(f, display)
    return builder.glb(model_origin(display), extras)


def model_to_tiles(
    file: str | Path,
    directory: Path,
    voxel_width: float = 1.0,
    voxel_height: float = 0.2,
    soil_boundaries: bool = False,
    voxel_edges: bool = True,
) -> Path:
    """Het model in `file` als OGC 3D Tiles 1.1, gezipt: 3d/tileset.json en een
    GLB per 250 m van de lijn (3d/tiles/), op de wereldbol geplaatst,
    standaard met normalen en de randen van elke voxel donkerder (zie
    tiles.py). Geschreven in `directory`; geeft het pad van de zip."""
    folder = directory / "3d"
    with h5py.File(file, "r") as f:
        display, chunks = display_chunks(f, voxel_width, voxel_height)
        writer = TilesetWriter(folder, display, EcefTransform(display.crs), soil_boundaries, voxel_edges)
        for chunk in chunks:
            writer.add(chunk)
        writer.finish()
    zip_path = directory / "tiles.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    shutil.rmtree(folder, ignore_errors=True)
    return zip_path


def model_to_laz(file: str | Path, path: Path, voxel_width: float = 1.0, voxel_height: float = 0.2) -> int:
    """Het model in `file` als LAZ-puntenwolk op `path` (zie laz.py): een punt
    per weergavevoxel. Geeft het aantal punten."""
    with h5py.File(file, "r") as f:
        display, chunks = display_chunks(f, voxel_width, voxel_height)
        writer = PointCloudWriter(path, display)
        try:
            for chunk in chunks:
                writer.add(chunk)
        finally:
            writer.close()
    return writer.count
