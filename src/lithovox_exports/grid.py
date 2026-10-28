"""Het grid van een LithoVox-corridormodel (.nc): de referentielijn (de
lijn waarlangs gestationeerd wordt), de kolommen langs die lijn, het
coördinatenstelsel en de brokken waarin het model gelezen wordt."""

import json
from dataclasses import dataclass, field, replace

import h5py
import numpy as np
import pyproj

# Het model wordt zoveel voxels tegelijk gelezen (een reeks hele
# kolommen langs de lijn), zodat het geheugengebruik niet meegroeit met de
# lengte van de lijn.
CHUNK_VOXELS = 16_000_000
FORMAT = "lithovox-corridor-model"


def chainage_label(chainage: float) -> str:
    """Bijvoorbeeld 1234 → "km1.234"."""
    return f"km{chainage / 1000:.3f}"


def _number(value) -> float:
    """Een numeriek attribuut (netCDF slaat ze soms op als array van één element)."""
    return float(np.asarray(value).reshape(-1)[0])


@dataclass
class ModelCrs:
    """Het coördinatenstelsel van een model: horizontaal (x, y in m) en,
    als het bekend is, het verticale datum (bijvoorbeeld NAP)."""

    crs: pyproj.CRS
    # Bijvoorbeeld "EPSG:28992" of "EPSG:7415" (RD New + NAP).
    name: str
    # Bijvoorbeeld "NAP"; None als het model het niet weet.
    vertical_datum: str | None

    @property
    def height_unit(self) -> str:
        """De eenheid van de hoogte, bijvoorbeeld "m NAP" of "m"."""
        return f"m {self.vertical_datum}" if self.vertical_datum else "m"


def read_crs(f: h5py.File, metadata: dict) -> ModelCrs:
    """Het coördinatenstelsel van een geopend model: uit de metadata, of
    anders uit het attribuut `crs` of de variabele `crs` van het bestand."""
    info = metadata.get("crs") or {}
    code = info.get("code")
    candidates = [
        f"EPSG:{code}" if isinstance(code, int) else code,
        f.attrs.get("crs"),
        info.get("proj4"),
    ]
    if "crs" in f:
        attrs = f["crs"].attrs
        candidates += [attrs.get("crs_wkt"), attrs.get("proj4_params")]
    crs = None
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, bytes):
            candidate = candidate.decode()
        try:
            crs = pyproj.CRS.from_user_input(str(candidate))
        except pyproj.exceptions.CRSError:
            continue
        break
    if crs is None:
        raise ValueError("Het model heeft geen leesbaar coördinatenstelsel")
    authority = crs.to_authority()
    name = f"{authority[0]}:{authority[1]}" if authority else crs.name
    vertical = info.get("vertical")
    if not vertical and crs.is_compound:
        vertical = crs.sub_crs_list[1].name
    # Een onbekend datum ("unknown datum") is geen datum.
    if not vertical or "unknown" in str(vertical).lower():
        vertical = None
    return ModelCrs(crs=crs, name=name, vertical_datum=vertical)


@dataclass
class CorridorGrid:
    """Het grid van een corridormodel, dat de referentielijn volgt: kolommen
    op stationering s (langs de lijn) en afstand d (haaks erop, positief
    binnenwaarts), rijen op hoogte z. De kolom (i, j) staat op

        x = station_x[i] + d[j] * normal_x[i],  y = station_y[i] + d[j] * normal_y[i]

    met het station het punt op de lijn en de normaal de eenheidsvector haaks
    op de lijn, binnenwaarts. Het bestand geeft deze tabellen voor de
    middens van de kolommen en (edge_*) voor hun randen."""

    # (n, 2) de punten van de referentielijn.
    chainage_line: np.ndarray
    outwards_side: str
    # (ns,) stationering, (nd,) afstand van de kolommiddens; (nz,) hoogte van de rijmiddens.
    chainage: np.ndarray
    offset: np.ndarray
    z: np.ndarray
    ds: float
    dd: float
    dz: float
    # (ns, 2) station en normaal per kolommidden, (ns + 1, 2) per kolomrand.
    station: np.ndarray
    normal: np.ndarray
    edge_station: np.ndarray
    edge_normal: np.ndarray
    crs: ModelCrs
    # De naam van de grondsoort per code, en de kleur (CSS, bijvoorbeeld "#b9d4b4") als die bekend is.
    soils: list[str]
    colors: dict[str, str] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.chainage), len(self.offset), len(self.z)

    @property
    def length(self) -> float:
        """De lengte van de referentielijn (m)."""
        return float(np.hypot(*np.diff(self.chainage_line, axis=0).T).sum())

    def offset_edges(self) -> np.ndarray:
        """(nd + 1,) de afstanden van de randen van de kolommen."""
        return np.concatenate([self.offset - self.dd / 2, [self.offset[-1] + self.dd / 2]])

    def z_edges(self) -> np.ndarray:
        """(nz + 1,) de hoogtes van de randen van de rijen."""
        return self.z[0] - self.dz / 2 + np.arange(len(self.z) + 1) * self.dz

    def column_xy(self, start: int, stop: int) -> np.ndarray:
        """(stop - start, nd, 2) x, y van de kolommiddens van de kolommen start..stop."""
        station, normal = self.station[start:stop], self.normal[start:stop]
        return station[:, np.newaxis, :] + self.offset[np.newaxis, :, np.newaxis] * normal[:, np.newaxis, :]

    def corner_xy(self, start: int, stop: int) -> np.ndarray:
        """(stop - start + 1, nd + 1, 2) x, y van de hoekpunten van de kolommen start..stop."""
        station, normal = self.edge_station[start : stop + 1], self.edge_normal[start : stop + 1]
        d = self.offset_edges()
        return station[:, np.newaxis, :] + d[np.newaxis, :, np.newaxis] * normal[:, np.newaxis, :]

    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        """(min x, min y) en (max x, max y) van de buitenste hoekpunten van het model."""
        corners = self.corner_xy(0, self.shape[0])[:, [0, -1]].reshape(-1, 2)
        return corners.min(axis=0), corners.max(axis=0)

    def chunks(self, voxels: int | None = None) -> list[tuple[int, int]]:
        """De reeksen kolommen (start, stop) waarin het model gelezen wordt, elk ongeveer `voxels` (CHUNK_VOXELS) groot."""
        voxels = voxels or CHUNK_VOXELS
        ns, nd, nz = self.shape
        size = max(1, voxels // max(1, nd * nz))
        return [(start, min(start + size, ns)) for start in range(0, ns, size)]

    def coarsened(self, fd: int, fz: int) -> tuple["CorridorGrid", int]:
        """Het grid met voxels die `fd` keer breder en `fz` keer hoger zijn (even
        lang), de rijen op veelvouden van de nieuwe hoogte; en hoeveel lege
        fijne rijen daarvoor onder het fijne grid komen (zie coarsen)."""
        dz = self.dz * fz
        bottom = float(self.z[0]) - self.dz / 2
        coarse_bottom = np.floor(bottom / dz + 1e-6) * dz
        pad = int(round((bottom - coarse_bottom) / self.dz))
        nz = -(-(len(self.z) + pad) // fz)
        nd = -(-len(self.offset) // fd)
        dd = self.dd * fd
        left = float(self.offset[0]) - self.dd / 2
        return (
            replace(
                self,
                offset=left + (np.arange(nd) + 0.5) * dd,
                z=coarse_bottom + (np.arange(nz) + 0.5) * dz,
                dd=dd,
                dz=dz,
            ),
            pad,
        )


def read_metadata(f: h5py.File) -> dict:
    """De metadata (JSON) van een geopend model; ValueError als het geen LithoVox-model is."""
    if f.attrs.get("lithovox_format") != FORMAT:
        raise ValueError("Dit bestand is geen LithoVox-corridormodel")
    return json.loads(f.attrs.get("lithovox_metadata", "{}"))


def read_grid(f: h5py.File) -> CorridorGrid:
    """Het grid van een geopend model (.nc)."""
    metadata = read_metadata(f)
    soil = f["soil"]

    def xy(name: str) -> np.ndarray:
        return np.stack([np.asarray(f[f"{name}_x"][()], dtype=np.float64), np.asarray(f[f"{name}_y"][()], dtype=np.float64)], axis=1)

    return CorridorGrid(
        chainage_line=np.asarray(json.loads(f.attrs["chainage_line"]), dtype=np.float64).reshape(-1, 2),
        outwards_side=str(f.attrs["outwards_side"]),
        chainage=np.asarray(f["s"][()], dtype=np.float64),
        offset=np.asarray(f["d"][()], dtype=np.float64),
        z=np.asarray(f["z"][()], dtype=np.float64),
        ds=_number(f.attrs["ds"]),
        dd=_number(f.attrs["dd"]),
        dz=_number(f.attrs["dz"]),
        station=xy("station"),
        normal=xy("normal"),
        edge_station=xy("edge_station"),
        edge_normal=xy("edge_normal"),
        crs=read_crs(f, metadata),
        soils=list(json.loads(soil.attrs["soil_names"])),
        colors=dict(json.loads(soil.attrs.get("soil_colors", "{}"))),
    )


@dataclass
class CorridorChunk:
    """De voxels van de kolommen start..stop van een corridormodel, en van de
    kolommen er net voor en na (lo..hi): het mesh moet die zien om te weten
    of een vlak aan een andere grondsoort grenst."""

    start: int
    stop: int
    lo: int
    hi: int
    # (hi - lo, nd, nz) code van de grondsoort, -1 = geen voxel.
    matrix: np.ndarray

    @property
    def core(self) -> slice:
        """De eigen kolommen van het brok binnen matrix."""
        return slice(self.start - self.lo, self.stop - self.lo)


def coarsen(chunk: CorridorChunk, fd: int, fz: int, pad: int, soil_count: int) -> CorridorChunk:
    """Het brok op een grid met voxels die `fd` keer breder en `fz` keer hoger
    zijn (CorridorGrid.coarsened geeft `pad`): elke grove voxel krijgt de
    grondsoort die de meeste van zijn fijne voxels hebben, of geen als de
    meeste leeg zijn (bij gelijkspel wint een grondsoort)."""
    if fd == 1 and fz == 1 and pad == 0:
        return chunk
    n, nd, nz = chunk.matrix.shape
    ndc, nzc = -(-nd // fd), -(-(nz + pad) // fz)
    fine = np.full((n, ndc * fd, nzc * fz), -1, dtype=np.int8)
    fine[:, :nd, pad : pad + nz] = chunk.matrix
    # Tel elke code (leeg als laatste, zodat een gelijkspel naar een grondsoort gaat) per grove voxel.
    cells = fine.reshape(n, ndc, fd, nzc, fz).transpose(0, 1, 3, 2, 4).reshape(n * ndc * nzc, fd * fz)
    codes = np.where(cells < 0, soil_count, cells).astype(np.int64)
    counts = np.bincount(
        (np.arange(len(cells))[:, np.newaxis] * (soil_count + 1) + codes).ravel(),
        minlength=len(cells) * (soil_count + 1),
    )
    winner = counts.reshape(len(cells), soil_count + 1).argmax(axis=1)
    matrix = np.where(winner == soil_count, -1, winner).astype(np.int8).reshape(n, ndc, nzc)
    return CorridorChunk(start=chunk.start, stop=chunk.stop, lo=chunk.lo, hi=chunk.hi, matrix=matrix)
