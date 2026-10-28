"""De publieke interface: `Model` (een LithoVox-corridormodel, .nc), met
`CrossSection` (een dwarsdoorsnede) en `SoilColumn` (een grondkolom op één
plek). Elke export is een `to_*`-methode die het geschreven pad teruggeeft."""

import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from .convert import model_to_glb, model_to_laz, model_to_tiles
from .dgeoflow import write_flox
from .dstability import read_stix_soils, write_stix
from .dxf import write_dxf
from .grid import ModelCrs, chainage_label, read_grid, read_metadata
from .profile import SMOOTHING, ProfileSlice, SoilPolygon, read_slice, section_polygons
from .xslope import write_xslope, write_xslope_zip


def _soil_library(soils: str | Path | None, mapping: dict[str, str | None] | None) -> dict | None:
    """De grondsoorten van een D-Stability-bestand, met de koppeling."""
    if soils is None:
        if mapping:
            raise ValueError("Een koppeling van grondsoorten heeft het D-Stability-bestand met de grondsoorten nodig (soils=...)")
        return None
    library = read_stix_soils(Path(soils).read_bytes())
    if mapping is not None:
        library["mapping"] = dict(mapping)
    return library


@dataclass
class CrossSection:
    """Een dwarsdoorsnede van een corridormodel bij één stationering, haaks op
    de referentielijn: grondpolygonen in (x, z) met x de afstand tot de
    referentielijn (m, positief binnenwaarts) en z de hoogte."""

    # De gevraagde stationering (m); slice.chainage is die van de gebruikte kolom.
    chainage: float
    slice: ProfileSlice
    polygons: list[SoilPolygon]
    crs: ModelCrs

    @property
    def label(self) -> str:
        """Bijvoorbeeld "km0.250": de naam van het scenario in D-Stability en D-Geo Flow."""
        return chainage_label(self.chainage)

    @property
    def soils(self) -> list[str]:
        """De namen van de grondsoorten in deze doorsnede."""
        return sorted({self.slice.soils[p.soil] for p in self.polygons})

    @property
    def start(self) -> tuple[float, float]:
        """x, y (in het coördinatenstelsel van het model) van het buitenwaartse eind (A)."""
        return self.slice.to_world(float(self.slice.u[0] - self.slice.du / 2))

    @property
    def end(self) -> tuple[float, float]:
        """x, y van het binnenwaartse eind (B)."""
        return self.slice.to_world(float(self.slice.u[-1] + self.slice.du / 2))

    def _check(self) -> None:
        if not self.polygons:
            raise ValueError(f"Het model heeft geen voxels bij stationering {self.chainage:g} m")

    def to_dstability(self, path: str | Path, soils: str | Path | None = None, mapping: dict[str, str | None] | None = None) -> Path:
        """Schrijft een D-Stability-model (.stix). Met `soils` (een .stix-bestand)
        gaan al zijn grondsoorten met hun parameters erin, en krijgt elke
        modelgrondsoort de grondsoort die `mapping` op code noemt
        (modelgrondsoort → code; None: geen), of anders die met dezelfde
        code of naam. Water, belastingen en rekeninstellingen vul je zelf in."""
        self._check()
        path = Path(path)
        write_stix(self.slice, self.polygons, path, self.label, _soil_library(soils, mapping))
        return path

    def to_dgeoflow(self, path: str | Path) -> Path:
        """Schrijft een D-Geo Flow-model (.flox): de lagen en grondsoorten (alleen
        namen). Randvoorwaarden, mesh en rekeninstellingen vul je zelf in."""
        self._check()
        path = Path(path)
        write_flox(self.slice, self.polygons, path, self.label)
        return path

    def to_xslope(self, path: str | Path, soils: str | Path | None = None, mapping: dict[str, str | None] | None = None) -> Path:
        """Schrijft een XSLOPE-invoerbestand. Een `.xlsx`-pad krijgt het
        werkboek en `<naam>_styles.json` (de kleuren, die XSLOPE Studio leest)
        ernaast; een `.zip`-pad krijgt beide in één zip. `soils` en `mapping`
        als bij to_dstability: de materialen krijgen dan volumegewichten en,
        voor Mohr-Coulomb, c en phi."""
        self._check()
        path = Path(path)
        write = write_xslope_zip if path.suffix.lower() == ".zip" else write_xslope
        write(self.slice, self.polygons, path, self.label, _soil_library(soils, mapping))
        return path

    def to_dxf(self, path: str | Path) -> Path:
        """Schrijft een 2D-DXF-tekening: de grenzen tussen grondsoorten, de
        omtrek en een laag per grondsoort (bijvoorbeeld voor GEO5 of CAD)."""
        self._check()
        path = Path(path)
        write_dxf(self.slice, self.polygons, path, self.label)
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Schrijft de hoekpunten van de polygonen als CSV (komma als
        scheidingsteken, punt als decimaalteken): kolommen polygoon,
        grondsoort, punt, x, z; een regel per hoekpunt, tegen de klok in."""
        self._check()
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["polygoon", "grondsoort", "punt", "x", "z"])
            for n, polygon in enumerate(self.polygons):
                for i, (x, z) in enumerate(polygon.points):
                    writer.writerow([n, self.slice.soils[polygon.soil], i, round(x, 3), round(z, 3)])
        return path

    def plot(self, ax=None, ground: bool = True, legend: bool = True, edges: bool = True):
        """Tekent de doorsnede met matplotlib (zie plot.plot_section); geeft de `ax`."""
        from .plot import plot_section

        return plot_section(self, ax, ground, legend, edges)


@dataclass
class SoilLayer:
    """Een laag van één grondsoort in een grondkolom."""

    top: float
    bottom: float
    soil: str

    @property
    def thickness(self) -> float:
        return self.top - self.bottom


@dataclass
class SoilColumn:
    """De grondopbouw op één plek van het model, als een (virtuele) boring:
    de lagen van boven naar beneden."""

    chainage: float
    offset: float
    x: float
    y: float
    # Maaiveld uit de hoogtedata (NaN als onbekend).
    ground: float
    layers: list[SoilLayer]
    height_unit: str

    @property
    def top(self) -> float:
        """De bovenkant van de bovenste laag (NaN zonder lagen)."""
        return self.layers[0].top if self.layers else float("nan")

    def plot(self, ax=None, colors: dict[str, str] | None = None):
        """Tekent de kolom met matplotlib (zie plot.plot_column); geeft de `ax`."""
        from .plot import plot_column

        return plot_column(self, ax, colors)


class Model:
    """Een LithoVox-corridormodel: het voxelmodelbestand (.nc) dat LithoVox
    levert. ValueError als het bestand er geen is."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.metadata: dict = read_metadata(f)
            self.grid = read_grid(f)

    def __repr__(self) -> str:
        return f"Model({self.project!r}, {self.name!r}, {self.length:.0f} m, {len(self.soils)} grondsoorten)"

    @property
    def name(self) -> str:
        """De naam van de referentielijn van het model."""
        return self.metadata.get("name", "")

    @property
    def project(self) -> str:
        return self.metadata.get("project", "")

    @property
    def created(self) -> str:
        """Wanneer LithoVox het model maakte (ISO 8601, UTC)."""
        return self.metadata.get("created", "")

    @property
    def soils(self) -> list[str]:
        """De namen van de grondsoorten, in de volgorde van hun codes in het bestand."""
        return list(self.grid.soils)

    @property
    def colors(self) -> dict[str, str]:
        """De kleur van elke grondsoort (`#rrggbb`), waar het model er een heeft."""
        return dict(self.grid.colors)

    @property
    def length(self) -> float:
        """De lengte van de referentielijn (m)."""
        return self.grid.length

    @property
    def crs(self) -> str:
        """Het coördinatenstelsel, bijvoorbeeld "EPSG:28992"."""
        return self.grid.crs.name

    @property
    def vertical_datum(self) -> str | None:
        """Het verticale datum van de hoogtes, bijvoorbeeld "NAP"; None als het onbekend is."""
        return self.grid.crs.vertical_datum

    @property
    def shape(self) -> tuple[int, int, int]:
        """Het aantal voxels langs de lijn, dwars erop en omhoog."""
        return self.grid.shape

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """De afmetingen van een voxel (m): langs de lijn, dwars erop en omhoog."""
        return self.grid.ds, self.grid.dd, self.grid.dz

    @property
    def offsets(self) -> tuple[float, float]:
        """De afstanden (m) van de buiten- en binnenrand van het model tot de referentielijn."""
        d = self.grid.offset_edges()
        return float(d[0]), float(d[-1])

    @property
    def heights(self) -> tuple[float, float]:
        """De onder- en bovenkant van het grid (m)."""
        z = self.grid.z_edges()
        return float(z[0]), float(z[-1])

    def cross_section(self, chainage: float, smoothing: float = SMOOTHING) -> CrossSection:
        """De dwarsdoorsnede bij de kolom die het dichtst bij `chainage` ligt (m
        langs de referentielijn). `smoothing` (in voxels) maakt de grenzen
        tussen grondsoorten glad; 0 houdt de trap van de voxels."""
        with h5py.File(self.path, "r") as f:
            section = read_slice(f, chainage)
        return CrossSection(chainage=chainage, slice=section, polygons=section_polygons(section, smoothing), crs=self.grid.crs)

    def cross_sections(self, every: float, start: float = 0.0, stop: float | None = None, smoothing: float = SMOOTHING) -> Iterator[CrossSection]:
        """Dwarsdoorsneden om de `every` m, van `start` tot en met `stop`
        (standaard: het eind van de lijn), een voor een gemaakt."""
        stop = self.length if stop is None else stop
        for chainage in np.arange(start, stop + 1e-9, every):
            yield self.cross_section(float(chainage), smoothing)

    def locate(self, x: float, y: float) -> tuple[float, float]:
        """De stationering en afstand tot de referentielijn (m) van de modelkolom
        op punt x, y (in het coördinatenstelsel van het model). ValueError als
        het punt buiten het model ligt."""
        grid = self.grid
        dx, dy = x - grid.station[:, 0], y - grid.station[:, 1]
        offset = dx * grid.normal[:, 0] + dy * grid.normal[:, 1]
        along = np.abs(dx * grid.normal[:, 1] - dy * grid.normal[:, 0])
        low, high = self.offsets
        # De kolommen waarvan de lijn haaks op de referentielijn vlak langs het punt gaat.
        near = np.nonzero((offset >= low) & (offset <= high) & (along <= grid.ds))[0]
        with h5py.File(self.path, "r") as f:
            source = f["source"] if "source" in f else None
            for i in near[np.argsort(along[near])]:
                j = int(np.clip(np.rint((offset[i] - grid.offset[0]) / grid.dd), 0, len(grid.offset) - 1))
                # Aan de binnenkant van een bocht liet het model kolommen weg: dan de volgende.
                if source is None or source[i, j] >= 0:
                    return round(float(grid.chainage[i]), 3), round(float(grid.offset[j]), 3)
        raise ValueError(f"Het punt ({x:g}, {y:g}) ligt buiten het model")

    def soil_column(self, chainage: float, offset: float) -> SoilColumn:
        """De grondkolom van het model bij stationering `chainage` en afstand
        `offset` tot de referentielijn (m, positief binnenwaarts): de
        dichtstbijzijnde kolom van voxels."""
        grid = self.grid
        low, high = self.offsets
        if not low <= offset <= high:
            raise ValueError(f"Afstand {offset:g} m ligt buiten het model ({low:g} tot {high:g} m)")
        i = int(np.argmin(np.abs(grid.chainage - chainage)))
        j = int(np.clip(np.rint((offset - grid.offset[0]) / grid.dd), 0, len(grid.offset) - 1))
        with h5py.File(self.path, "r") as f:
            codes = np.asarray(f["soil"][i, j], dtype=np.int16)
            ground = float(f["ground"][i, j]) if "ground" in f else float("nan")
        # Op mm afgerond: de randen zijn sommen van floats.
        edges = np.round(grid.z_edges(), 3)
        layers: list[SoilLayer] = []
        # Van boven naar beneden; aaneengesloten voxels van dezelfde grondsoort vormen één laag.
        for k in range(len(codes) - 1, -1, -1):
            code = int(codes[k])
            if code < 0:
                continue
            name = grid.soils[code]
            if layers and layers[-1].soil == name and abs(layers[-1].bottom - edges[k + 1]) < 1e-9:
                layers[-1].bottom = float(edges[k])
            else:
                layers.append(SoilLayer(top=float(edges[k + 1]), bottom=float(edges[k]), soil=name))
        x, y = grid.station[i] + grid.offset[j] * grid.normal[i]
        return SoilColumn(
            chainage=round(float(grid.chainage[i]), 3),
            offset=round(float(grid.offset[j]), 3),
            x=float(x),
            y=float(y),
            ground=ground,
            layers=layers,
            height_unit=grid.crs.height_unit,
        )

    def soil_column_at(self, x: float, y: float) -> SoilColumn:
        """De grondkolom van het model op punt x, y (zie locate)."""
        return self.soil_column(*self.locate(x, y))

    def plot_map(self, ax=None, chainages: list[float] | None = None, ground: bool = True, step: float | None = None):
        """Tekent het model van bovenaf met matplotlib (zie plot.plot_map); geeft de `ax`."""
        from .plot import plot_map

        return plot_map(self, ax, chainages, ground, step)

    def to_glb(self, path: str | Path, voxel_width: float = 1.0, voxel_height: float = 0.2, voxel_edges: bool = False) -> Path:
        """Schrijft het model als 3D-mesh (GLB), met voxels van ongeveer
        `voxel_width` x `voxel_height` m; in lokale meters rond een oorsprong
        (die staat in de extras van de GLB). `voxel_edges` tekent de randen van
        elke voxel donkerder."""
        path = Path(path)
        path.write_bytes(model_to_glb(self.path, voxel_width, voxel_height, False, voxel_edges))
        return path

    def to_tiles(self, path: str | Path, voxel_width: float = 1.0, voxel_height: float = 0.2, voxel_edges: bool = True) -> Path:
        """Schrijft het model als OGC 3D Tiles (een zip met 3d/tileset.json), op de
        wereldbol geplaatst, bijvoorbeeld voor Cesium of ArcGIS."""
        path = Path(path)
        with tempfile.TemporaryDirectory() as folder:
            # Verplaatst, niet hernoemd: de tijdelijke map kan op een andere schijf staan.
            shutil.move(model_to_tiles(self.path, Path(folder), voxel_width, voxel_height, False, voxel_edges), path)
        return path

    def to_laz(self, path: str | Path, voxel_width: float = 1.0, voxel_height: float = 0.2) -> Path:
        """Schrijft het model als LAZ-puntenwolk (een punt per voxel, klasse 64 +
        code van de grondsoort), bijvoorbeeld voor QGIS of ArcGIS."""
        path = Path(path)
        model_to_laz(self.path, path, voxel_width, voxel_height)
        return path
