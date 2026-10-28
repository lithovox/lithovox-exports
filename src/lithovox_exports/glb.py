"""Een corridormodel als 3D-mesh (glTF 2.0, binair: GLB), een brok tegelijk
opgebouwd: één mesh per grondsoort, in de kleur van die grondsoort."""

import json
import re
import struct
import zlib
from typing import Callable

import numpy as np

from .grid import CorridorChunk, CorridorGrid

# Kleur van een grondsoort zonder (leesbare) kleur.
DEFAULT_COLOR = "#9a9a9a"
GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
FLOAT = 5126
UNSIGNED_INT = 5125


def srgb_to_linear(color: str) -> list[float]:
    """Een CSS-kleur (#rgb of #rrggbb) als lineaire RGB (glTF's baseColorFactor); grijs als hij onleesbaar is."""
    match = re.fullmatch(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})", color.strip())
    hex_digits = match.group(1) if match else DEFAULT_COLOR[1:]
    if len(hex_digits) == 3:
        hex_digits = "".join(c * 2 for c in hex_digits)
    channels = [int(hex_digits[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    return [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]


def model_origin(grid: CorridorGrid) -> tuple[float, float]:
    """x, y (hele meters) waar de posities van een lokale GLB ten opzichte van liggen: het midden van het model."""
    low, high = grid.extent()
    centre = (low + high) / 2
    return float(np.round(centre[0])), float(np.round(centre[1]))


def chunk_quads(grid: CorridorGrid, chunk: CorridorChunk, soil_boundaries: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """De vlakken van de eigen voxels van het brok die aan geen voxel grenzen
    (of, met `soil_boundaries`, aan een andere grondsoort), als vierhoeken
    van knoopnummers (m, 4), tegen de klok in gezien van buiten de voxel in
    het (s, d, z)-stelsel van het grid, en de code van de grondsoort van die
    voxel (m,).

    Knoop (i, j, k) is het kolomhoekpunt (i, j) (CorridorGrid.corner_xy) aan
    de onderkant van rij k; zijn nummer is (i * (nd + 1) + j) * (nz + 1) + k."""
    _, nd, nz = grid.shape
    p = np.pad(chunk.matrix, 1, constant_values=-1)
    c0, c1 = chunk.start - chunk.lo + 1, chunk.stop - chunk.lo + 1
    core = p[c0:c1, 1:-1, 1:-1]

    def node(i, j, k):
        return (i * (nd + 1) + j) * (nz + 1) + k

    quads: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    # Per as: de buren aan de plus- en de minkant.
    neighbours = {
        "s": (p[c0 + 1 : c1 + 1, 1:-1, 1:-1], p[c0 - 1 : c1 - 1, 1:-1, 1:-1]),
        "d": (p[c0:c1, 2:, 1:-1], p[c0:c1, :-2, 1:-1]),
        "z": (p[c0:c1, 1:-1, 2:], p[c0:c1, 1:-1, :-2]),
    }
    filled = core >= 0
    for axis, (plus, minus) in neighbours.items():
        for side, other in ((1, plus), (0, minus)):
            face = filled & ((core != other) if soil_boundaries else (other < 0))
            i, j, k = np.nonzero(face)
            i = i.astype(np.int64) + chunk.start
            j = j.astype(np.int64)
            k = k.astype(np.int64)
            # De hoekpunten van het vlak in de cyclische volgorde (s -> d -> z)
            # waarbij de normaal van het vlak langs +as wijst.
            if axis == "s":
                q = i + side
                corners = [node(q, j, k), node(q, j + 1, k), node(q, j + 1, k + 1), node(q, j, k + 1)]
            elif axis == "d":
                q = j + side
                corners = [node(i, q, k), node(i, q, k + 1), node(i + 1, q, k + 1), node(i + 1, q, k)]
            else:
                q = k + side
                corners = [node(i, j, q), node(i + 1, j, q), node(i + 1, j + 1, q), node(i, j + 1, q)]
            quad = np.stack(corners, axis=1)
            # De minkant kijkt de andere kant op.
            quads.append(quad if side == 1 else quad[:, ::-1])
            codes.append(core[face])
    return np.concatenate(quads), np.concatenate(codes)


class CorridorGlbBuilder:
    """Bouwt het mesh van een corridormodel een brok tegelijk (voeg de brokken
    op volgorde toe): één mesh per grondsoort, in de kleur van de grondsoort.
    Standaard alleen de vlakken die aan geen voxel grenzen: de buitenkant van
    het model, alles wat een ondoorzichtige weergave laat zien (met
    `soil_boundaries` ook de vlakken tussen grondsoorten, voor weergaven die
    het model openknippen). De hoekpunten blijven in modelcoördinaten (x, y,
    hoogte) tot er een GLB gemaakt wordt (`glb` voor een model in lokale
    meters, `ecef_glb` voor een tegel van 3D Tiles), zodat hetzelfde mesh
    voor beide dient. Geen verticale overdrijving.

    Standaard delen de vlakken hun hoekpunten en zijn er geen normalen
    (viewers die vlak belichten, zoals three.js met flatShading, hebben ze
    niet nodig). Met `voxel_edges` heeft elk vlak zijn eigen vier hoekpunten,
    met een normaal (zodat viewers die alleen meshes met normalen belichten,
    zoals CesiumJS, het belichten) en textuurcoördinaten voor een kleine
    textuur met een donkere rand, die de randen van elke voxel tekent; het
    bestand wordt dan ongeveer vier keer zo groot."""

    def __init__(self, grid: CorridorGrid, soil_boundaries: bool = False, voxel_edges: bool = False):
        self.grid = grid
        self.soil_boundaries = soil_boundaries
        self.voxel_edges = voxel_edges
        self.z_edges = grid.z_edges()
        # Per code: x, y, z van de hoekpunten en de driehoeken, en het aantal hoekpunten tot nu toe.
        self.points: dict[int, list[np.ndarray]] = {}
        self.indices: dict[int, list[np.ndarray]] = {}
        self.counts: dict[int, int] = {}

    @property
    def empty(self) -> bool:
        return not self.points

    def add(self, chunk: CorridorChunk) -> None:
        grid = self.grid
        _, nd, nz = grid.shape
        quads, codes = chunk_quads(grid, chunk, self.soil_boundaries)
        if len(quads) == 0:
            return
        # Het (s, d, z)-stelsel van het grid is linkshandig als de buitenkant
        # links ligt (d wijst binnenwaarts, rechts van s); dan worden de vlakken omgedraaid.
        if grid.outwards_side == "left":
            quads = quads[:, ::-1]
        corners = grid.corner_xy(chunk.start, chunk.stop)
        for code in np.unique(codes):
            code = int(code)
            if self.voxel_edges:
                # Elk vlak zijn eigen vier hoekpunten (op volgorde: 0-1-2-3 rond het vlak).
                ids = quads[codes == code].reshape(-1)
                local = np.arange(len(ids))
            else:
                ids, local = np.unique(quads[codes == code], return_inverse=True)
            k = ids % (nz + 1)
            ij = ids // (nz + 1)
            j = ij % (nd + 1)
            i = ij // (nd + 1) - chunk.start
            xy = corners[i, j]
            points = np.stack([xy[:, 0], xy[:, 1], self.z_edges[k]], axis=1)
            local = local.reshape(-1, 4).astype(np.uint32) + self.counts.get(code, 0)
            triangles = np.concatenate([local[:, [0, 1, 2]], local[:, [0, 2, 3]]], axis=1).reshape(-1)
            self.points.setdefault(code, []).append(points)
            self.indices.setdefault(code, []).append(triangles)
            self.counts[code] = self.counts.get(code, 0) + len(points)

    def world_points(self) -> np.ndarray:
        """(n, 3) alle hoekpunten (x, y, hoogte)."""
        parts = [p for code in self.points for p in self.points[code]]
        return np.concatenate(parts) if parts else np.zeros((0, 3))

    def glb(self, origin: tuple[float, float], extras: dict | None = None) -> bytes:
        """Een GLB in lokale meters, y omhoog: x = x - oorsprong x, y = hoogte,
        z = -(y - oorsprong y). De floats van glTF zijn te grof voor
        wereldcoördinaten, vandaar de oorsprong; die staat in de `extras` van
        de asset en de scene (met het coördinatenstelsel en de assen)."""
        x0, y0 = origin
        crs = self.grid.crs

        def to_local(points: np.ndarray) -> np.ndarray:
            return np.stack([points[:, 0] - x0, points[:, 2], -(points[:, 1] - y0)], axis=1)

        info = {
            "crs": crs.name,
            "verticalDatum": crs.vertical_datum,
            "origin": [x0, y0, 0.0],
            "units": "m",
            "axes": {"x": "x - origin x", "y": f"z ({crs.height_unit})", "z": "-(y - origin y)"},
            "verticalExaggeration": 1,
            "soilBoundaries": self.soil_boundaries,
            **(extras or {}),
        }
        return self._pack(to_local, info)

    def ecef_glb(self, to_ecef: Callable[[np.ndarray], np.ndarray], centre: np.ndarray) -> bytes:
        """Een GLB voor een tegel van 3D Tiles: hoekpunten als ECEF-verschil met
        `centre` (de transformatie van de tegel), in de y-omhoog-assen van glTF
        (3D Tiles maakt van glTF's y omhoog z omhoog, dus een verschil
        (X, Y, Z) wordt opgeslagen als (X, Z, -Y))."""

        def to_tile(points: np.ndarray) -> np.ndarray:
            offset = to_ecef(points) - centre
            return np.stack([offset[:, 0], offset[:, 2], -offset[:, 1]], axis=1)

        return self._pack(to_tile, {"soilBoundaries": self.soil_boundaries, "voxelEdges": self.voxel_edges})

    def _pack(self, transform: Callable[[np.ndarray], np.ndarray], extras: dict) -> bytes:
        grid = self.grid
        binary = bytearray()
        buffer_views: list[dict] = []
        accessors: list[dict] = []
        meshes: list[dict] = []
        materials: list[dict] = []
        nodes: list[dict] = []

        def add_view(data: bytes, target: int | None) -> int:
            while len(binary) % 4:
                binary.append(0)
            view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
            if target is not None:
                view["target"] = target
            buffer_views.append(view)
            binary.extend(data)
            return len(buffer_views) - 1

        def add_accessor(data: np.ndarray, kind: str, target: int, bounds: bool = False) -> int:
            accessor = {"bufferView": add_view(data.tobytes(), target), "componentType": FLOAT, "count": len(data), "type": kind}
            if bounds:
                accessor.update(min=data.min(axis=0).tolist(), max=data.max(axis=0).tolist())
            accessors.append(accessor)
            return len(accessors) - 1

        present = sorted(self.points)
        for n, code in enumerate(present):
            positions = transform(np.concatenate(self.points[code])).astype(np.float32)
            triangles = np.concatenate(self.indices[code])
            attributes = {"POSITION": add_accessor(positions, "VEC3", ARRAY_BUFFER, bounds=True)}
            if self.voxel_edges:
                attributes["NORMAL"] = add_accessor(_face_normals(positions), "VEC3", ARRAY_BUFFER)
                faces = len(positions) // 4
                uv = np.tile(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32), (faces, 1))
                attributes["TEXCOORD_0"] = add_accessor(uv, "VEC2", ARRAY_BUFFER)
            index_view = add_view(triangles.tobytes(), ELEMENT_ARRAY_BUFFER)
            accessors.append({"bufferView": index_view, "componentType": UNSIGNED_INT, "count": len(triangles), "type": "SCALAR"})
            name = grid.soils[code] if 0 <= code < len(grid.soils) else f"code {code}"
            pbr: dict = {
                "baseColorFactor": [*srgb_to_linear(grid.colors.get(name, DEFAULT_COLOR)), 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            }
            if self.voxel_edges:
                # De kleur van de grondsoort maal de randtextuur: donkerder aan de randen van elke voxel.
                pbr["baseColorTexture"] = {"index": 0}
            materials.append({"name": name, "pbrMetallicRoughness": pbr})
            meshes.append({"name": name, "primitives": [{"attributes": attributes, "indices": len(accessors) - 1, "material": n, "mode": 4}]})
            nodes.append({"name": name, "mesh": n})

        extras = {**extras, "soils": [grid.soils[c] for c in present if 0 <= c < len(grid.soils)]}
        gltf: dict = {
            "asset": {"version": "2.0", "generator": "lithovox-exports", "extras": extras},
            "scene": 0,
            "scenes": [{"name": "corridor model", "nodes": list(range(len(nodes))), "extras": extras}],
            "nodes": nodes,
        }
        if meshes and self.voxel_edges:
            gltf.update(
                images=[{"bufferView": add_view(border_png(), None), "mimeType": "image/png"}],
                # Lineair, met mipmaps (de randen vervagen op afstand in plaats van te flikkeren), aan de rand afgekapt.
                samplers=[{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
                textures=[{"sampler": 0, "source": 0}],
            )
        if meshes:
            gltf.update(
                meshes=meshes,
                materials=materials,
                accessors=accessors,
                bufferViews=buffer_views,
                buffers=[{"byteLength": len(binary)}],
            )
        return pack_glb(gltf, bytes(binary))


def _face_normals(positions: np.ndarray) -> np.ndarray:
    """(4m, 3) de eenheidsnormaal van elk vlak (elk vier hoekpunten, tegen de
    klok in gezien van buiten), voor elk van zijn hoekpunten."""
    faces = positions.reshape(-1, 4, 3).astype(np.float64)
    normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.where(length > 0, normal / np.where(length > 0, length, 1), [0.0, 1.0, 0.0])
    return np.repeat(normal, 4, axis=0).astype(np.float32)


# De randtextuur: wit met een donkere rand; vermenigvuldigd met de kleur van
# een grondsoort tekent hij een voxelvlak in die kleur met donkere randen.
BORDER_SIZE = 32
BORDER_WIDTH = 2
BORDER_SHADE = 165


def border_png(size: int = BORDER_SIZE, border: int = BORDER_WIDTH, shade: int = BORDER_SHADE) -> bytes:
    """Een grijswaarden-PNG van size x size: wit, met een rand van `border` pixels in grijs `shade`."""
    rows = []
    for y in range(size):
        edge_row = y < border or y >= size - border
        row = bytes(shade if edge_row or x < border or x >= size - border else 255 for x in range(size))
        rows.append(b"\x00" + row)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    header = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)  # 8 bit grijswaarden
    return signature + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def pack_glb(gltf: dict, binary: bytes) -> bytes:
    """Een GLB-bestand: header, het JSON-deel (aangevuld met spaties) en het binaire deel (aangevuld met nullen)."""
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    chunks = struct.pack("<II", len(json_bytes), CHUNK_JSON) + json_bytes
    if binary:
        padded = binary + b"\x00" * (-len(binary) % 4)
        chunks += struct.pack("<II", len(padded), CHUNK_BIN) + padded
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(chunks)) + chunks
