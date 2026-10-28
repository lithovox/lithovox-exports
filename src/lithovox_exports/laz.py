"""Een corridormodel als LAZ-puntenwolk: een punt per voxel."""

from pathlib import Path

import laspy
import numpy as np

from .grid import CorridorChunk, CorridorGrid

# LAS-klassen 0-63 hebben een vaste (ASPRS-)betekenis, zoals 2 = maaiveld;
# 64 en hoger zijn vrij, dus grondsoort i krijgt klasse FIRST_CLASS + i.
FIRST_CLASS = 64


class PointCloudWriter:
    """Schrijft de middens van de voxels van een corridormodel als LAZ-
    puntenwolk (LAS 1.4, puntformaat 6, in het coördinatenstelsel van het
    model): klasse FIRST_CLASS + code van de grondsoort, user data de code.
    QGIS en ArcGIS lezen hem (QGIS indexeert hem bij het eerste gebruik).
    Voeg de brokken op volgorde toe en roep daarna close aan."""

    def __init__(self, path: Path, grid: CorridorGrid):
        self.grid = grid
        header = laspy.LasHeader(point_format=6, version="1.4")
        low, _ = grid.extent()
        header.offsets = np.array([np.floor(low[0]), np.floor(low[1]), 0.0])
        header.scales = np.array([0.001, 0.001, 0.001])
        header.add_crs(grid.crs.crs)
        self.header = header
        self.writer = laspy.open(str(path), mode="w", header=header, do_compress=True)
        self.count = 0

    def add(self, chunk: CorridorChunk) -> None:
        matrix = chunk.matrix[chunk.core]
        i, j, k = np.nonzero(matrix >= 0)
        if len(i) == 0:
            return
        xy = self.grid.column_xy(chunk.start, chunk.stop)[i, j]
        codes = matrix[i, j, k].astype(np.uint8)
        points = laspy.ScaleAwarePointRecord.zeros(len(i), header=self.header)
        points.x = xy[:, 0]
        points.y = xy[:, 1]
        points.z = self.grid.z[k]
        points.classification = FIRST_CLASS + codes
        points.user_data = codes
        self.writer.write_points(points)
        self.count += len(i)

    def close(self) -> None:
        self.writer.close()
