import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pytest

# Figuren zonder scherm.
matplotlib.use("Agg")

SOILS = ["zand", "klei", "veen"]
COLORS = {"zand": "#ffcc00", "klei": "#336633", "veen": "#664422"}


def write_model(path: Path) -> Path:
    """Een klein corridormodel zoals LithoVox het schrijft: 30 m rechte lijn
    van west naar oost (buitenkant links, dus binnenwaarts is naar het
    zuiden), zand onder klei met een veenlens erin, maaiveld op 1 m."""
    ns, nd, nz = 30, 40, 20
    ds, dd, dz = 1.0, 0.5, 0.5
    line = [[100000.0, 400000.0], [100030.0, 400000.0]]
    s = (np.arange(ns) + 0.5) * ds
    s_edge = np.arange(ns + 1) * ds
    d = -5 + (np.arange(nd) + 0.5) * dd
    z = -8 + (np.arange(nz) + 0.5) * dz
    soil = np.full((ns, nd, nz), -1, dtype=np.int8)
    soil[:, :, z < -3] = 0
    soil[:, :, (z > -3) & (z < 1)] = 1
    soil[:, 10:20, (z > -2) & (z < -1)] = 2
    metadata = {
        "format": "lithovox-corridor-model",
        "version": 1,
        "name": "testlijn",
        "project": "test",
        "created": "2026-10-25T12:00:00+00:00",
        "crs": {"code": 28992, "horizontal": "EPSG:28992", "vertical": "unknown datum", "units": "m"},
        "chainageLine": {"points": line, "length": 30.0, "outwardsSide": "left"},
    }
    with h5py.File(path, "w") as f:
        f.attrs["lithovox_format"] = "lithovox-corridor-model"
        f.attrs["lithovox_format_version"] = 1
        f.attrs["lithovox_metadata"] = json.dumps(metadata)
        f.attrs["name"] = "testlijn"
        f.attrs["project"] = "test"
        f.attrs["crs"] = "EPSG:28992"
        f.attrs["chainage_line"] = json.dumps(line)
        f.attrs["outwards_side"] = "left"
        f.attrs["ds"], f.attrs["dd"], f.attrs["dz"] = ds, dd, dz
        f["s"], f["s_edge"], f["d"], f["z"] = s, s_edge, d, z
        f["station_x"], f["station_y"] = 100000.0 + s, np.full(ns, 400000.0)
        f["normal_x"], f["normal_y"] = np.zeros(ns), np.full(ns, -1.0)
        f["edge_station_x"], f["edge_station_y"] = 100000.0 + s_edge, np.full(ns + 1, 400000.0)
        f["edge_normal_x"], f["edge_normal_y"] = np.zeros(ns + 1), np.full(ns + 1, -1.0)
        dataset = f.create_dataset("soil", data=soil, chunks=(4, nd, nz), compression="gzip")
        dataset.attrs["soil_names"] = json.dumps(SOILS)
        dataset.attrs["soil_colors"] = json.dumps(COLORS)
        f["ground"] = np.full((ns, nd), 1.0, dtype=np.float32)
        f["source"] = np.zeros((ns, nd), dtype=np.int32)
    return path


@pytest.fixture(scope="session")
def model_path(tmp_path_factory) -> Path:
    return write_model(tmp_path_factory.mktemp("model") / "model.nc")
