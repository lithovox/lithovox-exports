"""De D-Geo Flow-export, gerekend door D-Geo Flow zelf, als bewaarde benchmark.

`tests/data/dgeoflow_benchmark.json` bewaart de invoer die de export schreef
(een SHA-256 per bestand in de .flox) en het resultaat van D-Geo Flow
daarop. De gewone `pytest` controleert dat de export nog precies die invoer
schrijft, zodat het bewaarde resultaat blijft gelden, zonder D-Geo Flow.
Met DGEOFLOW_FOLDER (de map `bin` van een D-Geo Flow-installatie) rekent
de tweede test de invoer opnieuw door en vergelijkt het resultaat. Met
ook LITHOVOX_WRITE_BENCHMARK=1 schrijft hij de benchmark opnieuw: doe dat
alleen bewust, en meld wat er veranderde.

De doorsnede is die van het synthetische testmodel (conftest.py), met zand
aan een doorlatende en klei aan een slecht doorlatende grondsoort uit een
eigen .flox, veen aan geen; een vaste stijghoogte van 2 m aan de buitenrand
en -1 m aan de binnenrand, zoals een gebruiker ze in D-Geo Flow toevoegt."""

import hashlib
import json
import math
import os
import platform
import subprocess
import zipfile
from pathlib import Path

import pytest
from geolib.geometry.one import Point
from geolib.models.dgeoflow import DGeoFlowModel
from geolib.soils.soil import Soil

from lithovox_exports import Model

BENCHMARK = Path(__file__).parent / "data" / "dgeoflow_benchmark.json"
PERMEABILITY = {"ZAND": (10.0, 5.0), "KLEI": (1e-3, 5e-4)}  # horizontaal, verticaal (m/dag)
MAPPING = {"zand": "ZAND", "klei": "KLEI", "veen": None}
RIVER_M, POLDER_M = 2.0, -1.0
# Relatieve tolerantie voor het resultaat van D-Geo Flow (C10: niet verruimen om een test te laten slagen).
TOLERANCE = 1e-6


def library(folder: Path) -> Path:
    flox = DGeoFlowModel()
    flox.datastructure.soils.Soils = []
    for code in PERMEABILITY:
        flox.add_soil(Soil(name=code.title(), code=code))
    for soil in flox.datastructure.soils.Soils:
        soil.HorizontalPermeability, soil.VerticalPermeability = PERMEABILITY[soil.Code]
    path = folder / "bibliotheek.flox"
    flox.serialize(path)
    return path


def case(model_path: Path, folder: Path) -> Path:
    """De export met de gekoppelde grondsoorten en de twee randvoorwaarden."""
    section = Model(model_path).cross_section(15.0)
    path = section.to_dgeoflow(folder / "case.flox", soils=library(folder), mapping=MAPPING)
    flox = DGeoFlowModel()
    flox.parse(path)
    points = [(p.X, p.Z) for layer in flox.datastructure.geometries[0].Layers for p in layer.Points]
    left, right = min(x for x, _ in points), max(x for x, _ in points)
    for x, head, label in ((left, RIVER_M, "Rivier"), (right, POLDER_M, "Polder")):
        z = sorted({pz for px, pz in points if abs(px - x) < 1e-9})
        flox.add_boundary_condition([Point(x=x, z=z[0]), Point(x=x, z=z[-1])], head, label=label)
    flox.serialize(path)
    return path


def input_hashes(path: Path) -> dict[str, str]:
    """Per JSON-bestand in de .flox zoals de export (met de randvoorwaarden)
    het schrijft, voor D-Geo Flow rekent (zonder projectinfo met de datum en
    zonder resultaten), de SHA-256 van zijn inhoud, genormaliseerd."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name == "projectinfo.json" or name.startswith("results/") or not name.endswith(".json"):
                continue
            text = json.dumps(json.loads(z.read(name)), sort_keys=True, separators=(",", ":"))
            out[name] = hashlib.sha256(text.encode()).hexdigest()
    return out


def summary(path: Path) -> dict:
    """Het resultaat van D-Geo Flow, samengevat: aantallen, stijghoogtes (m) en debieten."""
    with zipfile.ZipFile(path) as z:
        result = json.loads(z.read("results/groundwaterflow/groundwaterflowresult.json"))
    nodes = [n for element in result["Elements"] for n in element["NodeResults"]]
    heads = [n["HydraulicHead"] for n in nodes]
    discharge = [abs(n["HydraulicDischarge"]) for n in nodes]
    return {
        "elements": len(result["Elements"]),
        "nodes": len(nodes),
        "head_min_m": min(heads),
        "head_max_m": max(heads),
        "head_mean_m": sum(heads) / len(heads),
        "discharge_max": max(discharge),
        "discharge_sum": sum(discharge),
        "phreatic_line_segments": len(result.get("PhreaticLineSegments") or []),
    }


def test_the_dgeoflow_benchmark_still_holds(model_path, tmp_path):
    stored = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    now = input_hashes(case(model_path, tmp_path))
    changed = sorted(n for n in set(now) | set(stored["input"]) if now.get(n) != stored["input"].get(n))
    assert not changed, (
        f"De D-Geo Flow-export schrijft andere invoer dan waarop de benchmark gerekend is ({', '.join(changed)}). "
        "Reken opnieuw met DGEOFLOW_FOLDER en LITHOVOX_WRITE_BENCHMARK=1, en meld de verschillen in het resultaat."
    )


def console(folder: Path) -> list[str]:
    if platform.system() == "Windows":
        return [str(folder / "D-GeoFlow Console.exe")]
    return [os.environ.get("DOTNET", "dotnet"), str(folder / "D-GeoFlow Console.dll")]


@pytest.mark.skipif(not os.environ.get("DGEOFLOW_FOLDER"), reason="needs DGEOFLOW_FOLDER (D-Geo Flow's bin)")
def test_d_geoflow_calculates_the_case(model_path, tmp_path):
    path = case(model_path, tmp_path)
    # Voor het rekenen: D-Geo Flow schrijft ook in de invoer (het resultaat bij het scenario).
    exported = input_hashes(path)
    run = subprocess.run(console(Path(os.environ["DGEOFLOW_FOLDER"])) + [str(path), "1", "1"],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    result = summary(path)
    # Wat er hoe dan ook moet gelden: de stijghoogtes liggen tussen de twee randvoorwaarden.
    assert POLDER_M - 1e-6 <= result["head_min_m"] and result["head_max_m"] <= RIVER_M + 1e-6
    if os.environ.get("LITHOVOX_WRITE_BENCHMARK") == "1":
        BENCHMARK.parent.mkdir(exist_ok=True)
        BENCHMARK.write_text(json.dumps({
            "what": "lithovox-exports' D-Geo Flow export of the synthetic test model, calculated by D-Geo Flow; "
                    "see tests/test_dgeoflow_console.py",
            "d_geoflow": os.environ.get("DGEOFLOW_VERSION", "unknown"),
            "input": exported,
            "result": result,
        }, indent=1) + "\n", encoding="utf-8")
        return
    stored = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    assert exported == stored["input"], "de invoer is veranderd: zie test_the_dgeoflow_benchmark_still_holds"
    for key, value in stored["result"].items():
        got = result[key]
        assert math.isclose(got, value, rel_tol=TOLERANCE, abs_tol=1e-12), f"{key}: {got} ≠ bewaard {value}"
