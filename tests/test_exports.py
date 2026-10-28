import csv
import json
import zipfile

import ezdxf
import h5py
import laspy
import openpyxl
import pytest
from geolib.models.dgeoflow import DGeoFlowModel
from geolib.models.dstability import DStabilityModel

from lithovox_exports import Model
from lithovox_exports.cli import main


@pytest.fixture(scope="module")
def model(model_path) -> Model:
    return Model(model_path)


@pytest.fixture(scope="module")
def section(model):
    return model.cross_section(15.0)


class TestModel:
    def test_info(self, model):
        assert model.name == "testlijn"
        assert model.project == "test"
        assert model.soils == ["zand", "klei", "veen"]
        assert model.colors["zand"] == "#ffcc00"
        assert model.length == pytest.approx(30.0)
        assert model.offsets == pytest.approx((-5.0, 15.0))
        assert model.heights == pytest.approx((-8.0, 2.0))

    def test_crs(self, model):
        assert model.crs == "EPSG:28992"
        # "unknown datum" is geen datum.
        assert model.vertical_datum is None
        assert model.grid.crs.height_unit == "m"

    def test_not_a_model(self, tmp_path):
        path = tmp_path / "other.nc"
        with h5py.File(path, "w") as f:
            f.attrs["title"] = "iets anders"
        with pytest.raises(ValueError):
            Model(path)


class TestCrossSection:
    def test_polygons(self, section):
        assert section.label == "km0.015"
        assert section.soils == ["klei", "veen", "zand"]
        assert all(len(p.points) >= 3 for p in section.polygons)

    def test_lens_without_holes(self, section):
        # De klei om de veenlens is losgeknipt: geen polygoon heeft een gat, dus meer dan één kleivlak.
        assert sum(1 for p in section.polygons if p.soil == 1) >= 2

    def test_ends(self, section):
        (x0, y0), (x1, y1) = section.start, section.end
        # Haaks op een oost-westlijn: de einden verschillen alleen in y; binnenwaarts is naar het zuiden.
        assert x0 == pytest.approx(x1)
        assert y0 == pytest.approx(400005.0) and y1 == pytest.approx(399985.0)

    def test_series(self, model):
        assert [s.label for s in model.cross_sections(10)] == ["km0.000", "km0.010", "km0.020", "km0.030"]

    def test_dstability(self, section, tmp_path):
        path = section.to_dstability(tmp_path / "section.stix")
        stix = DStabilityModel()
        stix.parse(path)
        assert {soil.Code for soil in stix.datastructure.soils.Soils} == {"zand", "klei", "veen"}
        assert len(stix.datastructure.geometries[0].Layers) == len(section.polygons)

    def test_dstability_with_soils(self, section, tmp_path):
        library = section.to_dstability(tmp_path / "library.stix")
        path = section.to_dstability(tmp_path / "section.stix", soils=library, mapping={"zand": "klei", "veen": "klei"})
        stix = DStabilityModel()
        stix.parse(path)
        clay = {soil.Id for soil in stix.datastructure.soils.Soils if soil.Code == "klei"}
        used = {layer.SoilId for layer in stix.datastructure.soillayers[0].SoilLayers}
        assert used <= clay

    def test_mapping_needs_soils(self, section, tmp_path):
        with pytest.raises(ValueError):
            section.to_dstability(tmp_path / "section.stix", mapping={"zand": "klei"})

    def test_dgeoflow(self, section, tmp_path):
        flox = DGeoFlowModel()
        flox.parse(section.to_dgeoflow(tmp_path / "section.flox"))
        assert {soil.Code for soil in flox.datastructure.soils.Soils} == {"zand", "klei", "veen"}

    def test_xslope_xlsx(self, section, tmp_path):
        workbook = openpyxl.load_workbook(section.to_xslope(tmp_path / "section.xlsx"))
        assert "polygon" in workbook.sheetnames
        styles = json.loads((tmp_path / "section_styles.json").read_text(encoding="utf-8"))
        assert styles["materials"]

    def test_xslope_zip(self, section, tmp_path):
        path = section.to_xslope(tmp_path / "section.zip")
        assert sorted(zipfile.ZipFile(path).namelist()) == ["section.xlsx", "section_styles.json"]

    def test_dxf(self, section, tmp_path):
        doc = ezdxf.readfile(section.to_dxf(tmp_path / "section.dxf"))
        layers = {layer.dxf.name for layer in doc.layers}
        assert {"INTERFACES", "OUTLINE", "SOIL_zand", "SOIL_klei", "SOIL_veen"} <= layers

    def test_csv(self, section, tmp_path):
        with section.to_csv(tmp_path / "section.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        assert {row["grondsoort"] for row in rows} == {"zand", "klei", "veen"}
        assert len(rows) == sum(len(p.points) for p in section.polygons)

    def test_plot(self, section):
        ax = section.plot()
        assert ax.get_ylabel() == "hoogte [m]"


class TestSoilColumn:
    def test_layers(self, model):
        column = model.soil_column(15, 2.5)
        assert [layer.soil for layer in column.layers] == ["klei", "veen", "klei", "zand"]
        assert column.top == pytest.approx(1.0)
        assert column.layers[-1].bottom == pytest.approx(-8.0)
        assert column.ground == pytest.approx(1.0)

    def test_locate(self, model):
        column = model.soil_column(15, 2.5)
        assert model.locate(column.x, column.y) == pytest.approx((column.chainage, column.offset))
        assert model.soil_column_at(column.x, column.y).layers == column.layers

    def test_outside(self, model):
        with pytest.raises(ValueError):
            model.locate(0, 0)
        with pytest.raises(ValueError):
            model.soil_column(15, 100)


class TestWholeModel:
    def test_glb(self, model, tmp_path):
        assert model.to_glb(tmp_path / "model.glb").read_bytes()[:4] == b"glTF"

    def test_tiles(self, model, tmp_path):
        names = zipfile.ZipFile(model.to_tiles(tmp_path / "tiles.zip")).namelist()
        assert "3d/tileset.json" in names

    def test_laz(self, model, tmp_path):
        cloud = laspy.read(model.to_laz(tmp_path / "model.laz"))
        assert len(cloud.points) > 0
        assert set(cloud.classification) <= {64, 65, 66}
        assert cloud.header.parse_crs().to_epsg() == 28992

    def test_map(self, model):
        assert model.plot_map(chainages=[15]).get_aspect() == 1.0


class TestCli:
    def test_info(self, model_path, capsys):
        assert main(["info", str(model_path)]) == 0
        assert "testlijn" in capsys.readouterr().out

    def test_section(self, model_path, tmp_path):
        assert main(["doorsnede", str(model_path), "15", str(tmp_path / "s.stix")]) == 0
        assert (tmp_path / "s.stix").exists()

    def test_unknown_format(self, model_path, tmp_path):
        with pytest.raises(SystemExit):
            main(["doorsnede", str(model_path), "15", str(tmp_path / "s.txt")])
