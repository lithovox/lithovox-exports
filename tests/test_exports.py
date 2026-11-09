import csv
import json
import zipfile
from pathlib import Path

import ezdxf
import h5py
import laspy
import openpyxl
import pytest
from lithovox_stability.models.dstability import DStabilityModel

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
        assert [s.label for s in model.cross_sections(10)] == [
            "km0.000",
            "km0.010",
            "km0.020",
            "km0.030",
        ]

    def test_dstability(self, section, tmp_path):
        path = section.to_dstability(tmp_path / "section.stix")
        stix = DStabilityModel()
        stix.parse(path)
        assert {soil.Code for soil in stix.datastructure.soils.Soils} == {
            "zand",
            "klei",
            "veen",
        }
        assert len(stix.datastructure.geometries[0].Layers) == len(section.polygons)

    def linked_model(self, section, tmp_path, **attribute) -> Model:
        """Het testmodel met grondsoortparameters zoals LithoVox ze schrijft:
        zand en klei gekoppeld aan de D-Stability-grondsoort "KLEI" (van een
        .stix die de export zelf geeft), veen aan geen."""
        from lithovox_exports.dstability import read_stix_soils

        from conftest import write_model

        library = read_stix_soils(section.to_dstability(tmp_path / "library.stix").read_bytes())
        clay = next(s for s in library["soils"] if s["Code"] == "klei")
        clay = {**clay, "Code": "KLEI", "Name": "Klei, siltig", "VolumetricWeightAbovePhreaticLevel": 15.5}
        data = {
            "format": "lithovox-soil-parameters",
            "version": 1,
            "mappings": [
                {
                    "software": "d_stability",
                    "label": "D-Stability",
                    "file": "dijk.stix",
                    "sha256": "ab" * 32,
                    "loaded": "2026-11-08T12:00:00Z",
                    "soils": [
                        {"soil": "zand", "link": "soil", "code": "KLEI", "name": "Klei, siltig", "parameters": clay},
                        {"soil": "klei", "link": "soil", "code": "KLEI", "name": "Klei, siltig", "parameters": clay},
                        {"soil": "veen", "link": "none", "code": None},
                    ],
                },
                {
                    "software": "d_settlement",
                    "label": "D-Settlement",
                    "file": "agv.sli",
                    "sha256": "cd" * 32,
                    "loaded": "2026-11-08T12:00:00Z",
                    "soils": [
                        {"soil": "zand", "link": "undecided", "code": None},
                        {"soil": "klei", "link": "soil", "code": "K [13,5 - 15,0]", "name": "K [13,5 - 15,0]",
                         "parameters": [["SoilGamDry", "14.00"], ["SoilCp", "4.42E+01"]]},
                        {"soil": "veen", "link": "suggested", "code": None},
                    ],
                },
            ],
            **attribute,
        }
        return Model(write_model(tmp_path / "linked.nc", data))

    def soils_of_layers(self, path) -> dict:
        stix = DStabilityModel()
        stix.parse(path)
        by_id = {s.Id: s for s in stix.datastructure.soils.Soils}
        layers = {layer.Id: by_id[sl.SoilId] for sl in stix.datastructure.soillayers[0].SoilLayers
                  for layer in stix.datastructure.geometries[0].Layers if layer.Id == sl.LayerId}
        return {layer.Label: layers[layer.Id] for layer in stix.datastructure.geometries[0].Layers}

    def test_soil_parameters(self, section, tmp_path, model):
        assert model.soil_parameter_files == [] and model.soil_parameters() == {}
        linked = self.linked_model(section, tmp_path)
        assert [f["file"] for f in linked.soil_parameter_files] == ["dijk.stix", "agv.sli"]
        stability = linked.soil_parameters()
        assert stability["zand"]["code"] == "KLEI"
        assert stability["zand"]["parameters"]["VolumetricWeightAbovePhreaticLevel"] == 15.5
        assert stability["veen"] is None
        settlement = linked.soil_parameters("d_settlement")
        assert settlement["klei"]["parameters"] == [["SoilGamDry", "14.00"], ["SoilCp", "4.42E+01"]]
        assert settlement["zand"] is None and settlement["veen"] is None

    def soil_links(self, linked: Model, tmp_path, **changes) -> "Path":
        """Een koppelbestand zoals LithoVox Stability het schrijft: de
        D-Stability-koppeling van het testmodel, bij dat model."""
        data = {
            "format": "lithovox-soil-parameters",
            "version": 1,
            "model": {"file": linked.path.name, "sha256": linked.sha256},
            "mappings": [
                {"software": "d_stability", "label": "D-Stability", "file": "dijk.stix",
                 "sha256": "ab" * 32, "loaded": "2026-11-09T12:00:00Z",
                 "soils": list(linked._soil_parameters["mappings"][0]["soils"])},
            ],
            **changes,
        }
        path = tmp_path / "dijk.soil-links.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_dstability_with_soil_links(self, section, tmp_path):
        linked = self.linked_model(section, tmp_path)
        links = self.soil_links(linked, tmp_path)
        path = linked.cross_section(15.0).to_dstability(tmp_path / "section.stix", soil_links=links)
        soils = self.soils_of_layers(path)
        assert soils["zand"].Code == soils["klei"].Code == "KLEI"
        assert soils["zand"].VolumetricWeightAbovePhreaticLevel == 15.5
        assert soils["veen"].Code == "veen"  # gekoppeld aan geen: alleen de naam
        # De inhoud als dict werkt ook.
        data = json.loads(links.read_text(encoding="utf-8"))
        again = linked.cross_section(15.0).to_dstability(tmp_path / "again.stix", soil_links=data)
        assert self.soils_of_layers(again)["zand"].Code == "KLEI"

    def test_without_soil_links_only_names(self, section, tmp_path):
        # Een D-Stability-koppeling in het .nc telt niet meer: die is van LithoVox Stability.
        linked = self.linked_model(section, tmp_path)
        soils = self.soils_of_layers(linked.cross_section(15.0).to_dstability(tmp_path / "section.stix"))
        assert soils["zand"].Code == "zand" and soils["klei"].Code == "klei"

    def test_soil_links_of_another_model_or_unreadable(self, section, tmp_path):
        linked = self.linked_model(section, tmp_path)
        cut = linked.cross_section(15.0)
        other = self.soil_links(linked, tmp_path, model={"file": "ander.nc", "sha256": "00" * 32})
        with pytest.raises(ValueError, match="ander model"):
            cut.to_dstability(tmp_path / "s.stix", soil_links=other)
        with pytest.raises(ValueError, match="niet bij welk model"):
            cut.to_dstability(tmp_path / "s.stix", soil_links=self.soil_links(linked, tmp_path, model=None))
        with pytest.raises(ValueError, match="versie 2"):
            cut.to_xslope(tmp_path / "s.xlsx", soil_links=self.soil_links(linked, tmp_path, version=2))
        broken = tmp_path / "kapot.soil-links.json"
        broken.write_text("{", encoding="utf-8")
        with pytest.raises(ValueError, match="niet te lezen"):
            cut.to_dstability(tmp_path / "s.stix", soil_links=broken)

    def test_xslope_with_soil_links(self, section, tmp_path):
        linked = self.linked_model(section, tmp_path)
        links = self.soil_links(linked, tmp_path)
        path = linked.cross_section(15.0).to_xslope(tmp_path / "section.xlsx", soil_links=links)
        assert path.exists()

    def test_cli_takes_soil_links(self, section, tmp_path):
        linked = self.linked_model(section, tmp_path)
        links = self.soil_links(linked, tmp_path)
        out = tmp_path / "cli.stix"
        assert main(["doorsnede", str(linked.path), "15", str(out), "--koppelingen", str(links)]) == 0
        assert self.soils_of_layers(out)["zand"].Code == "KLEI"

    def test_unreadable_soil_parameters(self, section, tmp_path):
        with pytest.raises(ValueError, match="onbekend formaat"):
            self.linked_model(section, tmp_path, format="iets anders")
        with pytest.raises(ValueError, match="versie 2"):
            self.linked_model(section, tmp_path, version=2)

    def test_xslope_xlsx(self, section, tmp_path):
        workbook = openpyxl.load_workbook(section.to_xslope(tmp_path / "section.xlsx"))
        assert "polygon" in workbook.sheetnames
        styles = json.loads(
            (tmp_path / "section_styles.json").read_text(encoding="utf-8")
        )
        assert styles["materials"]

    def test_xslope_zip(self, section, tmp_path):
        path = section.to_xslope(tmp_path / "section.zip")
        assert sorted(zipfile.ZipFile(path).namelist()) == [
            "section.xlsx",
            "section_styles.json",
        ]

    def test_dxf(self, section, tmp_path):
        doc = ezdxf.readfile(section.to_dxf(tmp_path / "section.dxf"))
        layers = {layer.dxf.name for layer in doc.layers}
        assert {
            "INTERFACES",
            "OUTLINE",
            "SOIL_zand",
            "SOIL_klei",
            "SOIL_veen",
        } <= layers

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
        assert [layer.soil for layer in column.layers] == [
            "klei",
            "veen",
            "klei",
            "zand",
        ]
        assert column.top == pytest.approx(1.0)
        assert column.layers[-1].bottom == pytest.approx(-8.0)
        assert column.ground == pytest.approx(1.0)

    def test_locate(self, model):
        column = model.soil_column(15, 2.5)
        assert model.locate(column.x, column.y) == pytest.approx(
            (column.chainage, column.offset)
        )
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
