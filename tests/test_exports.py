import csv
import json
import zipfile
from pathlib import Path

import ezdxf
import h5py
import laspy
import pytest
from geolib.models.dstability import DStabilityModel

from lithovox_exports import CrossSection, Model
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
        """Het testmodel met grondsoortparameters zoals LithoVox ze schrijft
        (het attribuut van het .nc): zand en klei gekoppeld aan de
        D-Stability-grondsoort "KLEI", veen aan geen; en D-Settlement."""
        from conftest import write_model

        clay = {"Code": "KLEI", "Name": "Klei, siltig", "VolumetricWeightAbovePhreaticLevel": 15.5}
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

    def test_unreadable_soil_parameters(self, section, tmp_path):
        with pytest.raises(ValueError, match="onbekend formaat"):
            self.linked_model(section, tmp_path, format="iets anders")
        with pytest.raises(ValueError, match="versie 2"):
            self.linked_model(section, tmp_path, version=2)

    def test_the_nc_soil_parameters_do_not_go_into_an_export(self, section, tmp_path):
        # De D-Stability-koppeling in het .nc telt niet voor de export: alleen een eigen bestand.
        linked = self.linked_model(section, tmp_path)
        soils = soils_of_layers(linked.cross_section(15.0).to_dstability(tmp_path / "section.stix"))
        assert soils["zand"].Code == "zand" and soils["klei"].Code == "klei"


def soils_of_layers(path, program=None) -> dict:
    """Per laag (zijn label: de modelgrondsoort) de grondsoort die hij heeft."""
    model = (program or DStabilityModel)()
    model.parse(path)
    ds = model.datastructure
    by_id = {s.Id: s for s in ds.soils.Soils}
    soil_of = {sl.LayerId: by_id[sl.SoilId] for sl in ds.soillayers[0].SoilLayers}
    return {layer.Label: soil_of[layer.Id] for layer in ds.geometries[0].Layers}


@pytest.fixture
def stix_library(tmp_path) -> Path:
    """Een eigen D-Stability-bestand met drie grondsoorten: KLEI (eigen
    volumegewicht), veen (de code van een modelgrondsoort) en EXTRA."""
    from geolib.soils.soil import Soil

    stix = DStabilityModel()
    stix.datastructure.soils.Soils = []
    for code, name in (("KLEI", "Klei, siltig"), ("veen", "Veen, bibliotheek"), ("EXTRA", "Niet gebruikt")):
        stix.add_soil(Soil(name=name, code=code))
    for soil in stix.datastructure.soils.Soils:
        soil.VolumetricWeightAbovePhreaticLevel = {"KLEI": 15.5, "veen": 10.5, "EXTRA": 20.0}[soil.Code]
    path = tmp_path / "bibliotheek.stix"
    stix.serialize(path)
    return path


@pytest.fixture
def flox_library(tmp_path) -> Path:
    """Een eigen D-Geo Flow-bestand: ZAND (doorlatend) en KLEI (slecht doorlatend)."""
    from geolib.models.dgeoflow import DGeoFlowModel
    from geolib.soils.soil import Soil

    flox = DGeoFlowModel()
    flox.datastructure.soils.Soils = []
    for code in ("ZAND", "KLEI"):
        flox.add_soil(Soil(name=code.title(), code=code))
    for soil in flox.datastructure.soils.Soils:
        soil.HorizontalPermeability, soil.VerticalPermeability = {"ZAND": (5.0, 2.5), "KLEI": (1e-4, 5e-5)}[soil.Code]
    path = tmp_path / "bibliotheek.flox"
    flox.serialize(path)
    return path


class TestSoilLibrary:
    def test_dstability_with_own_soils(self, section, tmp_path, stix_library):
        mapping = {"zand": "KLEI", "klei": "KLEI", "veen": None}
        path = section.to_dstability(tmp_path / "s.stix", soils=stix_library, mapping=mapping)
        soils = soils_of_layers(path)
        assert soils["zand"].Code == soils["klei"].Code == "KLEI"
        assert soils["zand"].VolumetricWeightAbovePhreaticLevel == 15.5
        assert soils["veen"].Code == "veen"  # gekoppeld aan geen: alleen de naam
        # Alleen de gebruikte grondsoorten, niet de hele bibliotheek.
        stix = DStabilityModel()
        stix.parse(path)
        assert sorted(s.Code for s in stix.datastructure.soils.Soils) == ["KLEI", "veen"]

    def test_soils_not_named_get_their_name(self, section, tmp_path, stix_library):
        soils = soils_of_layers(section.to_dstability(tmp_path / "s.stix", soils=stix_library, mapping={"klei": "KLEI"}))
        assert soils["klei"].Code == "KLEI" and soils["zand"].Code == "zand" and soils["veen"].Code == "veen"

    def test_a_code_taken_by_a_linked_soil(self, section, tmp_path, stix_library):
        # klei krijgt de bibliotheekgrondsoort "veen"; de modelgrondsoort veen (niet gekoppeld) wijkt uit.
        soils = soils_of_layers(section.to_dstability(tmp_path / "s.stix", soils=stix_library, mapping={"klei": "veen"}))
        assert soils["klei"].Code == "veen" and soils["klei"].VolumetricWeightAbovePhreaticLevel == 10.5
        assert soils["veen"].Code == "veen (2)"

    def test_mapping_from_a_json_file(self, section, tmp_path, stix_library):
        mapping = tmp_path / "koppeling.json"
        mapping.write_text(json.dumps({"zand": "KLEI", "veen": None}), encoding="utf-8")
        soils = soils_of_layers(section.to_dstability(tmp_path / "s.stix", soils=stix_library, mapping=mapping))
        assert soils["zand"].Code == "KLEI"

    def test_dgeoflow(self, section, tmp_path):
        from geolib.models.dgeoflow import DGeoFlowModel

        path = section.to_dgeoflow(tmp_path / "section.flox")
        flox = DGeoFlowModel()
        flox.parse(path)
        assert {s.Code for s in flox.datastructure.soils.Soils} == {"zand", "klei", "veen"}
        assert len(flox.datastructure.geometries[0].Layers) == len(section.polygons)
        assert flox.datastructure.scenarios[0].Label == "km0.015"
        assert len(flox.datastructure.soilvisualizations.SoilVisualizations) == 3

    def test_dgeoflow_with_own_soils(self, section, tmp_path, flox_library):
        from geolib.models.dgeoflow import DGeoFlowModel

        path = section.to_dgeoflow(tmp_path / "s.flox", soils=flox_library, mapping={"zand": "ZAND", "klei": "KLEI"})
        soils = soils_of_layers(path, DGeoFlowModel)
        assert soils["zand"].HorizontalPermeability == 5.0 and soils["zand"].VerticalPermeability == 2.5
        assert soils["klei"].HorizontalPermeability == 1e-4
        assert soils["veen"].Code == "veen"

    def test_only_the_same_program(self, section, tmp_path, stix_library, flox_library):
        with pytest.raises(ValueError, match="uit een .stix-bestand"):
            section.to_dstability(tmp_path / "s.stix", soils=flox_library, mapping={"zand": "ZAND"})
        with pytest.raises(ValueError, match="uit een .flox-bestand"):
            section.to_dgeoflow(tmp_path / "s.flox", soils=stix_library, mapping={"zand": "KLEI"})

    def test_mistakes_are_errors(self, section, tmp_path, stix_library):
        cut = lambda **kw: section.to_dstability(tmp_path / "s.stix", **kw)  # noqa: E731
        with pytest.raises(ValueError, match="allebei"):
            cut(soils=stix_library)
        with pytest.raises(ValueError, match="allebei"):
            cut(mapping={"zand": "KLEI"})
        with pytest.raises(ValueError, match="geen grondsoort van het model"):
            cut(soils=stix_library, mapping={"leem": "KLEI"})
        with pytest.raises(ValueError, match="geen grondsoort met de code ONBEKEND"):
            cut(soils=stix_library, mapping={"zand": "ONBEKEND"})
        with pytest.raises(ValueError, match="verwacht de code"):
            cut(soils=stix_library, mapping={"zand": 5})
        array = tmp_path / "lijst.json"
        array.write_text("[1]", encoding="utf-8")
        with pytest.raises(ValueError, match="geen object"):
            cut(soils=stix_library, mapping=array)
        broken = tmp_path / "kapot.json"
        broken.write_text("{", encoding="utf-8")
        with pytest.raises(ValueError, match="niet te lezen"):
            cut(soils=stix_library, mapping=broken)
        garbage = tmp_path / "kapot.stix"
        garbage.write_bytes(b"geen zip")
        with pytest.raises(ValueError, match="geen D-Stability-bestand"):
            cut(soils=garbage, mapping={"zand": "KLEI"})
        with pytest.raises(ValueError, match="niet te lezen"):
            cut(soils=tmp_path / "bestaat-niet.stix", mapping={"zand": "KLEI"})

    def test_cli_takes_soils_and_mapping(self, model_path, tmp_path, stix_library, flox_library):
        mapping = tmp_path / "koppeling.json"
        mapping.write_text(json.dumps({"zand": "KLEI"}), encoding="utf-8")
        out = tmp_path / "cli.stix"
        assert main(["doorsnede", str(model_path), "15", str(out), "--grondsoorten", str(stix_library),
                     "--koppeling", str(mapping)]) == 0
        assert soils_of_layers(out)["zand"].Code == "KLEI"
        mapping.write_text(json.dumps({"zand": "ZAND"}), encoding="utf-8")
        assert main(["doorsnede", str(model_path), "15", str(tmp_path / "cli.flox"), "--grondsoorten",
                     str(flox_library), "--koppeling", str(mapping)]) == 0
        # Een verkeerd bestand is een foutmelding (1), geen crash.
        assert main(["doorsnede", str(model_path), "15", str(out), "--grondsoorten", str(flox_library),
                     "--koppeling", str(mapping)]) == 1
        assert main(["doorsnede", str(model_path), "15", str(tmp_path / "s.dxf"), "--koppeling", str(mapping)]) == 1


class TestExports:
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

    @pytest.mark.parametrize("suffix", [".xlsx", ".zip"])
    def test_no_xslope_any_more(self, model_path, tmp_path, suffix):
        # XSLOPE is vervallen in 0.5.0; .zip blijft alleen voor 3D Tiles van het hele model.
        with pytest.raises(SystemExit):
            main(["doorsnede", str(model_path), "15", str(tmp_path / f"s{suffix}")])
        assert not hasattr(CrossSection, "to_xslope")


class TestParameterAreas:
    """Het testmodel loopt west-oost; binnenwaarts is zuid (y neemt af). Een
    zone rond (100010, 399997) (3 m binnenwaarts) met straal 2 raakt de
    doorsneden op 8 tot 12 m en de afstanden 1 tot 5 m binnenwaarts (veen zit
    op 0 tot 5 m binnenwaarts, tussen -2 en -1 m hoog)."""

    AREA = {"kind": "cpt", "object": "CPT1", "x": 100010.0, "y": 399997.0, "radius": 2.0,
            "soil": "Veen", "soilClass": "veen", "gammaDry": 10.5, "gammaSat": 11.5}

    @pytest.fixture
    def zoned(self, tmp_path) -> Model:
        from conftest import parameter_areas, write_model

        return Model(write_model(tmp_path / "zoned.nc", areas=parameter_areas(self.AREA)))

    def test_read(self, zoned, model):
        assert model.parameter_areas == []
        (area,) = zoned.parameter_areas
        assert area.name == "veen (CPT1)" and area.radius == 2.0 and area.gamma_sat == 11.5

    def test_unreadable(self, tmp_path):
        from conftest import parameter_areas, write_model

        with pytest.raises(ValueError, match="geen grondsoort van het model"):
            Model(write_model(tmp_path / "a.nc", areas=parameter_areas({**self.AREA, "soilClass": "rots"})))
        with pytest.raises(ValueError, match="radius"):
            Model(write_model(tmp_path / "b.nc", areas=parameter_areas({**self.AREA, "radius": 0})))
        with pytest.raises(ValueError, match="versie 2"):
            Model(write_model(tmp_path / "c.nc", areas={"version": 2, "points": []}))

    def test_a_section_far_away_is_the_same(self, zoned, model):
        far = zoned.cross_section(25.0, use_parameter_area_values=True)
        assert far.parameter_areas == [] and far.soils == ["klei", "veen", "zand"]
        assert far.polygons == model.cross_section(25.0).polygons

    def test_a_section_through_an_area(self, zoned):
        plain = zoned.cross_section(10.0)
        assert [a.object for a in plain.parameter_areas] == ["CPT1"]
        assert "veen (CPT1)" not in plain.soils
        section = zoned.cross_section(10.0, use_parameter_area_values=True)
        assert "veen (CPT1)" in section.soils
        assert section.slice.colors["veen (CPT1)"] == section.slice.colors["veen"]
        # Alleen de voxels van veen in de cirkel (afstand 1 tot 5 m, minus de rand): de rest blijft veen.
        columns = {j for j in range(len(section.slice.u)) if (section.slice.labels[j] >= 3).any()}
        assert columns and all(1 <= section.slice.u[j] <= 5 for j in columns)
        assert section.slice.model_soils == ["zand", "klei", "veen"]

    def test_the_nearest_area_wins(self, tmp_path):
        from conftest import parameter_areas, write_model

        near = {**self.AREA, "object": "CPT2", "y": 399999.0, "gammaDry": 12.0, "gammaSat": 13.0}
        model = Model(write_model(tmp_path / "two.nc", areas=parameter_areas(self.AREA, near)))
        section = model.cross_section(10.0, use_parameter_area_values=True)
        assert {"veen (CPT1)", "veen (CPT2)"} <= set(section.slice.soils)
        # Het midden van CPT2 ligt op 1 m binnenwaarts: daar hoort de voxel bij CPT2.
        j = int(abs(section.slice.u - 1.25).argmin())
        k = int(abs(section.slice.z + 1.5).argmin())
        assert section.slice.soils[section.slice.labels[j, k]] == "veen (CPT2)"

    def test_dstability(self, zoned, tmp_path):
        section = zoned.cross_section(10.0, use_parameter_area_values=True)
        soils = soils_of_layers(section.to_dstability(tmp_path / "zone.stix"))
        zone = soils["veen (CPT1)"]
        assert zone.Code == "veen (CPT1)"
        assert zone.VolumetricWeightAbovePhreaticLevel == 10.5 and zone.VolumetricWeightBelowPhreaticLevel == 11.5
        # Zonder de waarden is er geen zone.
        plain = soils_of_layers(zoned.cross_section(10.0).to_dstability(tmp_path / "plain.stix"))
        assert set(plain) == {"zand", "klei", "veen"}

    def test_dstability_linked(self, zoned, tmp_path, stix_library):
        section = zoned.cross_section(10.0, use_parameter_area_values=True)
        mapping = {"veen": "KLEI"}
        soils = soils_of_layers(section.to_dstability(tmp_path / "zone.stix", stix_library, mapping))
        zone = soils["veen (CPT1)"]
        assert zone.Code == "KLEI (CPT1)"
        assert zone.VolumetricWeightAbovePhreaticLevel == 10.5 and zone.VolumetricWeightBelowPhreaticLevel == 11.5
        # De grondsoort uit het bestand blijft heel waar de zone niet komt.
        assert soils["klei"].Code == "klei"

    def test_dgeoflow_and_others(self, zoned, tmp_path):
        section = zoned.cross_section(10.0, use_parameter_area_values=True)
        assert section.to_dgeoflow(tmp_path / "zone.flox").exists()
        assert section.to_dxf(tmp_path / "zone.dxf").exists()
        assert section.to_csv(tmp_path / "zone.csv").exists()
        assert section.plot() is not None

    def test_cli(self, zoned, tmp_path, capsys):
        assert main(["info", str(zoned.path)]) == 0
        assert "veen (CPT1)" in capsys.readouterr().out
        out = tmp_path / "zone.stix"
        assert main(["doorsnede", str(zoned.path), "10", str(out), "--parameterzones"]) == 0
        assert "veen (CPT1)" in {s.Code for s in _stix_soils(out)}


def _stix_soils(path):
    stix = DStabilityModel()
    stix.parse(path)
    return stix.datastructure.soils.Soils
