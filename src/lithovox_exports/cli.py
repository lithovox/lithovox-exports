"""De opdrachtregel: `lithovox-exports info|doorsnede|model ...`. Het formaat
volgt uit de extensie van het uitvoerbestand."""

import argparse
import sys
from pathlib import Path

from .model import Model
from .profile import SMOOTHING

SECTION_FORMATS = [".stix", ".flox", ".xlsx", ".zip", ".dxf", ".csv", ".png"]
MODEL_FORMATS = [".glb", ".laz", ".zip"]


def _info(model: Model) -> None:
    ds, dd, dz = model.voxel_size
    low, high = model.offsets
    bottom, top = model.heights
    print(f"project         {model.project}")
    print(f"naam            {model.name}")
    print(f"gemaakt         {model.created}")
    print(f"lengte          {model.length:.1f} m")
    print(f"breedte         {low:g} tot {high:g} m")
    print(f"hoogte          {bottom:g} tot {top:g} {model.grid.crs.height_unit}")
    print(f"voxels          {' x '.join(map(str, model.shape))} ({ds:g} x {dd:g} x {dz:g} m)")
    print(f"crs             {model.crs}, verticaal: {model.vertical_datum or 'onbekend'}")
    print(f"grondsoorten    {', '.join(model.soils)}")


def _section(model: Model, args: argparse.Namespace) -> Path:
    section = model.cross_section(args.metrering, args.smoothing)
    path, suffix = args.uitvoer, args.uitvoer.suffix.lower()
    if suffix == ".stix":
        return section.to_dstability(path, args.grondsoorten)
    if suffix == ".flox":
        return section.to_dgeoflow(path)
    if suffix in (".xlsx", ".zip"):
        return section.to_xslope(path, args.grondsoorten)
    if suffix == ".dxf":
        return section.to_dxf(path)
    if suffix == ".csv":
        return section.to_csv(path)
    ax = section.plot()
    ax.figure.savefig(path, dpi=150)
    return path


def _whole(model: Model, args: argparse.Namespace) -> Path:
    suffix = args.uitvoer.suffix.lower()
    if suffix == ".glb":
        return model.to_glb(args.uitvoer, args.breedte, args.hoogte)
    if suffix == ".laz":
        return model.to_laz(args.uitvoer, args.breedte, args.hoogte)
    return model.to_tiles(args.uitvoer, args.breedte, args.hoogte)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lithovox-exports", description="Exports van LithoVox-corridormodellen (.nc)")
    commands = parser.add_subparsers(dest="opdracht", required=True)

    info = commands.add_parser("info", help="eigenschappen van het model")
    info.add_argument("model", type=Path)

    section = commands.add_parser("doorsnede", help="een dwarsdoorsnede: " + ", ".join(SECTION_FORMATS))
    section.add_argument("model", type=Path)
    section.add_argument("metrering", type=float, help="m langs de referentielijn")
    section.add_argument("uitvoer", type=Path)
    section.add_argument("--smoothing", type=float, default=SMOOTHING, help=f"in voxels (standaard {SMOOTHING})")
    section.add_argument("--grondsoorten", type=Path, help=".stix met grondsoorten en parameters")

    whole = commands.add_parser("model", help="het hele model: .glb, .laz, .zip (3D Tiles)")
    whole.add_argument("model", type=Path)
    whole.add_argument("uitvoer", type=Path)
    whole.add_argument("--breedte", type=float, default=1.0, help="voxelbreedte in m (standaard 1.0)")
    whole.add_argument("--hoogte", type=float, default=0.2, help="voxelhoogte in m (standaard 0.2)")

    args = parser.parse_args(argv)
    try:
        model = Model(args.model)
        if args.opdracht == "info":
            _info(model)
            return 0
        formats = SECTION_FORMATS if args.opdracht == "doorsnede" else MODEL_FORMATS
        if args.uitvoer.suffix.lower() not in formats:
            parser.error(f"onbekend formaat {args.uitvoer.suffix!r}, kies uit {', '.join(formats)}")
        path = _section(model, args) if args.opdracht == "doorsnede" else _whole(model, args)
    except (OSError, ValueError) as e:
        print(f"fout: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
