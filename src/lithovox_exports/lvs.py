"""Een LithoVox-grondsoortenbestand (.lvs, LithoVox soils): voor elke
grondsoort van een model de D-Stability-grondsoort die eraan gekoppeld is,
met al zijn parameters. Het bestand is JSON:

    {
      "format": "lithovox-soils", "version": 1,
      "model": {"file": "dijk.nc", "name": "…", "project": "…",
                "created": "2026-11-02T09:36:17+00:00", "sha256": "…"},
      "soils": [
        {"soil": "hollandveen", "code": "VEEN", "dstability": {…}},
        {"soil": "basalton", "code": null, "dstability": null}
      ]
    }

`soil` is de naam van de modelgrondsoort, `code` de code van de gekoppelde
D-Stability-grondsoort (null: geen; de laag krijgt dan een grondsoort met
alleen die naam) en `dstability` die grondsoort zoals in soils.json van een
.stix. `model` zegt bij welk model (en welke versie ervan) het bestand hoort."""

import hashlib
import json
from pathlib import Path

FORMAT = "lithovox-soils"
VERSION = 1


def read_lvs(path: str | Path) -> dict:
    """De grondsoorten van een .lvs als bibliotheek, zoals
    dstability.read_stix_soils ze geeft, met de koppeling (`mapping`:
    modelgrondsoort → code, None voor geen) en het model (`model`) erbij.
    ValueError als het geen .lvs is dat dit pakket kan lezen."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"{path.name} is geen LithoVox-grondsoortenbestand (.lvs): {e}") from e
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path.name} is geen LithoVox-grondsoortenbestand (.lvs)")
    if data.get("version") != VERSION:
        raise ValueError(
            f"{path.name} heeft versie {data.get('version')} van het .lvs-formaat; "
            f"dit pakket leest versie {VERSION}"
        )
    soils: dict[str, dict] = {}
    mapping: dict[str, str | None] = {}
    for entry in data.get("soils") or []:
        code = entry.get("code") or None
        mapping[entry["soil"]] = code
        # Meerdere modelgrondsoorten kunnen dezelfde grondsoort hebben: die komt er één keer in.
        if code and entry.get("dstability") and code not in soils:
            soils[code] = entry["dstability"]
    return {"soils": list(soils.values()), "mapping": mapping, "model": data.get("model") or {}}


def sha256(path: str | Path) -> str:
    """De controlesom van een bestand, zoals een .lvs die van zijn model noemt."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
