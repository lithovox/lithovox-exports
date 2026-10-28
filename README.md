# lithovox-exports

Zet de voxelmodellen die LithoVox levert (corridormodellen, `.nc`) om naar
invoer voor je eigen software: D-Stability, D-Geo Flow, XSLOPE, DXF en CSV
per dwarsdoorsnede, en GLB, 3D Tiles en LAZ voor het hele model. Daarnaast
maakt het pakket figuren (kaart, doorsnede, grondkolom) en geeft het de
grondopbouw op elk punt als een virtuele boring.

Uitgebreide, uitgelegde voorbeelden staan in het notebook
[`notebooks/voorbeelden.ipynb`](notebooks/voorbeelden.ipynb).

## Installatie

```
pip install lithovox-exports
```

of, in een uv-project, `uv add lithovox-exports`. Python 3.12 of nieuwer.

## Snel aan de slag

```python
from lithovox_exports import Model

model = Model("model.nc")
print(model.name, model.length, model.crs, model.soils)

# De dwarsdoorsnede bij stationering 250 m, als D-Stability-model
doorsnede = model.cross_section(250)
doorsnede.to_dstability("km0.250.stix")
```

## Het model

Een corridormodel volgt een referentielijn: s is de stationering (m langs
de lijn), d de afstand tot de lijn (m, negatief buitenwaarts, positief
binnenwaarts) en z de hoogte.

```python
model.length            # lengte van de referentielijn (m)
model.offsets           # (buitenrand, binnenrand) in m tot de lijn
model.heights           # onder- en bovenkant van het grid
model.voxel_size        # (langs, dwars, hoog) in m
model.crs               # bijvoorbeeld "EPSG:28992"
model.vertical_datum    # bijvoorbeeld "NAP", of None als het onbekend is
model.soils             # namen van de grondsoorten, in de volgorde van hun codes
model.colors            # kleur per grondsoort
model.plot_map(chainages=[250, 500])   # kaart met de plekken van doorsneden
```

## Dwarsdoorsneden

Een dwarsdoorsnede ligt haaks op de referentielijn, bij de modelkolom die
het dichtst bij de gevraagde stationering ligt. De grenzen tussen
grondsoorten worden uit de voxels opgespoord en gladgemaakt tot polygonen
die hun punten precies met hun buren delen en geen gaten hebben (zoals
D-Stability vereist).

```python
doorsnede = model.cross_section(250)                # gladgemaakte grenzen
trap = model.cross_section(250, smoothing=0)        # de trap van de voxels

doorsnede.to_dstability("km0.250.stix")   # D-Stability
doorsnede.to_dgeoflow("km0.250.flox")     # D-Geo Flow
doorsnede.to_xslope("km0.250.zip")        # XSLOPE: .xlsx + kleuren, gezipt
doorsnede.to_xslope("km0.250.xlsx")       # of de .xlsx met *_styles.json ernaast
doorsnede.to_dxf("km0.250.dxf")           # DXF (GEO5, CAD)
doorsnede.to_csv("km0.250.csv")           # hoekpunten van de polygonen
doorsnede.plot()                          # figuur (matplotlib)
```

Om de 50 m langs het hele model:

```python
for doorsnede in model.cross_sections(every=50):
    doorsnede.to_dstability(f"{doorsnede.label}.stix")   # km0.000.stix, km0.050.stix, ...
```

### Eigen grondsoorten en parameters (D-Stability, XSLOPE)

Standaard hebben de grondsoorten alleen een naam. Geef een
D-Stability-bestand met je grondsoortenbibliotheek mee om die grondsoorten
en parameters te gebruiken. Een modelgrondsoort krijgt de
bibliotheekgrondsoort met dezelfde code of naam (hoofdletters, spaties,
`_` en `-` maken niet uit), of de grondsoort die je in `mapping` noemt
(naam van de modelgrondsoort → code in de bibliotheek; `None` voor geen):

```python
doorsnede.to_dstability(
    "km0.250.stix",
    soils="mijn_grondsoorten.stix",
    mapping={"hollandveen": "VEEN", "jonge_zeeklei_na": "KLEI"},
)
doorsnede.to_xslope("km0.250.zip", soils="mijn_grondsoorten.stix", mapping={"hollandveen": "VEEN"})
```

XSLOPE-materialen krijgen de volumegewichten en, bij Mohr-Coulomb, c en phi.
Waterspanningen, belastingen, mesh en rekeninstellingen vul je altijd zelf in.

## Grondopbouw op een punt

```python
kolom = model.soil_column(250, 10)          # stationering 250 m, 10 m binnenwaarts
kolom = model.soil_column_at(x, y)          # op wereldcoördinaten
for laag in kolom.layers:
    print(laag.top, laag.bottom, laag.soil)
kolom.plot(colors=model.colors)
```

## Het hele model

```python
model.to_glb("model.glb")      # 3D-mesh, bijvoorbeeld Blender of een 3D-viewer
model.to_tiles("tiles.zip")    # OGC 3D Tiles (3d/tileset.json), bijvoorbeeld Cesium, ArcGIS
model.to_laz("model.laz")      # puntenwolk, bijvoorbeeld QGIS, ArcGIS
```

Elk neemt `voxel_width` en `voxel_height` (m; standaard 1,0 en 0,2): het model
wordt getoond met voxels van ongeveer die grootte, die elk de grondsoort
krijgen die de meeste van hun modelvoxels hebben. Grotere voxels geven
kleinere bestanden. Het model wordt in brokken gelezen, dus ook grote
modellen passen in het geheugen.

## Coördinaten

- Doorsneden: x is de afstand tot de referentielijn (m, positief
  binnenwaarts), z de hoogte. `doorsnede.start` en `doorsnede.end` geven de
  wereldcoördinaten van de uiteinden.
- Het hele model en de grondkolommen: het coördinatenstelsel van het model
  (`model.crs`). De GLB is in lokale meters rond een oorsprong die in zijn
  extras staat. De LAZ heeft een punt per voxel met klasse 64 + de code van
  de grondsoort (`model.soils[code]`).
- 3D Tiles rekent met ellipsoïdische hoogtes. Bij een onbekend verticaal
  datum wordt de modelhoogte daarvoor genomen; het model ligt dan horizontaal
  goed maar kan verticaal afwijken (`"accurate": false` in tileset.json).

## Opdrachtregel

```
lithovox-exports info model.nc
lithovox-exports doorsnede model.nc 250 km0.250.stix --grondsoorten mijn_grondsoorten.stix
lithovox-exports model model.nc model.laz --breedte 2 --hoogte 0.5
```

Het formaat volgt uit de extensie: `.stix`, `.flox`, `.xlsx`, `.zip`
(XSLOPE), `.dxf`, `.csv` of `.png` voor een doorsnede; `.glb`, `.laz` of
`.zip` (3D Tiles) voor het hele model.

## Meer

`doorsnede.polygons` bevat de grondpolygonen zelf (`soil` is een index in
`model.soils`, `points` de omtrek als (x, z)), `model.metadata` alle
metadata van het model en `model.grid` het grid, als je zelf een export wilt
maken.

Vragen, of een formaat dat je mist: lithovox@pm.me, https://www.lithovox.com.
