# lithovox-exports

Zet de voxelmodellen die LithoVox levert (corridormodellen, `.nc`) om naar
invoer voor je eigen software: D-Stability, D-Geo Flow, DXF en CSV per
dwarsdoorsnede, en GLB, 3D Tiles en LAZ voor het hele model. Daarnaast
maakt het pakket figuren (kaart, doorsnede, grondkolom) en geeft het de
grondopbouw op elk punt als een virtuele boring.

De D-Stability- en D-Geo Flow-bestanden worden geschreven met
[GEOLib](https://github.com/Deltares/GEOLib) van Deltares (`d-geolib`, MIT).

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

# De dwarsdoorsnede bij metrering 250 m, als D-Stability-model
doorsnede = model.cross_section(250)
doorsnede.to_dstability("km0.250.stix")
```

## Het model

Een corridormodel volgt een referentielijn: s is de metrering (m langs
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
het dichtst bij de gevraagde metrering ligt. De grenzen tussen
grondsoorten worden uit de voxels opgespoord en gladgemaakt tot polygonen
die hun punten precies met hun buren delen en geen gaten hebben (zoals
D-Stability vereist).

```python
doorsnede = model.cross_section(250)                # gladgemaakte grenzen
trap = model.cross_section(250, smoothing=0)        # de trap van de voxels

doorsnede.to_dstability("km0.250.stix")   # D-Stability
doorsnede.to_dgeoflow("km0.250.flox")     # D-Geo Flow
doorsnede.to_dxf("km0.250.dxf")           # DXF (GEO5, CAD)
doorsnede.to_csv("km0.250.csv")           # hoekpunten van de polygonen
doorsnede.plot()                          # figuur (matplotlib)
```

Om de 50 m langs het hele model:

```python
for doorsnede in model.cross_sections(every=50):
    doorsnede.to_dstability(f"{doorsnede.label}.stix")   # km0.000.stix, km0.050.stix, ...
```

### Grondsoorten met parameters (D-Stability, D-Geo Flow)

Zonder meer krijgt elke grondsoort alleen zijn naam (en kleur), met de
standaardparameters van het programma. Je eigen parameters komen uit een eigen
bestand van hetzelfde programma (een `.stix` voor D-Stability, een `.flox` voor
D-Geo Flow) met een koppeling: per grondsoort van het model de code van zijn
grondsoort in dat bestand.

```python
koppeling = {"klei": "K_Ozw", "zand": "Z_Pleist", "veen": None}
doorsnede.to_dstability("km0.250.stix", soils="grondsoorten.stix", mapping=koppeling)
doorsnede.to_dgeoflow("km0.250.flox", soils="grondsoorten.flox", mapping={"zand": "Zand"})
```

Een gekoppelde grondsoort komt met al zijn parameters in de export (bij D-Geo
Flow de doorlatendheden); een grondsoort die de koppeling niet noemt, of aan
`None` koppelt, krijgt alleen zijn naam. Alleen de grondsoorten die de
doorsnede gebruikt gaan erin. De koppeling mag ook een JSON-bestand met
hetzelfde object zijn. Een bestand van het andere programma, een naam die geen
grondsoort van het model is of een code die het bestand niet heeft, geeft een
foutmelding. Op de opdrachtregel: `lithovox-exports doorsnede dijk.nc 250
km0.250.stix --grondsoorten grondsoorten.stix --koppeling koppeling.json`.
LithoVox levert geen koppelingen mee: ze zijn van de gebruiker.

Een model dat LithoVox tussen 8 en 10 november 2026 maakte, kan zelf parameters
van gekoppelde grondsoorten hebben (formaatversie 3, ondertekend met de voxels;
het laatst alleen D-Settlement). Nieuwe modellen hebben ze niet meer: LithoVox
koppelt geen grondsoorten meer. Oudere bestanden lees je zo:

```python
model.soil_parameter_files                  # de voorbeeldbestanden: programma, bestand, ingelezen
model.soil_parameters("d_settlement")       # {"klei": {"code": ..., "parameters": [[sleutel, waarde], ...]}, ...}
```

Waterspanningen, belastingen, mesh en rekeninstellingen vul je altijd zelf in.

### Parameterzones

Een model kan parameterzones hebben: in LithoVox zet je bij een sondering of
boring eigen volumieke gewichten voor een grondsoort, die gelden in een
cirkel rond dat object. Het model bewaart ze naast de grondsoorten (niet in de
voxels, en niet ondertekend).

```python
model.parameter_areas                       # object, midden, straal, grondsoort, gamma_dry, gamma_sat
doorsnede = model.cross_section(100)
doorsnede.parameter_areas                   # de zones die deze doorsnede raken (leeg: geen)
doorsnede = model.cross_section(100, use_parameter_area_values=True)
doorsnede.soils                             # o.a. "Peat (CPT000000074503)"
```

Met `use_parameter_area_values=True` krijgen de voxels van de grondsoort van
een zone die in de cirkel liggen een eigen grondsoort, "grondsoort (object)",
met de kleur van de grondsoort van het model; ligt een voxel in meer zones,
dan telt de zone met het dichtstbijzijnde midden. Een doorsnede die geen zone
raakt, blijft gelijk. Die grondsoort komt in de D-Stability-export met de
volumieke gewichten van de zone (boven en onder de freatische lijn) en, met
een eigen `.stix` en koppeling, met de overige parameters van de gekoppelde
grondsoort onder een eigen code ("KLEI (CPT1)"). D-Geo Flow, DXF en CSV krijgen
de lagen apart; bij D-Geo Flow met de doorlatendheden van de grondsoort van het
model. Op de opdrachtregel: `--parameterzones`; `lithovox-exports info` toont
de zones.

## Grondopbouw op een punt

```python
kolom = model.soil_column(250, 10)          # metrering 250 m, 10 m binnenwaarts
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
lithovox-exports doorsnede model.nc 250 km0.250.stix
lithovox-exports model model.nc model.laz --breedte 2 --hoogte 0.5
```

Het formaat volgt uit de extensie: `.stix`, `.flox`, `.dxf`, `.csv` of `.png`
voor een doorsnede; `.glb`, `.laz` of `.zip` (3D Tiles) voor het hele model.

## Wijzigingen

- **0.5.0**:
  - D-Geo Flow (`to_dgeoflow`, `.flox`) is terug.
  - De grondsoortparameters komen uit een eigen `.stix` of `.flox` met een
    koppeling (`soils`, `mapping`; `--grondsoorten`, `--koppeling`), in plaats
    van uit het koppelbestand van LithoVox Stability (`soil_links`,
    `--koppelingen` zijn vervallen). Alleen de gebruikte grondsoorten gaan in
    de export, niet de hele bibliotheek.
  - De XSLOPE-export (`to_xslope`, `.xlsx`/`.zip` op de opdrachtregel) is
    vervallen.
  - D-Stability-bestanden worden geschreven met GEOLib (`d-geolib`) in plaats
    van lithovox-stability; zonder koppeling zijn ze inhoudelijk gelijk.
  - Nieuwe modellen van LithoVox hebben geen grondsoortparameters meer
    (`soil_parameters`); `model.soil_parameters()` leest ze nog uit oudere
    bestanden.

## Meer

`doorsnede.polygons` bevat de grondpolygonen zelf (`soil` is een index in
`model.soils`, `points` de omtrek als (x, z)), `model.metadata` alle
metadata van het model en `model.grid` het grid, als je zelf een export wilt
maken.

Vragen, of een formaat dat je mist: lithovox@pm.me, https://www.lithovox.com.
