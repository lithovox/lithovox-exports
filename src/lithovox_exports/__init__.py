"""Exports van LithoVox-corridormodellen (.nc) naar reken- en GIS-software:
D-Stability, D-Geo Flow, XSLOPE, DXF en CSV per dwarsdoorsnede; GLB, 3D
Tiles en LAZ voor het hele model. Zie README.md en notebooks/voorbeelden.ipynb."""

from importlib.metadata import version

from .model import CrossSection, Model, SoilColumn, SoilLayer
from .profile import SoilPolygon

__all__ = ["Model", "CrossSection", "SoilColumn", "SoilLayer", "SoilPolygon"]
__version__ = version("lithovox-exports")
