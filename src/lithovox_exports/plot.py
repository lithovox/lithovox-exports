"""Figuren met matplotlib: de kaart van een model, een dwarsdoorsnede en een
grondkolom. Elke functie tekent in de gegeven `ax` (of een nieuwe figuur)
en geeft die `ax` terug, zodat je hem verder kunt aanpassen."""

from typing import TYPE_CHECKING

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from .glb import DEFAULT_COLOR
from .grid import chainage_label

if TYPE_CHECKING:
    from .model import CrossSection, Model, SoilColumn

# Rond getal voor de afstand tussen de metreringslabels op de kaart.
NICE_STEPS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
# Hooguit zoveel cellen langs en dwars op de lijn voor het maaiveld op de kaart.
MAP_CELLS = (600, 120)


def _axes(ax, figsize: tuple[float, float]):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, layout="constrained")
    return ax


def _legend(ax, soils: list[str], colors: dict[str, str]) -> None:
    handles = [Patch(facecolor=colors.get(name, DEFAULT_COLOR), edgecolor="0.3", label=name) for name in soils]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)


def plot_section(section: "CrossSection", ax=None, ground: bool = True, legend: bool = True, edges: bool = True):
    """De grondpolygonen van een dwarsdoorsnede, in de kleuren van het model,
    met (`ground`) het maaiveld uit de hoogtedata als stippellijn."""
    ax = _axes(ax, (14, 5))
    names = section.slice.soils
    colors = section.slice.colors
    polygons = PolyCollection(
        [p.points for p in section.polygons],
        facecolors=[colors.get(names[p.soil], DEFAULT_COLOR) for p in section.polygons],
        edgecolors="0.25" if edges else "face",
        linewidths=0.5 if edges else 0.0,
    )
    ax.add_collection(polygons)
    if ground and np.isfinite(section.slice.ground).any():
        ax.plot(section.slice.u, section.slice.ground, color="k", linestyle="--", linewidth=1, label="maaiveld")
    ax.autoscale_view()
    ax.set_xlabel("afstand tot de referentielijn [m] (positief = binnenwaarts)")
    ax.set_ylabel(f"hoogte [{section.crs.height_unit}]")
    ax.set_title(f"{section.label} (metrering {section.slice.chainage:g} m)")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    if legend:
        _legend(ax, section.soils, colors)
    return ax


def plot_column(column: "SoilColumn", ax=None, colors: dict[str, str] | None = None):
    """Een grondkolom als staafje met de lagen, met hun grondsoort ernaast."""
    ax = _axes(ax, (3.5, 7))
    colors = colors or {}
    for layer in column.layers:
        ax.bar(0, layer.top - layer.bottom, bottom=layer.bottom, width=1, color=colors.get(layer.soil, DEFAULT_COLOR), edgecolor="0.25", linewidth=0.5)
        ax.text(0.6, (layer.top + layer.bottom) / 2, layer.soil, va="center", fontsize=8)
    if np.isfinite(column.ground):
        ax.axhline(column.ground, color="k", linestyle="--", linewidth=1)
    if column.layers:
        # Met wat ruimte boven en onder; bij gedeelde assen (sharey) de vereniging met de andere kolommen.
        low, high = ax.get_ylim()
        top = max(column.top, column.ground) if np.isfinite(column.ground) else column.top
        ax.set_ylim(min(low, column.layers[-1].bottom - 0.5), max(high, top + 0.5))
    ax.set_xlim(-0.5, 3)
    ax.set_xticks([])
    ax.set_ylabel(f"hoogte [{column.height_unit}]")
    ax.set_title(f"s = {column.chainage:g} m, d = {column.offset:g} m", fontsize=9)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    return ax


def _nice_step(length: float) -> float:
    return next((step for step in NICE_STEPS if length / step <= 10), NICE_STEPS[-1])


def plot_map(model: "Model", ax=None, chainages: list[float] | None = None, ground: bool = True, step: float | None = None):
    """Bovenaanzicht van het model: de omtrek, de referentielijn met
    metreringen om de `step` m, met (`ground`) het maaiveld in kleur, en de
    dwarsdoorsneden op `chainages`."""
    ax = _axes(ax, (10, 8))
    grid = model.grid
    ns, nd, _ = grid.shape
    if ground:
        a, b = max(1, ns // MAP_CELLS[0]), max(1, nd // MAP_CELLS[1])
        rows, cols = np.arange(0, ns + 1, a), np.arange(0, nd + 1, b)
        # Altijd tot en met de laatste rand, zodat het maaiveld het hele model bedekt.
        rows, cols = np.unique(np.r_[rows, ns]), np.unique(np.r_[cols, nd])
        corners = grid.corner_xy(0, ns)[rows][:, cols]
        with h5py.File(model.path, "r") as f:
            level = np.asarray(f["ground"][::a, ::b], dtype=np.float64)[: len(rows) - 1, : len(cols) - 1]
        mesh = ax.pcolormesh(corners[..., 0], corners[..., 1], level, cmap="terrain", shading="flat")
        plt.colorbar(mesh, ax=ax, label=f"maaiveld [{grid.crs.height_unit}]", shrink=0.7)
    # De omtrek: de buitenrand, het eind, de binnenrand terug en het begin.
    d = grid.offset_edges()
    outer = grid.edge_station + d[0] * grid.edge_normal
    inner = grid.edge_station + d[-1] * grid.edge_normal
    outline = np.r_[outer, inner[::-1], outer[:1]]
    ax.plot(outline[:, 0], outline[:, 1], color="0.3", linewidth=1)
    ax.plot(grid.chainage_line[:, 0], grid.chainage_line[:, 1], color="k", linewidth=1.5, label="referentielijn")
    # Metreringen langs de lijn, aan de buitenkant van het model.
    step = step or _nice_step(model.length)
    marks = np.arange(0, model.length + 1e-9, step)
    x = np.interp(marks, grid.chainage, grid.station[:, 0])
    y = np.interp(marks, grid.chainage, grid.station[:, 1])
    nx = np.interp(marks, grid.chainage, grid.normal[:, 0])
    ny = np.interp(marks, grid.chainage, grid.normal[:, 1])
    ax.plot(x, y, "k|", markersize=8)
    label_at = d[0] - 0.05 * (d[-1] - d[0])
    for s, px, py, qx, qy in zip(marks, x, y, nx, ny):
        ax.annotate(f"{s:g}", (px + label_at * qx, py + label_at * qy), ha="center", va="center", fontsize=7)
    for chainage in chainages or []:
        # Dezelfde kolom als Model.cross_section kiest.
        i = int(np.argmin(np.abs(grid.chainage - chainage)))
        start = grid.station[i] + d[0] * grid.normal[i]
        end = grid.station[i] + d[-1] * grid.normal[i]
        ax.plot([start[0], end[0]], [start[1], end[1]], color="crimson", linewidth=2)
        ax.annotate(chainage_label(chainage), end, color="crimson", fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_aspect("equal")
    ax.set_xlabel(f"x [m] ({grid.crs.name})")
    ax.set_ylabel(f"y [m] ({grid.crs.name})")
    ax.set_title(f"{model.project} – {model.name}" if model.project else model.name)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    return ax
