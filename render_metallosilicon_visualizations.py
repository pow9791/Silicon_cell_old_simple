#!/usr/bin/env python3
"""Render visualizations for the generated metallosilicon cell."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
VIS_DIR = OUTPUT_DIR / "visualizations"
SILICON_AMINO_DIR = Path(
    os.environ.get("SILICON_AMINO_DIR", ROOT.parent / "silicon_amino" / "sims" / "output")
)

COLORS = {
    "membrane": "#68727f",
    "scaffold": "#d97706",
    "redox_lithosome": "#b91c1c",
    "polymer_forge": "#0f766e",
    "template_lattice": "#1d4ed8",
    "Si": "#64748b",
    "N": "#2563eb",
    "S": "#ca8a04",
    "P": "#7c3aed",
    "H": "#cbd5e1",
    "B": "#db2777",
    "Al": "#14b8a6",
    "F": "#84cc16",
    "Fe": "#991b1b",
    "Ni": "#166534",
    "Ti": "#0891b2",
    "Mo": "#6d28d9",
}


def load_json(path: Path):
    return json.loads(path.read_text())


def project(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(points, dtype=float)
    if arr.shape[1] == 3:
        return arr[:, [0, 2]]
    return arr


def infer_bonds(atom_types: Sequence[str], positions: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    radii = {
        "H": 0.31,
        "B": 0.85,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "Al": 1.21,
        "Si": 1.11,
        "P": 1.07,
        "S": 1.05,
        "Fe": 1.16,
        "Ni": 1.10,
        "Ti": 1.36,
        "Mo": 1.39,
    }
    pts = np.array(positions, dtype=float)
    bonds: List[Tuple[int, int]] = []
    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            threshold = 1.25 * (radii.get(atom_types[i], 1.0) + radii.get(atom_types[j], 1.0))
            if np.linalg.norm(pts[i] - pts[j]) <= threshold:
                bonds.append((i, j))
    return bonds


def plot_whole_cell(cell: Dict[str, object], path: Path) -> None:
    membrane = np.array(cell["envelope"]["membrane_coordinates_nm"], dtype=float)
    scaffold = np.array(cell["genetic_scaffold"]["coarse_grained_coordinates_nm"], dtype=float)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(membrane[:, 0], membrane[:, 1], membrane[:, 2], s=8, alpha=0.24, c=COLORS["membrane"])
    ax.plot(scaffold[:, 0], scaffold[:, 1], scaffold[:, 2], linewidth=2.0, color=COLORS["scaffold"])

    for organelle in cell["organelles"]:
        shell = np.array(organelle["coarse_grained_shell_coordinates_nm"], dtype=float)
        color = COLORS.get(organelle["organelle_name"], "#334155")
        center = organelle["core_center_nm"]
        ax.scatter(shell[:, 0], shell[:, 1], shell[:, 2], s=16, alpha=0.7, c=color, label=organelle["organelle_name"])
        ax.scatter([center[0]], [center[1]], [center[2]], s=120, c=color, edgecolors="black")

    ax.set_title("Metallosilicon Cell: Whole-Cell CG Architecture")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_zlabel("z (nm)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_trajectory_snapshots(cell: Dict[str, object], path: Path) -> None:
    frames = cell["whole_cell_dynamics"]["trajectory_frames"]
    indices = [0, len(frames) // 2, len(frames) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, idx in zip(axes, indices):
        frame = frames[idx]
        membrane = project(frame["membrane_coordinates_nm"])
        scaffold = project(frame["genetic_scaffold_coordinates_nm"])
        organelles = np.array(frame["organelle_centers_nm"], dtype=float)[:, [0, 2]]
        ax.scatter(membrane[:, 0], membrane[:, 1], s=7, alpha=0.22, c=COLORS["membrane"])
        ax.plot(scaffold[:, 0], scaffold[:, 1], linewidth=1.6, color=COLORS["scaffold"])
        ax.scatter(organelle_centers := organelles[:, 0], organelles[:, 1], s=110, c=["#b91c1c", "#0f766e", "#1d4ed8"], edgecolors="black")
        ax.set_title(f"Step {frame['step']}")
        ax.set_aspect("equal")
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z (nm)")
    fig.suptitle("Whole-Cell Dynamics Snapshots")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_organelles(cell: Dict[str, object], path: Path) -> None:
    fig = plt.figure(figsize=(15, 4.8))
    for idx, organelle in enumerate(cell["organelles"], start=1):
        ax = fig.add_subplot(1, 3, idx)
        shell = project(organelle["coarse_grained_shell_coordinates_nm"])
        center = organelle["core_center_nm"]
        color = COLORS.get(organelle["organelle_name"], "#334155")
        ax.scatter(shell[:, 0], shell[:, 1], s=16, alpha=0.72, c=color)
        ax.scatter([center[0]], [center[2]], s=180, c=color, edgecolors="black")
        radius = organelle["shell_radius_nm"]
        ax.add_patch(Circle((center[0], center[2]), radius, fill=False, linestyle="--", linewidth=1.5, color=color, alpha=0.4))
        ax.set_title(organelle["organelle_name"])
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z (nm)")
        ax.set_aspect("equal")
        ax.text(
            0.02,
            0.02,
            f"{organelle['metal_cluster']}\n{organelle['role']}",
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
        )
    fig.suptitle("Litho-Organelle Layout")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_component_sheet(cell: Dict[str, object], silicon_monomer: Dict[str, object], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    membrane_nodes = {
        "Si1": (0.1, 0.5),
        "S1": (0.3, 0.7),
        "Si2": (0.5, 0.5),
        "N1": (0.7, 0.3),
        "Si3": (0.9, 0.5),
    }
    membrane_edges = [("Si1", "S1"), ("S1", "Si2"), ("Si2", "N1"), ("N1", "Si3")]
    for a, b in membrane_edges:
        xa, ya = membrane_nodes[a]
        xb, yb = membrane_nodes[b]
        ax.plot([xa, xb], [ya, yb], linewidth=3, color="#475569")
    for label, (x, y) in membrane_nodes.items():
        element = "".join(ch for ch in label if ch.isalpha())
        ax.scatter([x], [y], s=800, c=COLORS.get(element, "#334155"), edgecolors="black")
        ax.text(x, y, element, ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Membrane Repeat Unit")
    ax.text(0.05, 0.08, "Cross-linked thiosilazane motif", transform=ax.transAxes, fontsize=10)
    ax.axis("off")

    ax = axes[0, 1]
    theta = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    ring_labels = ["Si", "S", "Si", "S", "Si", "S", "Si", "S"]
    xy = np.column_stack([0.5 + 0.28 * np.cos(theta), 0.5 + 0.28 * np.sin(theta)])
    for idx in range(len(xy)):
        x1, y1 = xy[idx]
        x2, y2 = xy[(idx + 1) % len(xy)]
        ax.plot([x1, x2], [y1, y2], linewidth=3, color="#475569")
    for (x, y), label in zip(xy, ring_labels):
        ax.scatter([x], [y], s=820, c=COLORS[label], edgecolors="black")
        ax.text(x, y, label, ha="center", va="center", color="white", fontweight="bold")
    ax.set_title("Energy Carrier")
    ax.text(0.08, 0.08, "Cyclic Si4S4 charge-transfer ring", transform=ax.transAxes, fontsize=10)
    ax.axis("off")

    ax = axes[1, 0]
    polymer = ["Si", "N", "Si", "P", "Si", "N", "B", "N"]
    xs = np.linspace(0.08, 0.92, len(polymer))
    ys = np.array([0.55, 0.65, 0.48, 0.37, 0.55, 0.68, 0.52, 0.4])
    for i in range(len(polymer) - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i + 1]], linewidth=2.8, color="#475569")
    for x, y, label in zip(xs, ys, polymer):
        ax.scatter([x], [y], s=760, c=COLORS[label], edgecolors="black")
        ax.text(x, y, label, ha="center", va="center", color="white", fontweight="bold")
    ax.scatter([0.18, 0.82], [0.2, 0.2], s=900, c=[COLORS["Al"], COLORS["Fe"]], edgecolors="black")
    ax.text(0.18, 0.2, "Al", ha="center", va="center", color="white", fontweight="bold")
    ax.text(0.82, 0.2, "Fe", ha="center", va="center", color="white", fontweight="bold")
    ax.plot([0.18, 0.28], [0.2, 0.5], linestyle="--", color="#0f172a")
    ax.plot([0.82, 0.72], [0.2, 0.48], linestyle="--", color="#0f172a")
    ax.set_title("Information Polymer")
    ax.text(0.05, 0.08, "Phosphino-silazane chain with BN insertion", transform=ax.transAxes, fontsize=10)
    ax.axis("off")

    ax = axes[1, 1]
    atom_types = silicon_monomer["atom_types"]
    positions = np.array(silicon_monomer["positions"], dtype=float)
    coords = positions[:, [0, 2]]
    coords -= coords.mean(axis=0)
    bonds = infer_bonds(atom_types, positions)
    for i, j in bonds:
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#94a3b8", linewidth=1.8, zorder=1)
    for idx, (element, (x, y)) in enumerate(zip(atom_types, coords)):
        ax.scatter([x], [y], s=220, c=COLORS.get(element, "#334155"), edgecolors="black", zorder=2)
        if idx < 12:
            ax.text(x, y, element, ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=3)
    ax.set_title(f"Top Silicon-Amino Monomer: {silicon_monomer['candidate_id']}")
    ax.text(
        0.02,
        0.02,
        f"{silicon_monomer['formula']}\nΔE = {silicon_monomer['formation_energy']:.3f} eV/atom",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Cell Components and Chemical Structures")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metabolic_network(cell: Dict[str, object], path: Path) -> None:
    schematic = cell["metabolic_pathway"]["node_edge_schematic"]
    graph = nx.DiGraph()
    for node in schematic["nodes"]:
        graph.add_node(node["id"], kind=node["kind"])
    for edge in schematic["edges"]:
        graph.add_edge(edge["source"], edge["target"], reaction=edge["reaction"])

    pos = {
        "silane_feed": (-1.0, 0.8),
        "h2s_feed": (-1.0, 0.2),
        "ph3_feed": (-1.0, -0.4),
        "thiolated_intermediate": (-0.2, 0.45),
        "phosphinated_intermediate": (-0.2, -0.2),
        "fe_ni_redox_hub": (0.45, 0.2),
        "cyclic_si4s4_carrier": (1.05, 0.2),
        "membrane_polymer": (1.85, 0.55),
        "information_polymer": (1.85, -0.15),
        "division_unit": (2.65, 0.2),
    }
    node_colors = {
        "feedstock": "#64748b",
        "precursor": "#ca8a04",
        "metallocluster": "#b91c1c",
        "energy_carrier": "#7c3aed",
        "structural_polymer": "#0f766e",
        "genetic_polymer": "#1d4ed8",
        "biomass": "#16a34a",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=16, width=2.0, edge_color="#475569")
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[node_colors[graph.nodes[node]["kind"]] for node in graph.nodes],
        node_size=2100,
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels={node: node.replace("_", "\n") for node in graph.nodes}, font_size=9, ax=ax)
    edge_labels = {(u, v): graph.edges[u, v]["reaction"] for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax, label_pos=0.5)
    ax.set_title("Core Silicon-Sulfur Metabolic Pathway")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_channels(cell: Dict[str, object], path: Path) -> None:
    channels = cell["envelope"]["metal_gated_channels"][:16]
    anchors = np.array([c["anchor_coordinate_nm"] for c in channels], dtype=float)
    mouths = np.array([c["mouth_coordinate_nm"] for c in channels], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(anchors[:, 0], anchors[:, 2], s=90, c="#991b1b", edgecolors="black", label="metal anchors")
    ax.scatter(mouths[:, 0], mouths[:, 2], s=55, c="#c2410c", alpha=0.8, label="pore mouths")
    for start, end in zip(anchors, mouths):
        ax.plot([start[0], end[0]], [start[2], end[2]], color="#475569", linewidth=1.5)
    ax.set_title("Metal-Gated Membrane Channels")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("z (nm)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    architectures = load_json(OUTPUT_DIR / "metallosilicon_cell_architectures.json")
    silicon_candidates = load_json(SILICON_AMINO_DIR / "candidates.json")
    best_cell = architectures[0]
    best_monomer = min(silicon_candidates, key=lambda item: item["formation_energy"])

    plot_whole_cell(best_cell, VIS_DIR / "whole_cell_overview.png")
    plot_trajectory_snapshots(best_cell, VIS_DIR / "whole_cell_trajectory_snapshots.png")
    plot_organelles(best_cell, VIS_DIR / "litho_organelles.png")
    plot_component_sheet(best_cell, best_monomer, VIS_DIR / "cell_components_and_chemistry.png")
    plot_metabolic_network(best_cell, VIS_DIR / "metabolic_network.png")
    plot_channels(best_cell, VIS_DIR / "metal_gated_channels.png")

    manifest = {
        "whole_cell_overview": str(VIS_DIR / "whole_cell_overview.png"),
        "whole_cell_trajectory_snapshots": str(VIS_DIR / "whole_cell_trajectory_snapshots.png"),
        "litho_organelles": str(VIS_DIR / "litho_organelles.png"),
        "cell_components_and_chemistry": str(VIS_DIR / "cell_components_and_chemistry.png"),
        "metabolic_network": str(VIS_DIR / "metabolic_network.png"),
        "metal_gated_channels": str(VIS_DIR / "metal_gated_channels.png"),
    }
    (VIS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
