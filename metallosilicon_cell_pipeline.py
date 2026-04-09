#!/usr/bin/env python3
"""Speculative metallosilicon single-cell architecture generator.

This script does not attempt atomistic truth. It builds a reproducible,
coarse-grained screening and assembly pipeline under the user-specified
constraints:

- oxygen-poor, reducing chemistry
- >303 K operating window
- silicon, sulfur, nitrogen, phosphorus, boron, aluminum rich framework
- mandatory transition-metal organelle anchors
- no water or phospholipid assumptions

Outputs:
- outputs/metallosilicon_cell_architectures.json
- outputs/metallosilicon_candidate_screening.json
- outputs/metallosilicon_metabolism.sbml
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


SEED = 1729
N_CANDIDATES = 1000
TOP_ARCHITECTURES = 3
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


@dataclass
class EnvelopeCandidate:
    candidate_id: int
    envelope_family: str
    solvent: str
    temperature_k: float
    radius_nm: float
    thickness_nm: float
    crosslink_density: float
    sulfur_fraction: float
    nitrogen_fraction: float
    phosphorus_fraction: float
    boron_fraction: float
    aluminum_fraction: float
    fluorine_fraction: float
    metal_anchor: str
    metal_anchor_fraction: float
    channel_count: int
    pore_radius_nm: float
    curvature_penalty: float
    assembly_score: float
    thermal_score: float
    permeability_selectivity: float
    oxygen_exclusion: float
    leakage_score: float
    overall_score: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def gaussian_peak(value: float, optimum: float, spread: float) -> float:
    return math.exp(-((value - optimum) ** 2) / (2.0 * spread * spread))


def choose_weighted(rng: random.Random, options: Sequence[Tuple[str, float]]) -> str:
    names = [name for name, _ in options]
    weights = [weight for _, weight in options]
    return rng.choices(names, weights=weights, k=1)[0]


def generate_candidates(rng: random.Random, count: int) -> List[EnvelopeCandidate]:
    families = [
        ("polythiosilane", 0.35),
        ("crosslinked_polysilazane", 0.30),
        ("mixed_thiosilazane", 0.35),
    ]
    solvents = [
        ("liquid_pentasilane", 0.45),
        ("sulfur_rich_supercritical_mix", 0.35),
        ("heavy_sulfur_silane_fluid", 0.20),
    ]
    metals = [("Fe", 0.35), ("Ni", 0.25), ("Ti", 0.20), ("Mo", 0.20)]

    candidates: List[EnvelopeCandidate] = []
    for idx in range(count):
        family = choose_weighted(rng, families)
        solvent = choose_weighted(rng, solvents)
        metal = choose_weighted(rng, metals)

        temperature_k = rng.uniform(308.0, 510.0)
        radius_nm = rng.uniform(90.0, 240.0)
        thickness_nm = rng.uniform(8.0, 24.0)
        crosslink_density = rng.uniform(0.18, 0.76)
        sulfur_fraction = rng.uniform(0.22, 0.68)
        nitrogen_fraction = rng.uniform(0.12, 0.44)
        phosphorus_fraction = rng.uniform(0.04, 0.18)
        boron_fraction = rng.uniform(0.01, 0.08)
        aluminum_fraction = rng.uniform(0.01, 0.07)
        fluorine_fraction = rng.uniform(0.0, 0.025)
        metal_anchor_fraction = rng.uniform(0.03, 0.17)
        channel_count = rng.randint(10, 34)
        pore_radius_nm = rng.uniform(0.45, 1.75)

        family_optima = {
            "polythiosilane": (0.41, 0.58, 0.18),
            "crosslinked_polysilazane": (0.56, 0.28, 0.34),
            "mixed_thiosilazane": (0.49, 0.45, 0.26),
        }
        opt_crosslink, opt_sulfur, opt_nitrogen = family_optima[family]
        solvent_match = {
            "liquid_pentasilane": 1.0 if family != "crosslinked_polysilazane" else 0.93,
            "sulfur_rich_supercritical_mix": 1.0 if sulfur_fraction > 0.4 else 0.84,
            "heavy_sulfur_silane_fluid": 1.0 if family == "mixed_thiosilazane" else 0.90,
        }[solvent]
        metal_bonus = {"Fe": 0.98, "Ni": 1.00, "Ti": 0.94, "Mo": 0.96}[metal]

        assembly_score = (
            42.0
            * gaussian_peak(crosslink_density, opt_crosslink, 0.13)
            + 26.0
            * gaussian_peak(sulfur_fraction, opt_sulfur, 0.11)
            + 18.0
            * gaussian_peak(nitrogen_fraction, opt_nitrogen, 0.10)
            + 8.0 * solvent_match
            + 6.0 * metal_bonus
        )
        thermal_score = (
            48.0 * gaussian_peak(temperature_k, 388.0 if family != "polythiosilane" else 366.0, 72.0)
            + 18.0 * crosslink_density
            + 16.0 * metal_anchor_fraction
            + 10.0 * sulfur_fraction
        )
        oxygen_exclusion = clamp(
            100.0
            * (
                0.52
                + 0.24 * crosslink_density
                + 0.12 * nitrogen_fraction
                + 0.08 * boron_fraction
                + 0.06 * fluorine_fraction
                - 0.08 * pore_radius_nm / 2.0
            ),
            0.0,
            100.0,
        )
        permeability_selectivity = clamp(
            100.0
            * (
                0.45
                + 0.16 * sulfur_fraction
                + 0.15 * phosphorus_fraction
                + 0.12 * metal_anchor_fraction
                + 0.08 * min(channel_count / 24.0, 1.4)
                - 0.12 * abs(pore_radius_nm - 0.95)
            ),
            0.0,
            100.0,
        )
        leakage_score = clamp(
            100.0
            * (
                0.72
                + 0.16 * crosslink_density
                + 0.08 * thickness_nm / 24.0
                - 0.24 * pore_radius_nm / 1.75
                - 0.09 * channel_count / 34.0
            ),
            0.0,
            100.0,
        )
        curvature_penalty = clamp(
            100.0 * abs(radius_nm - 155.0) / 155.0 + 15.0 * abs(thickness_nm - 15.0) / 15.0,
            0.0,
            100.0,
        )
        overall_score = (
            0.29 * assembly_score
            + 0.24 * thermal_score
            + 0.19 * permeability_selectivity
            + 0.18 * oxygen_exclusion
            + 0.16 * leakage_score
            - 0.14 * curvature_penalty
        )

        candidates.append(
            EnvelopeCandidate(
                candidate_id=idx + 1,
                envelope_family=family,
                solvent=solvent,
                temperature_k=round(temperature_k, 2),
                radius_nm=round(radius_nm, 2),
                thickness_nm=round(thickness_nm, 2),
                crosslink_density=round(crosslink_density, 4),
                sulfur_fraction=round(sulfur_fraction, 4),
                nitrogen_fraction=round(nitrogen_fraction, 4),
                phosphorus_fraction=round(phosphorus_fraction, 4),
                boron_fraction=round(boron_fraction, 4),
                aluminum_fraction=round(aluminum_fraction, 4),
                fluorine_fraction=round(fluorine_fraction, 4),
                metal_anchor=metal,
                metal_anchor_fraction=round(metal_anchor_fraction, 4),
                channel_count=channel_count,
                pore_radius_nm=round(pore_radius_nm, 3),
                curvature_penalty=round(curvature_penalty, 3),
                assembly_score=round(assembly_score, 3),
                thermal_score=round(thermal_score, 3),
                permeability_selectivity=round(permeability_selectivity, 3),
                oxygen_exclusion=round(oxygen_exclusion, 3),
                leakage_score=round(leakage_score, 3),
                overall_score=round(overall_score, 3),
            )
        )
    return candidates


def fibonacci_sphere_points(n: int, radius: float, rng: random.Random, jitter: float) -> List[List[float]]:
    points: List[List[float]] = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = math.cos(theta) * radial
        z = math.sin(theta) * radial
        vec = np.array([x, y, z], dtype=float)
        vec *= radius + rng.uniform(-jitter, jitter)
        points.append([round(float(v), 3) for v in vec])
    return points


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def make_channel_sites(
    membrane_points: Sequence[Sequence[float]],
    count: int,
    pore_radius_nm: float,
    metal: str,
) -> List[Dict[str, object]]:
    step = max(1, len(membrane_points) // count)
    channels: List[Dict[str, object]] = []
    for idx in range(0, len(membrane_points), step):
        if len(channels) >= count:
            break
        point = np.array(membrane_points[idx], dtype=float)
        direction = normalize(point)
        anchor = point * 0.94
        mouth = point * 1.02
        channels.append(
            {
                "channel_id": f"channel_{len(channels) + 1}",
                "metal_gate": metal,
                "pore_radius_nm": round(pore_radius_nm, 3),
                "anchor_coordinate_nm": [round(float(v), 3) for v in anchor],
                "mouth_coordinate_nm": [round(float(v), 3) for v in mouth],
                "normal_vector": [round(float(v), 4) for v in direction],
            }
        )
    return channels


def make_organelle(
    name: str,
    center: Sequence[float],
    core_radius_nm: float,
    shell_radius_nm: float,
    metal_cluster: str,
    bead_count: int,
    rng: random.Random,
    role: str,
) -> Dict[str, object]:
    local_points = fibonacci_sphere_points(bead_count, shell_radius_nm, rng, shell_radius_nm * 0.08)
    translated = []
    for point in local_points:
        translated.append(
            [
                round(point[0] + center[0], 3),
                round(point[1] + center[1], 3),
                round(point[2] + center[2], 3),
            ]
        )
    return {
        "organelle_name": name,
        "role": role,
        "metal_cluster": metal_cluster,
        "core_center_nm": [round(float(v), 3) for v in center],
        "core_radius_nm": round(core_radius_nm, 3),
        "shell_radius_nm": round(shell_radius_nm, 3),
        "coarse_grained_shell_coordinates_nm": translated,
    }


def make_genetic_scaffold(center: Sequence[float], radius_nm: float, length: int) -> Dict[str, object]:
    points = []
    for idx in range(length):
        theta = idx * 0.64
        axial = -radius_nm + 2.0 * radius_nm * idx / max(1, length - 1)
        x = center[0] + 0.36 * radius_nm * math.cos(theta)
        y = center[1] + axial * 0.33
        z = center[2] + 0.36 * radius_nm * math.sin(theta)
        points.append([round(x, 3), round(y, 3), round(z, 3)])
    return {
        "polymer_class": "phosphino_silazane_information_chain",
        "stabilization_spindle": "Al-Fe mixed axial core",
        "monomer_logic": ["SiN", "SiP", "BN"],
        "coarse_grained_coordinates_nm": points,
    }


def normalized_envelope_composition(candidate: EnvelopeCandidate) -> Dict[str, float]:
    raw = {
        "Si": 1.0,
        "S": candidate.sulfur_fraction,
        "N": candidate.nitrogen_fraction,
        "P": candidate.phosphorus_fraction,
        "B": candidate.boron_fraction,
        "Al": candidate.aluminum_fraction,
        "F": candidate.fluorine_fraction,
        "metal": candidate.metal_anchor_fraction,
    }
    total = sum(raw.values())
    return {element: round(value / total, 4) for element, value in raw.items()}


def simulate_whole_cell_dynamics(
    candidate: EnvelopeCandidate,
    membrane_points: Sequence[Sequence[float]],
    organelles: Sequence[Dict[str, object]],
    genetic_scaffold: Dict[str, object],
    rng: random.Random,
    steps: int = 180,
    save_every: int = 20,
) -> Dict[str, object]:
    membrane = np.array(membrane_points, dtype=float)
    scaffold = np.array(genetic_scaffold["coarse_grained_coordinates_nm"], dtype=float)
    organelle_centers = np.array([item["core_center_nm"] for item in organelles], dtype=float)
    target_radius = candidate.radius_nm

    saved_frames: List[Dict[str, object]] = []
    rmsd_values: List[float] = []
    radius_variances: List[float] = []
    channel_open_fraction: List[float] = []
    min_organelle_clearance: List[float] = []

    membrane_noise = 0.08 + 0.18 * (candidate.temperature_k - 303.0) / 207.0
    centering_strength = 0.028 + 0.04 * candidate.crosslink_density
    scaffold_tension = 0.012 + 0.02 * candidate.nitrogen_fraction

    base_membrane = membrane.copy()
    base_scaffold = scaffold.copy()

    for step in range(steps + 1):
        membrane_norms = np.linalg.norm(membrane, axis=1, keepdims=True)
        membrane_norms[membrane_norms == 0.0] = 1.0
        membrane_unit = membrane / membrane_norms
        radial_error = membrane_norms - target_radius
        membrane -= centering_strength * radial_error * membrane_unit
        membrane += np.random.default_rng(SEED + step).normal(0.0, membrane_noise, membrane.shape)

        organelle_norms = np.linalg.norm(organelle_centers, axis=1, keepdims=True)
        organelle_norms[organelle_norms == 0.0] = 1.0
        organelle_unit = organelle_centers / organelle_norms
        organelle_target = target_radius * np.array([[0.32], [0.27], [0.24]])
        organelle_centers -= 0.05 * (organelle_norms - organelle_target) * organelle_unit
        organelle_centers += np.array(
            [[rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)] for _ in organelles]
        )

        organelle_field = organelle_centers.mean(axis=0) * 0.03
        scaffold += scaffold_tension * (base_scaffold - scaffold)
        scaffold += organelle_field
        scaffold += np.array(
            [[rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025)] for _ in scaffold]
        )

        current_radii = np.linalg.norm(membrane, axis=1)
        rmsd = float(np.sqrt(np.mean(np.sum((membrane - base_membrane) ** 2, axis=1))))
        rmsd_values.append(rmsd)
        radius_variances.append(float(np.var(current_radii)))
        phase = 0.5 + 0.5 * math.sin(step / 11.0 + candidate.channel_count / 7.0)
        channel_open_fraction.append(float(0.18 + 0.44 * phase * (candidate.permeability_selectivity / 100.0)))

        clearance_values = []
        for i in range(len(organelle_centers)):
            for j in range(i + 1, len(organelle_centers)):
                clearance_values.append(float(np.linalg.norm(organelle_centers[i] - organelle_centers[j])))
        min_organelle_clearance.append(min(clearance_values) if clearance_values else 0.0)

        if step % save_every == 0:
            saved_frames.append(
                {
                    "step": step,
                    "membrane_coordinates_nm": [[round(float(v), 3) for v in point] for point in membrane.tolist()],
                    "organelle_centers_nm": [
                        [round(float(v), 3) for v in center] for center in organelle_centers.tolist()
                    ],
                    "genetic_scaffold_coordinates_nm": [
                        [round(float(v), 3) for v in point] for point in scaffold.tolist()
                    ],
                    "channel_open_fraction": round(channel_open_fraction[-1], 4),
                }
            )

    membrane_stability_index = clamp(
        100.0
        * (
            0.74
            + 0.16 * candidate.crosslink_density
            - 0.22 * np.mean(radius_variances) / max(1.0, target_radius)
            - 0.05 * np.mean(rmsd_values) / max(1.0, candidate.thickness_nm)
        ),
        0.0,
        100.0,
    )
    organelle_segregation_index = clamp(
        100.0 * min(min_organelle_clearance) / max(1.0, target_radius * 0.42),
        0.0,
        100.0,
    )
    scaffold_coherence = clamp(
        100.0
        * (
            0.82
            - 0.45 * float(np.mean(np.linalg.norm(scaffold - base_scaffold, axis=1))) / max(1.0, target_radius * 0.18)
        ),
        0.0,
        100.0,
    )

    return {
        "model": "coarse_grained_whole_cell_dynamics",
        "steps": steps,
        "save_every": save_every,
        "temperature_k": candidate.temperature_k,
        "summary": {
            "membrane_stability_index": round(membrane_stability_index, 4),
            "mean_membrane_rmsd_nm": round(float(np.mean(rmsd_values)), 4),
            "mean_radius_variance_nm2": round(float(np.mean(radius_variances)), 4),
            "organelle_segregation_index": round(organelle_segregation_index, 4),
            "minimum_organelle_clearance_nm": round(float(min(min_organelle_clearance)), 4),
            "scaffold_coherence_index": round(scaffold_coherence, 4),
            "mean_channel_open_fraction": round(float(np.mean(channel_open_fraction)), 4),
        },
        "trajectory_frames": saved_frames,
    }


def solve_flux_balance(candidate: EnvelopeCandidate) -> Dict[str, object]:
    reactions = [
        "silane_import",
        "h2s_import",
        "ph3_import",
        "sulfur_activation",
        "phosphorus_activation",
        "carrier_reoxidation",
        "membrane_precursor_synthesis",
        "genome_precursor_synthesis",
        "membrane_assembly",
        "genome_assembly",
        "maintenance",
        "division_assembly",
    ]
    metabolites = [
        "silane",
        "h2s",
        "ph3",
        "activated_sulfur",
        "activated_phosphorus",
        "carrier_red",
        "carrier_ox",
        "energy_token",
        "membrane_precursor",
        "genome_precursor",
        "membrane_polymer",
        "info_polymer",
    ]
    s_matrix = np.array(
        [
            [1, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, -1, -1, 0, -2, 0, 0],
            [0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, -1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, -1, -1, -1, -1, -2],
            [0, 0, 0, 0, 0, 0, 0, 1, -2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, -2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
        ],
        dtype=float,
    )

    selectivity = candidate.permeability_selectivity / 100.0
    exclusion = candidate.oxygen_exclusion / 100.0
    tightness = candidate.leakage_score / 100.0
    import_caps = {
        "silane_import": 9.0 + 9.0 * selectivity,
        "h2s_import": 8.0 + 7.0 * candidate.sulfur_fraction,
        "ph3_import": 4.0 + 8.0 * candidate.phosphorus_fraction,
    }
    maintenance_floor = max(0.4, 1.8 - 0.9 * tightness + 0.35 * (1.0 - exclusion))

    bounds = [
        (0.0, import_caps["silane_import"]),
        (0.0, import_caps["h2s_import"]),
        (0.0, import_caps["ph3_import"]),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (0.0, None),
        (maintenance_floor, maintenance_floor),
        (0.0, None),
    ]

    objective = np.zeros(len(reactions))
    objective[-1] = -1.0

    result = linprog(c=objective, A_eq=s_matrix, b_eq=np.zeros(len(metabolites)), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Flux balance failed: {result.message}")

    fluxes = {reaction: round(float(value), 6) for reaction, value in zip(reactions, result.x)}
    node_edge_schematic = {
        "nodes": [
            {"id": "silane_feed", "kind": "feedstock"},
            {"id": "h2s_feed", "kind": "feedstock"},
            {"id": "ph3_feed", "kind": "feedstock"},
            {"id": "thiolated_intermediate", "kind": "precursor"},
            {"id": "phosphinated_intermediate", "kind": "precursor"},
            {"id": "fe_ni_redox_hub", "kind": "metallocluster"},
            {"id": "cyclic_si4s4_carrier", "kind": "energy_carrier"},
            {"id": "membrane_polymer", "kind": "structural_polymer"},
            {"id": "information_polymer", "kind": "genetic_polymer"},
            {"id": "division_unit", "kind": "biomass"},
        ],
        "edges": [
            {"source": "silane_feed", "target": "thiolated_intermediate", "reaction": "sulfur_activation"},
            {"source": "h2s_feed", "target": "thiolated_intermediate", "reaction": "sulfur_activation"},
            {"source": "silane_feed", "target": "phosphinated_intermediate", "reaction": "phosphorus_activation"},
            {"source": "ph3_feed", "target": "phosphinated_intermediate", "reaction": "phosphorus_activation"},
            {"source": "thiolated_intermediate", "target": "fe_ni_redox_hub", "reaction": "electron_injection"},
            {"source": "fe_ni_redox_hub", "target": "cyclic_si4s4_carrier", "reaction": "carrier_reoxidation"},
            {"source": "cyclic_si4s4_carrier", "target": "membrane_polymer", "reaction": "membrane_assembly"},
            {"source": "cyclic_si4s4_carrier", "target": "information_polymer", "reaction": "genome_assembly"},
            {"source": "membrane_polymer", "target": "division_unit", "reaction": "division_assembly"},
            {"source": "information_polymer", "target": "division_unit", "reaction": "division_assembly"},
        ],
    }
    return {
        "objective": "maximize_division_unit_flux",
        "metabolites": metabolites,
        "reactions": reactions,
        "fluxes": fluxes,
        "max_division_flux": round(float(result.x[-1]), 6),
        "maintenance_flux": round(float(result.x[10]), 6),
        "node_edge_schematic": node_edge_schematic,
    }


def permeability_metrics(candidate: EnvelopeCandidate) -> Dict[str, float]:
    base = candidate.permeability_selectivity / 100.0
    tight = candidate.leakage_score / 100.0
    oxygen_block = candidate.oxygen_exclusion / 100.0
    return {
        "silane_influx": round(0.42 + 0.74 * base + 0.18 * candidate.pore_radius_nm, 4),
        "h2s_influx": round(0.45 + 0.68 * candidate.sulfur_fraction + 0.10 * base, 4),
        "ph3_influx": round(0.18 + 1.05 * candidate.phosphorus_fraction + 0.07 * base, 4),
        "heavy_solvent_retention": round(0.52 + 0.38 * tight, 4),
        "oxygen_intrusion_probability": round(max(0.0001, 0.09 * (1.0 - oxygen_block)), 5),
        "uncontrolled_leakage_probability": round(max(0.0001, 0.07 * (1.0 - tight)), 5),
    }


def build_architecture(candidate: EnvelopeCandidate, rank: int, rng: random.Random) -> Dict[str, object]:
    membrane_points = fibonacci_sphere_points(256, candidate.radius_nm, rng, candidate.thickness_nm * 0.12)
    channels = make_channel_sites(membrane_points, candidate.channel_count, candidate.pore_radius_nm, candidate.metal_anchor)

    organelle_centers = [
        (-0.28 * candidate.radius_nm, 0.18 * candidate.radius_nm, -0.08 * candidate.radius_nm),
        (0.22 * candidate.radius_nm, -0.16 * candidate.radius_nm, 0.12 * candidate.radius_nm),
        (0.04 * candidate.radius_nm, 0.03 * candidate.radius_nm, -0.22 * candidate.radius_nm),
    ]
    organelles = [
        make_organelle(
            "redox_lithosome",
            organelle_centers[0],
            core_radius_nm=10.0,
            shell_radius_nm=18.0,
            metal_cluster=f"{candidate.metal_anchor}-Ni-S",
            bead_count=42,
            rng=rng,
            role="charge separation and cyclic Si-S carrier regeneration",
        ),
        make_organelle(
            "polymer_forge",
            organelle_centers[1],
            core_radius_nm=9.0,
            shell_radius_nm=15.0,
            metal_cluster=f"Mo-{candidate.metal_anchor}-P",
            bead_count=34,
            rng=rng,
            role="phosphino-silazane monomer synthesis and membrane strand welding",
        ),
        make_organelle(
            "template_lattice",
            organelle_centers[2],
            core_radius_nm=8.0,
            shell_radius_nm=13.0,
            metal_cluster=f"Al-Fe-BN",
            bead_count=28,
            rng=rng,
            role="information polymer templating and segregation before division",
        ),
    ]
    genetic_scaffold = make_genetic_scaffold((0.0, 0.0, 0.0), candidate.radius_nm * 0.26, 72)
    flux_model = solve_flux_balance(candidate)
    dynamics = simulate_whole_cell_dynamics(candidate, membrane_points, organelles, genetic_scaffold, rng)

    architecture = {
        "architecture_id": f"metallosilicon_cell_{rank}",
        "rank": rank,
        "screening_score": candidate.overall_score,
        "environment": {
            "temperature_k": candidate.temperature_k,
            "solvent": candidate.solvent,
            "oxygen_atomic_fraction_upper_bound": 0.01,
            "hydrogen_inventory_reference": "Earth ocean + crust + atmosphere equivalent",
        },
        "envelope": {
            "family": candidate.envelope_family,
            "radius_nm": candidate.radius_nm,
            "thickness_nm": candidate.thickness_nm,
            "crosslink_density": candidate.crosslink_density,
            "composition_atomic_fraction": normalized_envelope_composition(candidate),
            "membrane_coordinates_nm": membrane_points,
            "metal_gated_channels": channels,
        },
        "organelles": organelles,
        "genetic_scaffold": genetic_scaffold,
        "whole_cell_dynamics": dynamics,
        "permeability_metrics": permeability_metrics(candidate),
        "metabolic_pathway": flux_model,
        "design_notes": [
            "Envelope is treated as an amphiphilic ceramic-polymer network, not a lipid bilayer.",
            "Transition-metal nodes act as both channel gates and electron-transport anchors.",
            "Information polymer is modeled as a phosphino-silazane chain stabilized by Al/Fe scaffolding.",
            "All scores and fluxes are heuristic outputs from a coarse-grained speculative model.",
        ],
    }
    return architecture


def build_sbml(best_architecture: Dict[str, object]) -> str:
    fluxes = best_architecture["metabolic_pathway"]["fluxes"]
    species = [
        ("silane", "cytosol"),
        ("h2s", "cytosol"),
        ("ph3", "cytosol"),
        ("activated_sulfur", "cytosol"),
        ("activated_phosphorus", "cytosol"),
        ("carrier_red", "redox_lithosome"),
        ("carrier_ox", "redox_lithosome"),
        ("energy_token", "cytosol"),
        ("membrane_precursor", "polymer_forge"),
        ("genome_precursor", "template_lattice"),
        ("membrane_polymer", "cytosol"),
        ("info_polymer", "cytosol"),
    ]
    reactions = [
        ("silane_import", "silane import through metal-gated pore"),
        ("h2s_import", "hydrogen sulfide import through sulfur-tuned pore"),
        ("ph3_import", "phosphine import through narrow phosphorus channel"),
        ("sulfur_activation", "sulfur activation on Fe/Ni cluster"),
        ("phosphorus_activation", "phosphorus activation on Mo-containing forge"),
        ("carrier_reoxidation", "cyclic silicon-sulfur carrier recharge"),
        ("membrane_precursor_synthesis", "membrane precursor synthesis"),
        ("genome_precursor_synthesis", "information precursor synthesis"),
        ("membrane_assembly", "polythiosilazane membrane assembly"),
        ("genome_assembly", "phosphino-silazane genome assembly"),
        ("maintenance", "homeostatic energy drain"),
        ("division_assembly", "complete cell division unit assembly"),
    ]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">',
        '  <model id="metallosilicon_single_cell" name="Metallosilicon Single Cell">',
        '    <notes>',
        '      <body xmlns="http://www.w3.org/1999/xhtml">',
        '        <p>Speculative low-oxygen metallosilicon cell model generated by a coarse-grained pipeline.</p>',
        f'        <p>Selected envelope family: {best_architecture["envelope"]["family"]}</p>',
        f'        <p>Operating solvent: {best_architecture["environment"]["solvent"]}</p>',
        '      </body>',
        '    </notes>',
        '    <listOfCompartments>',
        '      <compartment id="exterior" constant="true" size="1"/>',
        '      <compartment id="cytosol" constant="true" size="1"/>',
        '      <compartment id="redox_lithosome" constant="true" size="1"/>',
        '      <compartment id="polymer_forge" constant="true" size="1"/>',
        '      <compartment id="template_lattice" constant="true" size="1"/>',
        '    </listOfCompartments>',
        '    <listOfSpecies>',
    ]
    for species_id, compartment in species:
        lines.append(
            f'      <species id="{species_id}" compartment="{compartment}" initialAmount="0" boundaryCondition="false" constant="false"/>'
        )
    lines.extend(['    </listOfSpecies>', '    <listOfParameters>'])
    for reaction_id, _ in reactions:
        lines.append(
            f'      <parameter id="flux_{reaction_id}" value="{fluxes[reaction_id]}" constant="true"/>'
        )
    lines.extend(['    </listOfParameters>', '    <listOfReactions>'])
    for reaction_id, reaction_name in reactions:
        lines.extend(
            [
                f'      <reaction id="{reaction_id}" name="{reaction_name}" reversible="false" fast="false">',
                '        <notes>',
                '          <body xmlns="http://www.w3.org/1999/xhtml">',
                f'            <p>Flux solution: {fluxes[reaction_id]}</p>',
                '          </body>',
                '        </notes>',
                '      </reaction>',
            ]
        )
    lines.extend(['    </listOfReactions>', '  </model>', '</sbml>'])
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    candidates = generate_candidates(rng, N_CANDIDATES)
    ranked = sorted(candidates, key=lambda item: item.overall_score, reverse=True)
    top_candidates = ranked[:TOP_ARCHITECTURES]

    architecture_rng = random.Random(SEED + 99)
    architectures = [
        build_architecture(candidate, rank=index + 1, rng=architecture_rng)
        for index, candidate in enumerate(top_candidates)
    ]

    screening_summary = {
        "model_type": "speculative_multiscale_cgmd_fba",
        "candidate_count": len(candidates),
        "constraints": {
            "oxygen_atomic_fraction_upper_bound": 0.01,
            "temperature_k_lower_bound": 303.0,
            "primary_elements": ["Si", "N", "H", "S", "P", "Al", "B", "F"],
            "mandatory_transition_metals": ["Fe", "Ni", "Ti", "Mo"],
            "excluded_defaults": ["water cytosol", "phospholipid bilayer", "carbon-dominant biochemistry"],
        },
        "top_candidates": [asdict(candidate) for candidate in top_candidates],
        "all_candidate_scores": [
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.envelope_family,
                "solvent": candidate.solvent,
                "overall_score": candidate.overall_score,
                "oxygen_exclusion": candidate.oxygen_exclusion,
                "permeability_selectivity": candidate.permeability_selectivity,
            }
            for candidate in ranked
        ],
    }

    architecture_path = OUTPUT_DIR / "metallosilicon_cell_architectures.json"
    screening_path = OUTPUT_DIR / "metallosilicon_candidate_screening.json"
    sbml_path = OUTPUT_DIR / "metallosilicon_metabolism.sbml"

    architecture_path.write_text(json.dumps(architectures, indent=2))
    screening_path.write_text(json.dumps(screening_summary, indent=2))
    sbml_path.write_text(build_sbml(architectures[0]))

    print(f"Wrote {architecture_path}")
    print(f"Wrote {screening_path}")
    print(f"Wrote {sbml_path}")
    print(
        json.dumps(
            {
                "best_architecture_id": architectures[0]["architecture_id"],
                "best_score": architectures[0]["screening_score"],
                "best_division_flux": architectures[0]["metabolic_pathway"]["max_division_flux"],
                "best_oxygen_intrusion_probability": architectures[0]["permeability_metrics"][
                    "oxygen_intrusion_probability"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
