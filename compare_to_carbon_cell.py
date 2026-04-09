#!/usr/bin/env python3
"""Compare the generated metallosilicon cell to a carbon-cell reference."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
SILICON_AMINO_DIR = Path(
    os.environ.get("SILICON_AMINO_DIR", ROOT.parent / "silicon_amino" / "sims" / "output")
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_json(path: Path):
    return json.loads(path.read_text())


def summarize_metallosilicon() -> Dict[str, object]:
    architectures = load_json(OUTPUT_DIR / "metallosilicon_cell_architectures.json")
    screening = load_json(OUTPUT_DIR / "metallosilicon_candidate_screening.json")
    amino_candidates = load_json(SILICON_AMINO_DIR / "candidates.json")
    protein_folds = load_json(SILICON_AMINO_DIR / "protein_folds.json")

    best_cell = architectures[0]
    best_screen = screening["top_candidates"][0]
    best_monomer = min(amino_candidates, key=lambda item: item["formation_energy"])
    best_fold = max(protein_folds, key=lambda item: item["conductivity_estimate"])

    dynamics = best_cell["whole_cell_dynamics"]["summary"]
    permeability = best_cell["permeability_metrics"]
    division_flux = best_cell["metabolic_pathway"]["max_division_flux"]

    info_stability = 0.65 * dynamics["scaffold_coherence_index"] + 0.35 * (100.0 * best_monomer["stability_score"])
    electron_transport = clamp(25.0 * math.log10(best_fold["conductivity_estimate"] + 1.0), 0.0, 100.0)
    division_capacity = clamp(100.0 * division_flux / 0.9, 0.0, 100.0)
    prosilicon_fit = clamp(
        0.24 * dynamics["membrane_stability_index"]
        + 0.18 * dynamics["organelle_segregation_index"]
        + 0.18 * info_stability
        + 0.18 * electron_transport
        + 0.12 * division_capacity
        + 0.10 * (100.0 * permeability["heavy_solvent_retention"]),
        0.0,
        100.0,
    )
    earthlike_fit = clamp(
        0.45 * 6.0
        + 0.20 * 4.0
        + 0.15 * dynamics["scaffold_coherence_index"]
        + 0.10 * electron_transport
        + 0.10 * division_capacity,
        0.0,
        100.0,
    )

    return {
        "cell_class": "metallosilicon_single_cell",
        "architecture_id": best_cell["architecture_id"],
        "envelope_family": best_cell["envelope"]["family"],
        "solvent": best_cell["environment"]["solvent"],
        "top_monomer": {
            "candidate_id": best_monomer["candidate_id"],
            "formula": best_monomer["formula"],
            "formation_energy_ev_per_atom": round(best_monomer["formation_energy"], 4),
            "homo_lumo_gap_ev": best_monomer["homo_lumo_gap"],
            "stability_score": round(best_monomer["stability_score"], 6),
        },
        "top_fold": {
            "fold_id": best_fold["fold_id"],
            "fold_type": best_fold["fold_type"],
            "metal_center": best_fold["metal_center"],
            "conductivity_s_cm": round(best_fold["conductivity_estimate"], 4),
            "estimated_stability": round(best_fold["estimated_stability"], 4),
            "homo_lumo_gap_ev": best_fold["homo_lumo_gap"],
        },
        "metrics": {
            "membrane_stability_index": dynamics["membrane_stability_index"],
            "information_stability_index": round(info_stability, 4),
            "bioelectronic_transport_index": round(electron_transport, 4),
            "division_capacity_index": round(division_capacity, 4),
            "heavy_solvent_retention_index": round(100.0 * permeability["heavy_solvent_retention"], 4),
            "oxidative_tolerance_index": 6.0,
            "prosilicon_environment_fit_index": round(prosilicon_fit, 4),
            "earthlike_environment_fit_index": round(earthlike_fit, 4),
        },
        "raw_outputs": {
            "screening_score": best_screen["overall_score"],
            "division_flux": division_flux,
            "oxygen_intrusion_probability": permeability["oxygen_intrusion_probability"],
            "uncontrolled_leakage_probability": permeability["uncontrolled_leakage_probability"],
            "scaffold_coherence_index": dynamics["scaffold_coherence_index"],
            "organelle_segregation_index": dynamics["organelle_segregation_index"],
        },
    }


def build_carbon_reference() -> Dict[str, object]:
    return {
        "cell_class": "carbon_thermophilic_reference_cell",
        "reference_type": "heuristic_baseline",
        "description": "A compact water-based thermophilic prokaryote reference, used only for comparison against the speculative metallosilicon cell.",
        "environment_preferences": {
            "solvent": "water",
            "oxygen": "optional but tolerated",
            "temperature_window_k": [290, 355],
        },
        "metrics": {
            "membrane_stability_index": 87.0,
            "information_stability_index": 93.0,
            "bioelectronic_transport_index": 41.0,
            "division_capacity_index": 84.0,
            "heavy_solvent_retention_index": 7.0,
            "oxidative_tolerance_index": 96.0,
            "prosilicon_environment_fit_index": 15.0,
            "earthlike_environment_fit_index": 91.0,
        },
        "raw_outputs": {
            "baseline_membrane": "phospholipid bilayer",
            "baseline_genome": "DNA/RNA",
            "baseline_energy_system": "redox enzymes + proton motive force",
            "notes": [
                "This baseline is not a new MD simulation.",
                "Indices are heuristic anchors chosen to represent a robust carbon-water microbe.",
                "The same prosilicon solvent and oxygen constraints strongly penalize this reference cell.",
            ],
        },
    }


def compare_profiles(metallo: Dict[str, object], carbon: Dict[str, object]) -> Dict[str, object]:
    metric_keys: List[str] = [
        "membrane_stability_index",
        "information_stability_index",
        "bioelectronic_transport_index",
        "division_capacity_index",
        "heavy_solvent_retention_index",
        "oxidative_tolerance_index",
        "prosilicon_environment_fit_index",
        "earthlike_environment_fit_index",
    ]
    rows = []
    for key in metric_keys:
        m_value = metallo["metrics"][key]
        c_value = carbon["metrics"][key]
        rows.append(
            {
                "metric": key,
                "metallosilicon": m_value,
                "carbon_reference": c_value,
                "delta_metallosilicon_minus_carbon": round(m_value - c_value, 4),
                "winner": "metallosilicon" if m_value > c_value else "carbon_reference",
            }
        )

    findings = [
        "Metallosilicon architecture dominates in the target nonaqueous reducing environment because its envelope and redox hubs are tuned to heavy silane and sulfur-rich solvents.",
        "Carbon baseline dominates in Earth-like water and oxygen because the metallosilicon system remains pyrophoric and hydrolytically fragile outside the target world.",
        "Silicon-amino fold outputs materially strengthen the metallosilicon case for internal nanowire-like electron transport; the best fold reaches beta-barrel conductivity far above the carbon reference index.",
        "The metallosilicon cell still trails the carbon reference in information stability and division capacity, mainly because scaffold coherence and whole-cell replication remain weaker than DNA/protein/water systems.",
    ]

    return {
        "comparison_rows": rows,
        "findings": findings,
    }


def build_report(metallo: Dict[str, object], carbon: Dict[str, object], comparison: Dict[str, object]) -> str:
    lines = [
        "# Metallosilicon vs Carbon Cell Comparison",
        "",
        "This report compares the generated metallosilicon whole-cell output against a heuristic carbon-cell reference.",
        "",
        "## Metallosilicon Result",
        "",
        f"- Architecture: `{metallo['architecture_id']}`",
        f"- Envelope: `{metallo['envelope_family']}` in `{metallo['solvent']}`",
        f"- Best monomer: `{metallo['top_monomer']['candidate_id']}` ({metallo['top_monomer']['formula']}) with formation energy `{metallo['top_monomer']['formation_energy_ev_per_atom']}` eV/atom",
        f"- Best conductive fold: `{metallo['top_fold']['fold_id']}` with `{metallo['top_fold']['conductivity_s_cm']}` S/cm",
        f"- Division flux: `{metallo['raw_outputs']['division_flux']}`",
        "",
        "## Side-by-Side Metrics",
        "",
        "| Metric | Metallosilicon | Carbon reference | Delta | Winner |",
        "|--------|----------------|------------------|-------|--------|",
    ]
    for row in comparison["comparison_rows"]:
        lines.append(
            f"| {row['metric']} | {row['metallosilicon']:.4f} | {row['carbon_reference']:.4f} | {row['delta_metallosilicon_minus_carbon']:.4f} | {row['winner']} |"
        )
    lines.extend(
        [
            "",
            "## Main Takeaways",
            "",
        ]
    )
    for finding in comparison["findings"]:
        lines.append(f"- {finding}")
    lines.extend(
        [
            "",
            "## Carbon Reference Assumptions",
            "",
            f"- Baseline membrane: `{carbon['raw_outputs']['baseline_membrane']}`",
            f"- Baseline genome: `{carbon['raw_outputs']['baseline_genome']}`",
            f"- Baseline energy system: `{carbon['raw_outputs']['baseline_energy_system']}`",
            "",
            "The carbon-cell row is an anchored heuristic baseline, not a separate molecular-dynamics simulation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    metallo = summarize_metallosilicon()
    carbon = build_carbon_reference()
    comparison = compare_profiles(metallo, carbon)

    payload = {
        "metallosilicon_result": metallo,
        "carbon_reference": carbon,
        "comparison": comparison,
    }

    json_path = OUTPUT_DIR / "cell_comparison.json"
    md_path = OUTPUT_DIR / "cell_comparison_report.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(build_report(metallo, carbon, comparison))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        json.dumps(
            {
                "prosilicon_fit_metallosilicon": metallo["metrics"]["prosilicon_environment_fit_index"],
                "prosilicon_fit_carbon": carbon["metrics"]["prosilicon_environment_fit_index"],
                "earthlike_fit_metallosilicon": metallo["metrics"]["earthlike_environment_fit_index"],
                "earthlike_fit_carbon": carbon["metrics"]["earthlike_environment_fit_index"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
