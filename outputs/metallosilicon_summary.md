# Metallosilicon Cell Summary

## Best Architecture

- Architecture ID: `metallosilicon_cell_1`
- Envelope: `crosslinked_polysilazane`
- Solvent: `liquid_pentasilane`
- Screening score: `75.285`
- Division flux: `0.554988`

## Whole-Cell Dynamics

- Membrane stability index: `81.9273`
- Organelle segregation index: `86.7035`
- Scaffold coherence index: `53.4315`
- Mean channel open fraction: `0.3238`

## Permeability

- Silane influx: `1.0832`
- H2S influx: `0.7204`
- PH3 influx: `0.3161`
- Heavy solvent retention: `0.7519`
- Oxygen intrusion probability: `0.03209`

## Structural Features

- Membrane modeled as a cross-linked thiosilazane/polysilazane network rather than a lipid bilayer.
- Three litho-organelles were instantiated: `redox_lithosome`, `polymer_forge`, and `template_lattice`.
- Information storage uses a `phosphino_silazane_information_chain` stabilized by an `Al-Fe` axial spindle.
- Transition metals are used as structural and electron-transfer anchors.

## Metabolic Features

- Feedstocks: silanes, hydrogen sulfide, phosphine.
- Energy carrier: cyclic silicon-sulfur charge-transfer ring.
- Core products: membrane polymer, information polymer, division unit.
- SBML export: `outputs/metallosilicon_metabolism.sbml`

## Related Outputs

- Architectures JSON: `outputs/metallosilicon_cell_architectures.json`
- Candidate screening JSON: `outputs/metallosilicon_candidate_screening.json`
- Comparison report: `outputs/cell_comparison_report.md`
- Visualizations: `outputs/visualizations/`
