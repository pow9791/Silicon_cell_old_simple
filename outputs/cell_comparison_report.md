# Metallosilicon vs Carbon Cell Comparison

This report compares the generated metallosilicon whole-cell output against a heuristic carbon-cell reference.

## Metallosilicon Result

- Architecture: `metallosilicon_cell_1`
- Envelope: `crosslinked_polysilazane` in `liquid_pentasilane`
- Best monomer: `MSA-Si5N3SFH12-i01` (Si5N3SFH12) with formation energy `-3.6303` eV/atom
- Best conductive fold: `SiProtein-beta_barrel-Fe-12res` with `3922.388` S/cm
- Division flux: `0.554988`

## Side-by-Side Metrics

| Metric | Metallosilicon | Carbon reference | Delta | Winner |
|--------|----------------|------------------|-------|--------|
| membrane_stability_index | 81.9273 | 87.0000 | -5.0727 | carbon_reference |
| information_stability_index | 69.7305 | 93.0000 | -23.2695 | carbon_reference |
| bioelectronic_transport_index | 89.8415 | 41.0000 | 48.8415 | metallosilicon |
| division_capacity_index | 61.6653 | 84.0000 | -22.3347 | carbon_reference |
| heavy_solvent_retention_index | 75.1900 | 7.0000 | 68.1900 | metallosilicon |
| oxidative_tolerance_index | 6.0000 | 96.0000 | -90.0000 | carbon_reference |
| prosilicon_environment_fit_index | 78.9110 | 15.0000 | 63.9110 | metallosilicon |
| earthlike_environment_fit_index | 26.6654 | 91.0000 | -64.3346 | carbon_reference |

## Main Takeaways

- Metallosilicon architecture dominates in the target nonaqueous reducing environment because its envelope and redox hubs are tuned to heavy silane and sulfur-rich solvents.
- Carbon baseline dominates in Earth-like water and oxygen because the metallosilicon system remains pyrophoric and hydrolytically fragile outside the target world.
- Silicon-amino fold outputs materially strengthen the metallosilicon case for internal nanowire-like electron transport; the best fold reaches beta-barrel conductivity far above the carbon reference index.
- The metallosilicon cell still trails the carbon reference in information stability and division capacity, mainly because scaffold coherence and whole-cell replication remain weaker than DNA/protein/water systems.

## Carbon Reference Assumptions

- Baseline membrane: `phospholipid bilayer`
- Baseline genome: `DNA/RNA`
- Baseline energy system: `redox enzymes + proton motive force`

The carbon-cell row is an anchored heuristic baseline, not a separate molecular-dynamics simulation.
