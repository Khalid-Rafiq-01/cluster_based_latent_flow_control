# Simulation data

The raw CFD snapshot data used in this study are not included directly in this repository because of their size.

The actuation-enriched dataset consists of 56 open-loop simulations generated from four forcing amplitudes and fourteen forcing frequencies. Flow snapshots were sampled over the statistically developed portion of each simulation, giving 22,400 snapshots in total.

Compact quantities derived from these simulations and required for the analysis are included in:

- `../artifacts/` — latent states, cluster centroids, cluster labels, aerodynamic statistics, and the fitted UMAP transformation.
- `../control_laws/` — optimized feedback-control parameters for the C0 hold and C6 escape cases.

The trained residual VAE checkpoint and large CFD datasets will be distributed separately through an archival data release.

A permanent download link will be added here upon release.
