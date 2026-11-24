# FrameLA1 Interactive Quantum Lab

Interactive browser simulator and Python verification script for the `FrameLA1` near-tight frame operator lemma (`Q = sum_i gamma_i B_i B_i*`) from the Alpöge–Claude (2026) Lean proof of the Riemann Hypothesis, applied to noisy classical shadows, spectral flatness (Cantelli tail bound), and overpacked non-isometric AQECCs.

Crossley uses Alpöge’s `FrameLA1` to bound rank loss in Hermitian lattices. The proof takes the Gram and trace errors used here and sums twinheterodox’s rank estimate over the slope thresholds. [Lean proof](https://github.com/leecrossley/prime-window-rh-criterion/blob/8f8ce11af908bb98df5c2adc2c8a72856fc2b8cd/results/geometric-window-descent/lean/FiniteGeometricFrame.lean#L35-L49) and [The Crossley Criteria, §1](https://doi.org/10.5281/zenodo.24466908).

## Deploy to GitHub Pages
1. Push `index.html` to a GitHub repository.
2. Go to **Settings → Pages → Source: Deploy from a branch (`main` / `/root`)**.
3. Open `https://<your-username>.github.io/<repo-name>/` — no backend or Python server required.
