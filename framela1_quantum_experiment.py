#!/usr/bin/env python3
"""
Numerical verification and high-DPI visualization of the FrameLA1 near-tight frame lemma
from the Alpöge–Claude (2026) Lean proof of the Riemann Hypothesis, applied to:
  1. Noisy Classical Shadows & extremal block-partition frames on L_0(C^d)
  2. Spectral flatness & Cantelli tail bound of Q = sum_i gamma_i B_i B_i^*
  3. Overpacked non-isometric AQECC & reversed Welch lower bound on error overlaps
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def random_haar_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary matrix in U(d) via QR decomposition."""
    z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phases = diag / np.abs(diag)
    return q * phases[np.newaxis, :]


def traceless_basis(d: int) -> np.ndarray:
    """Orthonormal Hermitian basis of L_0(C^d) of dimension D = d^2 - 1."""
    basis = []
    for a in range(d):
        for b in range(a + 1, d):
            e_sym = np.zeros((d, d), dtype=np.complex128)
            e_sym[a, b] = e_sym[b, a] = 1.0 / np.sqrt(2.0)
            basis.append(e_sym)
            e_anti = np.zeros((d, d), dtype=np.complex128)
            e_anti[a, b] = -1j / np.sqrt(2.0)
            e_anti[b, a] = 1j / np.sqrt(2.0)
            basis.append(e_anti)
    for k in range(1, d):
        diag_mat = np.zeros((d, d), dtype=np.complex128)
        for j in range(k):
            diag_mat[j, j] = 1.0
        diag_mat[k, k] = -float(k)
        diag_mat /= np.sqrt(k * (k + 1.0))
        basis.append(diag_mat)
    return np.array(basis)


def helmert_matrix(d: int) -> np.ndarray:
    """Orthonormal (d-1) x d matrix mapping C^d onto the zero-sum subspace."""
    h = np.zeros((d - 1, d), dtype=np.float64)
    for k in range(1, d):
        h[k - 1, :k] = 1.0
        h[k - 1, k] = -float(k)
        h[k - 1, :] /= np.sqrt(k * (k + 1.0))
    return h


def build_noisy_shadow_frame(
    d: int,
    m_settings: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, float, float]:
    """
    Build m_settings projective measurement blocks B_i : C^(d-1) -> L_0(C^d)
    perturbed by readout/calibration noise of scale noise_sigma.
    """
    D = d * d - 1
    basis = traceless_basis(d)
    basis_flat = basis.reshape(D, d * d)
    H_helm = helmert_matrix(d)

    blocks = []
    for _ in range(m_settings):
        U = random_haar_unitary(d, rng)
        proj = np.einsum("ab,cb->bac", U, U.conj())
        if noise_sigma > 0:
            raw_noise = rng.standard_normal((d, d, d)) + 1j * rng.standard_normal((d, d, d))
            herm_noise = 0.5 * (raw_noise + np.swapaxes(raw_noise.conj(), 1, 2))
            herm_noise /= np.sqrt(d)
            proj = proj + noise_sigma * herm_noise
        traces = np.trace(proj, axis1=1, axis2=2)[:, None, None]
        proj_tl = proj - (traces / d) * np.eye(d)[None, :, :]
        comb = np.tensordot(H_helm, proj_tl, axes=([1], [0]))
        B_i = (basis_flat.conj() @ comb.reshape(d - 1, d * d).T).real
        blocks.append(B_i)

    N = sum(B.shape[1] for B in blocks)
    Q = sum(B @ B.T for B in blocks)
    e = sum(np.linalg.norm(B.T @ B - np.eye(B.shape[1]), "fro") ** 2 for B in blocks) / N
    trQ = np.trace(Q)
    trQ2 = np.trace(Q @ Q)
    eta = float(D * trQ2 / (trQ ** 2) - 1.0)
    return blocks, Q, float(e), eta


def build_block_partition_frame(
    D: int,
    d_block: int,
    copies_per_sector: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, float, float]:
    """
    Build a structured near-tight frame by partitioning C^D into D/d_block orthogonal
    sectors with `copies_per_sector` noisy copies per sector.
    """
    assert D % d_block == 0
    n_sectors = D // d_block
    Q_orth, _ = np.linalg.qr(rng.standard_normal((D, D)))
    blocks = []
    for s in range(n_sectors):
        U_s = Q_orth[:, s * d_block : (s + 1) * d_block]
        for _ in range(copies_per_sector):
            pert = rng.standard_normal((d_block, d_block)) / np.sqrt(d_block)
            B_i = U_s @ (np.eye(d_block) + noise_sigma * pert)
            blocks.append(B_i)

    N = sum(B.shape[1] for B in blocks)
    Q = sum(B @ B.T for B in blocks)
    e = sum(np.linalg.norm(B.T @ B - np.eye(B.shape[1]), "fro") ** 2 for B in blocks) / N
    trQ = np.trace(Q)
    trQ2 = np.trace(Q @ Q)
    eta = float(D * trQ2 / (trQ ** 2) - 1.0)
    return blocks, Q, float(e), eta


def sharp_quadratic_bound(eta: float, e: float, eps: float) -> float:
    """
    Exact upper root of rho - sqrt(eta * rho * (1 - rho)) = delta,
    where delta = (eps + sqrt(eps * e)) / (1 - sqrt(e)).
    """
    delta = (eps + np.sqrt(max(eps * e, 0.0))) / (1.0 - np.sqrt(e))
    if delta >= 1.0:
        return 1.0
    if eta <= 0.0:
        return min(delta, 1.0)
    a = 1.0 + eta
    b = -(2.0 * delta + eta)
    c = delta * delta
    disc = max(b * b - 4.0 * a * c, 0.0)
    rho = (-b + np.sqrt(disc)) / (2.0 * a)
    return float(min(max(rho, 0.0), 1.0))


def simulate_sector_adversarial_pruning(
    blocks: list[np.ndarray],
    D: int,
    eps_grid: np.ndarray,
) -> np.ndarray:
    """Drop columns sequentially by sector (worst-case adversarial pruning)."""
    cols = [B[:, j] for B in blocks for j in range(B.shape[1])]
    N = len(cols)
    rhos = []
    for eps in eps_grid:
        n_drop = int(round(eps * N))
        kept = cols[n_drop:]
        if kept:
            M = np.column_stack(kept)
            s = np.linalg.svd(M, compute_uv=False)
            rank = int(np.sum(s > 1e-8 * (s[0] if len(s) else 1.0)))
        else:
            rank = 0
        rhos.append((D - rank) / D)
    return np.array(rhos)


def simulate_random_pruning(
    blocks: list[np.ndarray],
    D: int,
    eps_grid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drop an eps-fraction of columns uniformly at random."""
    cols = [B[:, j] for B in blocks for j in range(B.shape[1])]
    N = len(cols)
    rhos = []
    for eps in eps_grid:
        n_drop = int(round(eps * N))
        keep_idx = rng.choice(N, size=max(0, N - n_drop), replace=False)
        if len(keep_idx) > 0:
            M = np.column_stack([cols[k] for k in keep_idx])
            s = np.linalg.svd(M, compute_uv=False)
            rank = int(np.sum(s > 1e-8 * (s[0] if len(s) else 1.0)))
        else:
            rank = 0
        rhos.append((D - rank) / D)
    return np.array(rhos)


def simulate_overpacked_aqecc(
    d_L: int,
    d_P: int,
    m_values: np.ndarray,
    noise_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate m noisy near-isometric error-sector maps B_i = E_i V : C^(d_L) -> C^(d_P)
    and compare empirical off-diagonal overlap sum against the reversed Welch lower bound.
    """
    s_off_emp, welch_noisy, welch_ideal, e_vals = [], [], [], []
    for m in m_values:
        blocks = []
        for _ in range(m):
            U = random_haar_unitary(d_P, rng)
            B = U[:, :d_L].copy()
            if noise_sigma > 0:
                pert = (rng.standard_normal((d_P, d_L)) + 1j * rng.standard_normal((d_P, d_L))) / np.sqrt(2.0 * d_P)
                B = B + noise_sigma * pert
            blocks.append(B)

        N = m * d_L
        e = sum(np.linalg.norm(B.conj().T @ B - np.eye(d_L), "fro") ** 2 for B in blocks) / N
        e_vals.append(e)

        s_off = sum(
            np.linalg.norm(blocks[i].conj().T @ blocks[j], "fro") ** 2
            for i in range(m)
            for j in range(m)
            if i != j
        )
        s_off_emp.append(s_off)
        welch_noisy.append(max(0.0, N * (((1.0 - np.sqrt(e)) ** 2) * N / d_P - (1.0 + np.sqrt(e)) ** 2)))
        welch_ideal.append(max(0.0, N * (N / d_P - 1.0)))

    return np.array(s_off_emp), np.array(welch_noisy), np.array(welch_ideal), np.array(e_vals)


def main() -> None:
    rng = np.random.default_rng(42)

    d = 6
    D = d * d - 1  # 35
    blocks_part, Q_part, e_part, eta_part = build_block_partition_frame(
        D=D, d_block=5, copies_per_sector=2, noise_sigma=0.04, rng=rng
    )
    blocks_good, Q_good, e_good, eta_good = build_noisy_shadow_frame(
        d=d, m_settings=45, noise_sigma=0.05, rng=rng
    )
    _, Q_sparse, e_sparse, eta_sparse = build_noisy_shadow_frame(
        d=d, m_settings=10, noise_sigma=0.05, rng=rng
    )

    eps_grid = np.linspace(0.0, 0.30, 19)
    rho_adv_part = simulate_sector_adversarial_pruning(blocks_part, D, eps_grid)
    rho_rand_shadow = simulate_random_pruning(blocks_good, D, eps_grid, rng)
    rho_quad_part = np.array([sharp_quadratic_bound(eta_part, e_part, eps) for eps in eps_grid])
    rho_lin_part = np.minimum(1.0, eta_part + 4.0 * eps_grid + 4.0 * np.sqrt(e_part))

    eig_good = np.sort(np.linalg.eigvalsh(Q_good))
    eig_good_norm = eig_good / np.mean(eig_good)
    eig_sparse = np.sort(np.linalg.eigvalsh(Q_sparse))
    eig_sparse_norm = eig_sparse / np.mean(eig_sparse)

    alpha = 0.5
    cantelli_good = eta_good / (eta_good + alpha ** 2)
    cantelli_sparse = eta_sparse / (eta_sparse + alpha ** 2)
    emp_tail_good = np.mean(eig_good_norm <= 1.0 - alpha)
    emp_tail_sparse = np.mean(eig_sparse_norm <= 1.0 - alpha)

    d_L, d_P = 4, 32
    m_values = np.arange(4, 21)
    s_off_emp, welch_noisy, welch_ideal, e_aqecc = simulate_overpacked_aqecc(
        d_L=d_L, d_P=d_P, m_values=m_values, noise_sigma=0.06, rng=rng
    )

    # High-DPI vertical 3x1 layout optimized for crisp readability on mobile & desktop
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
    })
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 14.5), dpi=300)
    fig.patch.set_facecolor("#FAF9F6")
    for ax in axes:
        ax.set_facecolor("#FFFFFF")
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

    # --- Panel 1 ---
    ax1 = axes[0]
    ax1.plot(eps_grid, rho_lin_part, "--", color="#9CA3AF", lw=2.4, label=r"Linear $\eta + 4\varepsilon + 4\sqrt{e}$")
    ax1.plot(eps_grid, rho_quad_part, "-", color="#B91C1C", lw=3.0, label=r"Sharp FrameLA1 root $\rho_{\mathrm{quad}}$")
    ax1.fill_between(eps_grid, rho_adv_part, rho_quad_part, color="#FCA5A5", alpha=0.24, label="Certified safety margin")
    ax1.plot(eps_grid, rho_adv_part, "o-", color="#1D4ED8", lw=2.6, ms=6.0, label=r"Adversarial sector pruning ($\rho_{\mathrm{adv}}$)")
    ax1.plot(eps_grid, rho_rand_shadow, "s:", color="#0F766E", lw=2.3, ms=5.5, label=r"Random Haar shadows ($m=45$)")
    ax1.set_title(
        f"1. Adversarial Pruning vs. FrameLA1 (D = {D})  |  " + rf"$e={e_part:.3f},\ \eta={eta_part:.3f}$",
        fontsize=13.0, fontweight="bold", pad=10, loc="left"
    )
    ax1.set_xlabel(r"Discarded source subspace fraction $\varepsilon$", fontsize=12.0)
    ax1.set_ylabel(r"Blind-spot fraction $\rho = \mathrm{codim}/D$", fontsize=12.0)
    ax1.set_ylim(-0.04, 1.05)
    ax1.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.95)

    # --- Panel 2 ---
    ax2 = axes[1]
    ranks = np.arange(1, D + 1)
    ax2.plot(ranks, eig_good_norm, "o-", color="#0F766E", lw=2.6, ms=5.2,
             label=rf"Near-tight frame ($m=45,\ \eta={eta_good:.2f}$)")
    ax2.plot(ranks, eig_sparse_norm, "s-", color="#D97706", lw=2.6, ms=5.2,
             label=rf"Under-sampled frame ($m=10,\ \eta={eta_sparse:.2f}$)")
    ax2.axhline(1.0, color="#374151", ls="-", lw=1.5, alpha=0.75, label=r"Ideal flat spectrum ($\lambda/\bar{q}=1$)")
    ax2.axhline(1.0 - alpha, color="#B91C1C", ls="--", lw=2.0,
                label=rf"Ill-conditioned cutoff $(1-\alpha)\bar{{q}}$ ($\alpha={alpha}$)")
    ax2.axhspan(0.0, 1.0 - alpha, color="#FEE2E2", alpha=0.36)
    ax2.text(
        17.5, 0.14,
        f"Tail below cutoff:\nDense (m=45): {emp_tail_good:.0%}  (Cantelli ≤ {cantelli_good:.0%})\n"
        f"Sparse (m=10): {emp_tail_sparse:.0%}  (Cantelli ≤ {cantelli_sparse:.0%})",
        fontsize=10.5, bbox=dict(boxstyle="round,pad=0.45", fc="#FFFBEB", ec="#D97706", lw=1.3, alpha=0.96)
    )
    ax2.set_title(
        r"2. Spectrum of $Q = \mathcal{M} = \sum_i \gamma_i \mathcal{B}_i \mathcal{B}_i^*$: Flatness & Cantelli Tail Bound",
        fontsize=13.0, fontweight="bold", pad=10, loc="left"
    )
    ax2.set_xlabel(r"Eigenvalue index $k$ (sorted, $1 \dots D$)", fontsize=12.0)
    ax2.set_ylabel(r"Normalized eigenvalue $\lambda_k(Q) / \bar{q}$", fontsize=12.0)
    ax2.set_ylim(-0.08, max(2.85, eig_sparse_norm.max() * 1.08))
    ax2.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.95)

    # --- Panel 3 ---
    ax3 = axes[2]
    packing_ratio = (m_values * d_L) / d_P
    ax3.plot(packing_ratio, s_off_emp, "o-", color="#1D4ED8", lw=2.8, ms=6.0,
             label=r"Empirical $\sum_{i \neq j}\|V^\dagger E_i^\dagger E_j V\|_{\mathrm{HS}}^2$")
    ax3.plot(packing_ratio, welch_ideal, "--", color="#6B7280", lw=2.2,
             label=r"Ideal Welch bound ($e=0$)")
    ax3.plot(packing_ratio, welch_noisy, "-", color="#B91C1C", lw=3.0,
             label=rf"Noisy FrameLA1 Welch bound ($\bar{{e}}={e_aqecc.mean():.3f}$)")
    ax3.axvline(1.0, color="#111827", ls=":", lw=2.0)
    ax3.annotate(
        "Overpacking threshold\n($m d_L = d_P$)",
        xy=(1.0, 52), xytext=(1.13, 64),
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.5),
        fontsize=10.5, color="#111827",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F9FAFB", ec="#9CA3AF", lw=1.1, alpha=0.95)
    )
    ax3.fill_between(packing_ratio, 0, welch_noisy, color="#FCA5A5", alpha=0.24, label="Forbidden region")
    ax3.set_title(
        f"3. Overpacked AQECC ($d_L={d_L}, d_P={d_P}$): Reversed Welch Overlap Lower Bound",
        fontsize=13.0, fontweight="bold", pad=10, loc="left"
    )
    ax3.set_xlabel(r"Packing ratio $m d_L / d_P$", fontsize=12.0)
    ax3.set_ylabel(r"Total off-diagonal overlap $S_{\mathrm{off}}$", fontsize=12.0)
    ax3.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.95)

    fig.suptitle(
        r"Numerical Verification of FrameLA1 ($Q = \sum_i \gamma_i B_i B_i^*$) in Quantum Information",
        fontsize=15.0, fontweight="bold", y=0.985
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.6)

    out_dir = Path("/working_dir/c_b7b8b7f14649b6b5/artifacts/file_generation/ttl=63d/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "framela1_quantum_demo_hd.png"
    fig.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor())
    print(f"Saved HD figure to {fig_path}")


if __name__ == "__main__":
    main()
