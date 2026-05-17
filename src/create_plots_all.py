# src/create_plots_all.py
import os
import glob
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from core import Config
from models import GPT2
from dyn_models import (
    apply_kf,
    generate_lti_sample,
    generate_changing_lti_sample,
    generate_drone_sample,
    apply_ekf_drone,
)
from utils import RLS, plot_errs

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Helpers
# ---------------------------

def latest_ckpt(search_root=None):
    """Find the most recent .ckpt if config.ckpt_path is empty or a directory."""
    search_dirs = []
    if search_root:
        if os.path.isfile(search_root) and search_root.endswith(".ckpt"):
            return search_root
        if os.path.isdir(search_root):
            search_dirs.append(search_root)
    # default: search ../outputs/**/checkpoints/*.ckpt relative to this file
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    search_dirs.append(default_root)

    candidates = []
    for root in search_dirs:
        candidates += glob.glob(os.path.join(root, "**", "checkpoints", "*.ckpt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(
            "No .ckpt found. Set Config.ckpt_path to a checkpoint file or train first."
        )
    return max(candidates, key=os.path.getmtime)


def evaluate_once(config, model, dataset_typ, *, changing=False, n_noise=1, n_samples=1000):
    """Replicates your create_plots.py logic once for a specific setting."""
    ys, sim_objs, us = [], [], []

    for _ in range(n_samples):
        if dataset_typ == "drone":
            sim_obj, entry = generate_drone_sample(config.n_positions)
            us.append(entry["actions"])
        else:
            if changing:
                sim_obj, entry = generate_changing_lti_sample(
                    config.n_positions, config.nx, config.ny, n_noise=n_noise
                )
            else:
                sim_obj, entry = generate_lti_sample(
                    dataset_typ, config.n_positions, config.nx, config.ny, n_noise=n_noise
                )
        ys.append(entry["obs"])
        sim_objs.append(sim_obj)

    ys = np.array(ys)
    us = np.array(us) if dataset_typ == "drone" else None

    # ---- MOP predictions
    with torch.no_grad():
        I = ys[:, :-1]
        if dataset_typ == "drone" and us is not None:
            I = np.concatenate([I, us], axis=-1)

        if changing:
            # (autoregressive prediction path used by original script)
            preds_tf = model.predict_ar(ys[:, :-1])
        else:
            _, out = model.predict_step({"xs": torch.from_numpy(I).to(device)})
            preds_tf = out["preds"].cpu().numpy()
            preds_tf = np.concatenate(
                [np.zeros((preds_tf.shape[0], 1, preds_tf.shape[-1])), preds_tf],
                axis=1
            )
    errs_tf = np.linalg.norm(ys - preds_tf, axis=-1)

    # ---- Baseline (KF/EKF)
    if dataset_typ == "drone":
        preds_kf = np.array([
            apply_ekf_drone(dsim, _ys, _us)
            for dsim, _ys, _us in zip(sim_objs, ys, us)
        ])
    else:
        preds_kf = np.array([
            apply_kf(fsim, _ys,
                     sigma_w=fsim.sigma_w * np.sqrt(n_noise),
                     sigma_v=fsim.sigma_v * np.sqrt(n_noise))
            for fsim, _ys in zip(sim_objs, ys[:, :-1])
        ])
    errs_kf = np.linalg.norm(ys - preds_kf, axis=-1)

    # ---- Optional OLS/RLS (non-drone)
    names = ["Kalman", "MOP"]
    err_lss = [errs_kf, errs_tf]

    if dataset_typ != "drone":
        preds_rls = []
        for _ys in ys:
            ls = [np.zeros(config.ny)]
            rls = RLS(config.nx, config.ny)
            for t in range(len(_ys) - 1):
                if t < 2:
                    ls.append(_ys[t])
                else:
                    rls.add_data(_ys[t-2:t].flatten(), _ys[t])
                    ls.append(rls.predict(_ys[t-1:t+1].flatten()))
            preds_rls.append(ls)
        preds_rls = np.array(preds_rls)
        errs_rls = np.linalg.norm(ys - preds_rls, axis=-1)
        err_lss.append(errs_rls)
        names.append("OLS")

    return names, err_lss




def save_plot(names, err_lss, out_path, shade=True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    plot_errs(names, err_lss, ax=ax, shade=shade)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def _ratio_curve(config, model, dataset_typ: str, n_samples: int = 1000, n_noise_val: int = 1):
    """
    Returns ratio_t (len T):  E_i ||y_t - yhat_t||_2  /  E_i ||y_t - yhat_t^{KF}||_2
    Average over sequences only; keep time dimension.
    """
    ys, sim_objs = [], []
    for _ in range(n_samples):
        fsim, entry = generate_lti_sample(
            dataset_typ,
            config.n_positions,
            config.nx,
            config.ny,
            n_noise=n_noise_val,      # 1 => white noise
        )
        ys.append(entry["obs"])
        sim_objs.append(fsim)

    ys = np.array(ys, dtype=np.float32)        # [N, T, ny]
    N, T, ny = ys.shape

    # MOP predictions (no retrain)
    with torch.no_grad():
        I = ys[:, :-1]
        _, out = model.predict_step({"xs": torch.from_numpy(I).to(device)})
        preds_tf = out["preds"].cpu().numpy()  # [N, T-1, ny]
        preds_tf = np.concatenate(
            [np.zeros((N, 1, preds_tf.shape[-1]), dtype=preds_tf.dtype), preds_tf],
            axis=1
        )                                      # [N, T, ny]

    # If checkpoint head != ny, clip to ny (safety)
    if preds_tf.shape[-1] != ny:
        preds_tf = preds_tf[..., :ny]

    # KF baseline with generator’s std devs
    preds_kf = np.array([
        apply_kf(
            fsim, _ys,
            sigma_w=fsim.sigma_w,
            sigma_v=fsim.sigma_v
        )
        for fsim, _ys in zip(sim_objs, ys[:, :-1])
    ], dtype=np.float32)                        # [N, T, ny]

    # Ratio per time step (avg over sequences only)
    err_mop_t = np.linalg.norm(ys - preds_tf, axis=-1).mean(axis=0)  # [T]
    err_kf_t  = np.linalg.norm(ys - preds_kf, axis=-1).mean(axis=0)  # [T]
    ratio_t = err_mop_t / np.maximum(err_kf_t, 1e-12)
    return ratio_t



# ---------------------------
# Main (generate all figures)
# ---------------------------

def main():
    config = Config()
    config.parse_args()

    # Load model (use config.ckpt_path or auto-pick latest)
    ckpt = config.ckpt_path or latest_ckpt(None)
    model = GPT2.load_from_checkpoint(
        ckpt,
        n_dims_in=config.n_dims_in, n_positions=config.n_positions,
        n_dims_out=config.n_dims_out, n_embd=config.n_embd,
        n_layer=config.n_layer, n_head=config.n_head,
    ).eval().to(device)
    print("Loaded checkpoint:", ckpt)

    figs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
    os.makedirs(figs_dir, exist_ok=True)

    # ------------- Fig. 2(a): linear i.i.d. ----------------
    #names, err_lss = evaluate_once(config, model, dataset_typ="ypred",
    #                               changing=False, n_noise=1, n_samples=1000)
    #save_plot(names, err_lss, os.path.join(figs_dir, "ypred.png"), shade=True)

    # ------------- Fig. 2(b): colored noise ----------------
    # In their configs, noniid uses n_noise=5
    #names, err_lss = evaluate_once(config, model, dataset_typ="noniid",
    #                               changing=False, n_noise=5, n_samples=1000)
    #save_plot(names, err_lss, os.path.join(figs_dir, "noniid.png"), shade=True)

    # ------------- Fig. 2(c): time-varying dynamics -------
    #names, err_lss = evaluate_once(config, model, dataset_typ="ypred",
    #                               changing=True, n_noise=1, n_samples=1000)
    #save_plot(names, err_lss, os.path.join(figs_dir, "ypred-changing.png"), shade=True)

    # ------------- Fig. 3: drone (EKF baseline) -----------
    #names, err_lss = evaluate_once(config, model, dataset_typ="drone",
    #                               changing=False, n_noise=1, n_samples=1000)
    #save_plot(names, err_lss, os.path.join(figs_dir, "drone.png"), shade=False)

        # ------------- Fig. 4: denseA vs upperTriA — error/optimal-error vs time (MOP/KF) -------------
    ratio_dense = _ratio_curve(config, model, dataset_typ="ypred",     n_samples=1000, n_noise_val=1)
    ratio_upper = _ratio_curve(config, model, dataset_typ="upperTriA", n_samples=1000, n_noise_val=1)

    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111)
    ts = np.arange(1, ratio_dense.shape[0] + 1)
    ax.plot(ts, ratio_dense, linewidth=2.0, label="dense A")
    ax.plot(ts, ratio_upper, linewidth=2.0, label="upperTriA")
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"error / optimal error  $\frac{\mathbb{E}_i\|y_t-\hat y_t\|_2}{\mathbb{E}_i\|y_t-\hat y^{KF}_t\|_2}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path = os.path.join(figs_dir, "fig4_dense_vs_upperTriA_error_over_opt.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    # plt.close(fig)
    print(f"Saved: {out_path}")


    # ---------- Fig. 5: error / optimal-error vs time (sweep σ² = 0.1..1.5) ----------
    # y-axis:  [ E_i ||y_t - ŷ_t||_2 ] / [ E_i ||y_t - ŷ_t^{KF}||_2 ]  (avg over sequences only)
    # x-axis:  time step t = 1..T
    # sweep:   σ²_w = σ²_v in {0.1, 0.2, ..., 1.5}

    sigma2_sweep = [round(s, 1) for s in np.arange(0.1, 1.5 + 1e-9, 0.1)]
    #sigma2_sweep = [0.05, 0.1, 0.25, 0.5, 1]
    ratio_curves, labels = [], []
    ratio_kfs_curves = []

    N_SAMPLES = 1000  # number of test trajectories
    DATASET = "ypred" # linear i.i.d. setup for Fig. 5
    COLORED = False   # set True to test colored noise (n_noise>1); paper uses i.i.d.

    for sigma2 in sigma2_sweep:
        sigma = float(np.sqrt(sigma2))  # convert variance -> std

        # --- generate test set at this σ² (white noise unless COLORED=True) ---
        n_noise_val = (5 if COLORED else 1)
        ys, sim_objs = [], []
        for _ in range(N_SAMPLES):
            fsim, entry = generate_lti_sample(
                DATASET,
                config.n_positions,
                config.nx,
                config.ny,
                sigma_w=sigma,
                sigma_v=sigma,
                n_noise=n_noise_val,
            )
            ys.append(entry["obs"])
            sim_objs.append(fsim)
        ys = np.array(ys)                      # [N, T, ny]
        T = ys.shape[1]

        # --- MOP predictions (same checkpoint; no retrain) ---
        with torch.no_grad():
            I = ys[:, :-1]                     # inputs to predictor
            _, out = model.predict_step({"xs": torch.from_numpy(I).to(device)})
            preds_tf = out["preds"].cpu().numpy()             # [N, T-1, ny]
            preds_tf = np.concatenate(
                [np.zeros((preds_tf.shape[0], 1, preds_tf.shape[-1])), preds_tf],
                axis=1,
            )                                   # [N, T, ny]

        # --- KF baseline at matching σ (std dev) ---
        preds_kf = np.array([
            apply_kf(
                fsim, _ys,
                sigma_w=sigma,                 # pass std directly
                sigma_v=sigma
            )
            for fsim, _ys in zip(sim_objs, ys[:, :-1])
        ])                                      # [N, T, ny]
        preds_kf_2 = np.array([
            apply_kf(
                fsim, _ys,
                sigma_w=1e-1,                 # pass std directly
                sigma_v=1e-1
            )
            for fsim, _ys in zip(sim_objs, ys[:, :-1])
        ])                                      # [N, T, ny]

        # --- per-time Euclidean errors, average over sequences only ---
        err_mop_t = np.linalg.norm(ys - preds_tf, axis=-1).mean(axis=0)  # [T]
        err_kf_t  = np.linalg.norm(ys - preds_kf, axis=-1).mean(axis=0)  # [T]
        err_kf_2_t  = np.linalg.norm(ys - preds_kf_2, axis=-1).mean(axis=0)  # [T]
        eps = 1e-12
        ratio_t = err_mop_t / np.maximum(err_kf_t, eps)                  # [T]
        ratio_kfs = err_kf_t / np.maximum(err_kf_2_t, eps)                  # [T]



        ratio_kfs_curves.append(ratio_kfs)
        ratio_curves.append(ratio_t)
        labels.append(f"σ²={sigma2:.1f}")

    # --- plot: error/optimal-error vs time; one curve per σ² ---
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ts = np.arange(1, T + 1)
    for r, lab in zip(ratio_curves, labels):
        ax.plot(ts, r, label=lab, linewidth=1.5)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"error / optimal error  $\frac{\mathbb{E}_i\|y_t-\hat y_t\|_2}{\mathbb{E}_i\|y_t-\hat y^{KF}_t\|_2}$")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    out_path = os.path.join(figs_dir, "fig5_error_over_opt_vs_time.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    # plt.close(fig)
    print(f"Saved: {out_path}")
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ts = np.arange(1, T + 1)
    for r, lab in zip(ratio_kfs_curves, labels):
        ax.plot(ts, r, label=lab, linewidth=1.5)
    ax.set_xlabel("time step $t$")
    ax.set_ylabel(r"Optimal KF/Mismatched KF")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    out_path = os.path.join(figs_dir, "fig5_kf_error_vs_time.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    # plt.close(fig)
    print(f"Saved: {out_path}")




if __name__ == "__main__":
    main()
