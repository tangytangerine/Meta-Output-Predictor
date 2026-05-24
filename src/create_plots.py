import logging
import os
import glob
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytorch_lightning as pl
import torch
if torch.xpu.is_available(): import intel_extension_for_pytorch as ipex

from core import Config, training
from models import GPT2
from dyn_models import apply_kf, generate_lti_sample, generate_changing_lti_sample, generate_drone_sample, apply_ekf_drone
from datasources import FilterDataset, DataModuleWrapper
from utils import RLS, plot_errs
import matplotlib.pyplot as plt
import numpy as np

import concurrent.futures
import multiprocessing

device = "" 
if torch.cuda.is_available():
    device = "cuda"
elif torch.xpu.is_available():
    device = "xpu"
else: 
    device = "cpu"
    
logger = logging.getLogger(__name__)
config = Config()
config.parse_args()

# from create_plots_all import latest_ckpt

# model = GPT2.load_from_checkpoint("../outputs/GPT2/Train_04_07/checkpoints/step=10000.ckpt",
#                                   n_dims_in=config.n_dims_in, n_positions=config.n_positions,
#                                   n_dims_out=config.n_dims_out, n_embd=config.n_embd,
#                                   n_layer=config.n_layer, n_head=config.n_head).eval().to(device)

# model2 = GPT2.load_from_checkpoint("../outputs/GPT2/Train_04_07/checkpoints/step=10000.ckpt",
#                                   n_dims_in=config.n_dims_in, n_positions=config.n_positions,
#                                   n_dims_out=config.n_dims_out, n_embd=config.n_embd,
#                                   n_layer=config.n_layer, n_head=config.n_head).eval().to(device)

# model3 = GPT2.load_from_checkpoint("../outputs/GPT2/Train_04_07_Gaußmean_025_0/checkpoints/step=10000.ckpt",
#                                   n_dims_in=config.n_dims_in, n_positions=config.n_positions,
#                                   n_dims_out=config.n_dims_out, n_embd=config.n_embd,
#                                   n_layer=config.n_layer, n_head=config.n_head).eval().to(device)

# model4 = GPT2.load_from_checkpoint("../outputs/GPT2/Train_04_07_Gaußmean_025_1/checkpoints/step=10000.ckpt",
#                                   n_dims_in=config.n_dims_in, n_positions=config.n_positions,
#                                   n_dims_out=config.n_dims_out, n_embd=config.n_embd,
#                                   n_layer=config.n_layer, n_head=config.n_head).eval().to(device)

# model5 = GPT2.load_from_checkpoint("../outputs/GPT2/Train_04_07_Gaußmean_neg025_1/checkpoints/step=10000.ckpt",
#                                   n_dims_in=config.n_dims_in, n_positions=config.n_positions,
#                                   n_dims_out=config.n_dims_out, n_embd=config.n_embd,
#                                   n_layer=config.n_layer, n_head=config.n_head).eval().to(device)
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

model = GPT2.load_from_checkpoint(latest_ckpt(),
                                   n_dims_in=config.n_dims_in, n_positions=config.n_positions,
                                   n_dims_out=config.n_dims_out, n_embd=config.n_embd,
                                   n_layer=config.n_layer, n_head=config.n_head).eval().to(device)

def processys(ys, n):
    ys = ys[:,n:,:]
    for i in range(n):
        ys = np.concatenate([ys, ys[:,-1:,:]], axis=1)
    return ys

exs, ys, sim_objs, us = [], [], [], []
exs_cl, ys_cl = [], []
for i in range(1000): 
    if config.dataset_typ == "drone": 
        m, l, J = (2, 10), (11, 15), (1, 5) 
        sim_obj, entry = generate_drone_sample(config.n_positions, m, l, J) 
        us.append(entry["actions"]) 
    else: 
        if config.changing: 
            sim_obj, entry = generate_changing_lti_sample(config.n_positions, config.nx, config.ny, n_noise=config.n_noise)
            us.append(entry["inputs"])
        else: 
            sim_obj, entry = generate_lti_sample(config.dataset_typ, config.n_positions, config.nx, config.ny, sigma_w=config.sigma_w, sigma_v=config.sigma_w, n_noise=config.n_noise)
            us.append(entry["inputs"])
    exs.append(entry["states"])
    ys.append(entry["obs"])
    exs_cl.append(entry["statesCL"])
    ys_cl.append(entry["obsCL"])
    sim_objs.append(sim_obj) 
          
exs = np.array(exs)      
ys = np.array(ys) 
us = np.array(us)
exs_cl, ys_cl = np.array(exs_cl), np.array(ys_cl)
# ys_cut = processys(ys, 50)
    
with torch.no_grad():
    I = exs[:, :-1]
    if config.dataset_typ == "drone":
        I = np.concatenate([I, us], axis=-1)

    if config.changing: # True and config.dataset_typ != "drone": #
        preds_tf = model.predict_ar(exs[:, :-1])
        # preds_tf2 = model2.predict_ar(exs[:, :-1])
        # preds_tf3 = model3.predict_ar(ys[:, :-1])
        # preds_tf4 = model4.predict_ar(ys[:, :-1])
        # preds_tf5 = model5.predict_ar(ys[:, :-1])
        # preds_tf_cut = model.predict_ar(ys_cut[:, :-1])
    else:
        _, preds_tf = model.predict_step({"xs":torch.from_numpy(I).to(device)})
        preds_tf = preds_tf["preds"].cpu().numpy()
        preds_tf = np.concatenate([np.zeros((preds_tf.shape[0],1,preds_tf.shape[-1])),preds_tf], axis=1)
        
        # _, preds_tf2 = model2.predict_step({"xs":torch.from_numpy(I).to(device)})
        # preds_tf2 = preds_tf2["preds"].cpu().numpy()
        # preds_tf2 = np.concatenate([np.zeros((preds_tf2.shape[0],1,preds_tf2.shape[-1])),preds_tf2], axis=1)
        
        # _, preds_tf3 = model3.predict_step({"xs":torch.from_numpy(I).to(device)})
        # preds_tf3 = preds_tf3["preds"].cpu().numpy()
        # preds_tf3 = np.concatenate([np.zeros((preds_tf3.shape[0],1,preds_tf3.shape[-1])),preds_tf3], axis=1)
        
        # _, preds_tf4 = model4.predict_step({"xs":torch.from_numpy(I).to(device)})
        # preds_tf4 = preds_tf4["preds"].cpu().numpy()
        # preds_tf4 = np.concatenate([np.zeros((preds_tf4.shape[0],1,preds_tf4.shape[-1])),preds_tf4], axis=1)
        
        # _, preds_tf5 = model5.predict_step({"xs":torch.from_numpy(I).to(device)})
        # preds_tf5 = preds_tf5["preds"].cpu().numpy()
        # preds_tf5 = np.concatenate([np.zeros((preds_tf5.shape[0],1,preds_tf5.shape[-1])),preds_tf5], axis=1)
errs_tf = np.linalg.norm((exs-preds_tf), axis=-1)
# errs_tf2 = np.linalg.norm((exs-preds_tf2), axis=-1)
# errs_tf3 = np.linalg.norm((ys-preds_tf3), axis=-1)
# errs_tf4 = np.linalg.norm((ys-preds_tf4), axis=-1)
# errs_tf5 = np.linalg.norm((ys-preds_tf5), axis=-1)

err_tf_cl = np.linalg.norm((exs_cl[:,:-1,:]-(preds_tf[:,:-1,:]-us)), axis=-1)

# errs_tf_cut = np.linalg.norm((ys_cut-preds_tf_cut), axis=-1)

# n_noise = config.n_noise
# if config.dataset_typ == "drone":
#     preds_kf = np.array([apply_ekf_drone(dsim, _ys, _us) for dsim, _ys, _us in zip(sim_objs, ys, us)])
# else:
#     preds_kf = np.array([apply_kf(fsim, _ys, sigma_w=fsim.sigma_w*np.sqrt(n_noise), sigma_v=fsim.sigma_v*np.sqrt(n_noise)) for fsim, _ys in zip(sim_objs, ys[:, :-1])])
# errs_kf = np.linalg.norm((ys-preds_kf), axis=-1)

err_lss = [errs_tf]
names = ["MOP"]

# err_lss = [errs_tf, errs_tf2, errs_tf3, errs_tf4, errs_tf5]
# names = ["μ = 0", "μ = N(0, 1)", "μ = 0.25", "μ = N(0.25, 1)", "μ = N(-0.25, 1)"]

# if config.dataset_typ != "drone":
#     preds_rls = []
#     for _ys in ys:
#         ls = [np.zeros(config.ny)]
#         rls = RLS(config.nx, config.ny)
#         for i in range(len(_ys)-1):
#             if i < 2:
#                 ls.append(_ys[i])
#             else:
#                 rls.add_data(_ys[i-2:i].flatten(), _ys[i])
#                 ls.append(rls.predict(_ys[i-1:i+1].flatten()))

#         preds_rls.append(ls)
#     preds_rls = np.array(preds_rls)
#     errs_rls = np.linalg.norm(ys-preds_rls, axis=-1)
#     err_lss.append(errs_rls)
#     names.append("OLS")


fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(111)
ax.set_title("Time-Varying A", fontsize=32)
plot_errs(names, err_lss, ax=ax, shade=config.dataset_typ != "drone")

fig = plt.figure(figsize=(15,9))
ax = fig.add_subplot(111)
ax.set_title("Time-Varying A", fontsize=32)
plot_errs(["Subtracted Control"], [err_tf_cl], ax=ax, shade=config.dataset_typ != "drone")
# os.makedirs("../figures", exist_ok=True)
# fig.savefig(f"../figures/{config.dataset_typ}" + ("-changing" if config.changing else ""))