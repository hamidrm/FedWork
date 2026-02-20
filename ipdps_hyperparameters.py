import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Circle

def plot_5x5_from_tuples(
    records,
    fedavg_features,
    save_path,
    r_range=(0.12, 0.40),
    lw_range=(1.0, 5.0),
    cmap_name="Blues",
    figsize=(9, 9),
    dpi=300,
    show_grid=True,
):
    # Coerce to numpy-friendly rows and handle torch scalars
    rows = []
    for a, t, acc, sz, auc in records:
        rows.append((float(a), float(t), float(acc), float(sz), float(auc)))
    arr = np.array(rows, dtype=float)

    # Build grid axes (expect 5×5)
    alphas = np.unique(arr[:, 0])
    taus   = np.unique(arr[:, 1])
    if len(alphas) != 5 or len(taus) != 5:
        raise ValueError(f"Expected 5 unique alphas and 5 unique taus, got {len(alphas)} and {len(taus)}.")

    alphas_sorted = np.sort(alphas)          # y-axis (top = largest)
    taus_sorted   = np.sort(taus)            # x-axis (left = smallest)
    alpha_to_row = {a: i for i, a in enumerate(alphas_sorted[::-1])}  # reverse so big alpha at top
    tau_to_col   = {t: j for j, t in enumerate(taus_sorted)}

    # Allocate grid
    acc  = np.full((5, 5), np.nan)
    size = np.full((5, 5), np.nan)
    auc  = np.full((5, 5), np.nan)

    for a, t, ac, sz, au in arr:
        i = alpha_to_row[a]
        j = tau_to_col[t]
        acc[i, j]  = ac
        size[i, j] = sz
        auc[i, j]  = au

    if np.isnan(acc).any():
        missing = np.argwhere(np.isnan(acc))
        raise ValueError(f"Grid not full; missing {len(missing)} cells.")

    acc_min, acc_max = float(np.min(acc)),  float(np.max(acc))
    size_min, size_max = float(np.min(size)), float(np.max(size))
    auc_min, auc_max = float(np.min(auc)),   float(np.max(auc))

    acc_norm = Normalize(vmin=acc_min, vmax=acc_max)
    cmap = get_cmap(cmap_name)

    def linmap(x, a, b, A, B):
        d = (b - a) if (b - a) != 0 else 1.0
        return A + (np.clip(x, a, b) - a) * (B - A) / d

    radii  = linmap(size, size_min, size_max, r_range[0], r_range[1])
    if auc_max == auc_min:
        lwidth = np.full_like(auc, lw_range[0])
    else:
        lwidth = linmap(auc, auc_min, auc_max, lw_range[0], lw_range[1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal")

    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in taus_sorted])
    ax.set_yticklabels([f"{v:g}" for v in alphas_sorted[::-1]])
    ax.set_xlabel("τ"); ax.set_ylabel("α")

    if show_grid:
        for g in range(36):
            a = g / 6
            color = "gray" if g % 6 != 0 else "black"
            ax.plot([0, 6], [a, a], linewidth=0.6, color=color)
            ax.plot([a, a], [0, 6], linewidth=0.6, color=color)

    for i in range(5):
        for j in range(5):
            x, y = j + 0.5, i + 0.5
            color = cmap(acc_norm(acc[i, j]))
            c = Circle((x, y), float(radii[i, j]), facecolor=color,
                       edgecolor="black", linewidth=float(0.05))
            ax.add_patch(c)
            c = Circle((x, y), float(radii[i, j]+0.05), facecolor="none",
                       edgecolor="red", linewidth=float(lwidth[i, j]))
            ax.add_patch(c)

    sm = ScalarMappable(norm=acc_norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Top-1 Accuracy")

    fig.tight_layout()
    plt.style.use('default')




    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def load_pickle_file(filename):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

running_path = os.getcwd()
path = "fedalaq_hyperparams_compare_parhmnist2"
filepath = os.path.join(running_path, "output")
filepath = os.path.join(filepath, path)
files = [f for f in os.listdir(filepath) if f.endswith(".data")]
print(f"{path}:")


features = []
features2 = []
features3 = []

for file in files:
    if len(str.split(file, "_probes_data.data")[0].split("FedALAQ_")) == 1:
        continue
    
    combination = str.split(file, "_probes_data.data")[0].split("FedALAQ_")[1]
    

    alpha = float(combination.split("_")[-1])
    tau = float(file.split("_"+str(int(alpha))+"_probes_data.data")[0].split("FedALAQ_")[1].replace("_", "."))
        
    file = os.path.join(filepath, file)
    data = load_pickle_file(file)

    probes_vars = data["var_values"]
    value_acc = probes_vars["EvaluationAccuracy"]
    value_upl = probes_vars["ServerTotalRecvBytes"]
    value_mia = probes_vars["MIA_TPRS_0_01"]
    
    acc = sum([value_acc[i][2].to("cpu") for i in range(-4,-1)]) / 3
    upl = sum([value_upl[i][2] for i in range(-4,-1)]) / 3
    mia = value_mia[-1][2]
    
    features.append((alpha, tau, acc, upl, mia))
    
    value_mia = probes_vars["MIA_TPRS_0_1"]
    mia = value_mia[-1][2]
    features2.append((alpha, tau, acc, upl, mia))

    value_mia = probes_vars["MIA_AUC"]
    mia = value_mia[-1][2]
    features3.append((alpha, tau, acc, upl, mia))
    print(f"\t{file} loaded.")

# file = os.path.join(filepath, "FedAvg_probes_data.data")
# data = load_pickle_file(file)

# probes_vars = data["var_values"]
# value_acc = probes_vars["EvaluationAccuracy"]
# value_upl = probes_vars["ServerTotalRecvBytes"]
# value_mia = probes_vars["MIA_AUC"]

# acc = value_acc[-2][2].to("cpu")
# upl = value_upl[-2][2]
# mia = value_mia[-1][2]

fedavg_features = (acc, upl, mia)
plot_5x5_from_tuples(features, fedavg_features, path + "_fpr1p" + ".pdf", show_grid=True)
plot_5x5_from_tuples(features2, fedavg_features, path + "_fpr10p" + ".pdf", show_grid=True)
plot_5x5_from_tuples(features3, fedavg_features, path + "_auc" + ".pdf", show_grid=True)



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Before pulling from Hyperion:
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Circle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.patches import Circle

def plot_5x5_from_tuples(
    records,
    fedavg_features,
    save_path,
    r_range=(0.12, 0.40),
    lw_range=(1.0, 5.0),
    cmap_name="Blues",
    figsize=(9, 9),
    dpi=300,
    show_grid=True,
):
    # Coerce to numpy-friendly rows and handle torch scalars
    rows = []
    for a, t, acc, sz, auc in records:
        rows.append((float(a), float(t), float(acc), float(sz), float(auc)))
    arr = np.array(rows, dtype=float)

    # Build grid axes (expect 5×5)
    alphas = np.unique(arr[:, 0])
    taus   = np.unique(arr[:, 1])
    if len(alphas) != 5 or len(taus) != 5:
        raise ValueError(f"Expected 5 unique alphas and 5 unique taus, got {len(alphas)} and {len(taus)}.")

    alphas_sorted = np.sort(alphas)          # y-axis (top = largest)
    taus_sorted   = np.sort(taus)            # x-axis (left = smallest)
    alpha_to_row = {a: i for i, a in enumerate(alphas_sorted[::-1])}  # reverse so big alpha at top
    tau_to_col   = {t: j for j, t in enumerate(taus_sorted)}

    # Allocate grid
    acc  = np.full((5, 5), np.nan)
    size = np.full((5, 5), np.nan)
    auc  = np.full((5, 5), np.nan)

    for a, t, ac, sz, au in arr:
        i = alpha_to_row[a]
        j = tau_to_col[t]
        acc[i, j]  = ac
        size[i, j] = sz
        auc[i, j]  = au

    if np.isnan(acc).any():
        missing = np.argwhere(np.isnan(acc))
        raise ValueError(f"Grid not full; missing {len(missing)} cells.")

    acc_min, acc_max = float(np.min(acc)),  float(np.max(acc))
    size_min, size_max = float(np.min(size)), float(np.max(size))
    auc_min, auc_max = float(np.min(auc)),   float(np.max(auc))

    acc_norm = Normalize(vmin=acc_min, vmax=acc_max)
    cmap = get_cmap(cmap_name)

    def linmap(x, a, b, A, B):
        d = (b - a) if (b - a) != 0 else 1.0
        return A + (np.clip(x, a, b) - a) * (B - A) / d

    radii  = linmap(size, size_min, size_max, r_range[0], r_range[1])
    if auc_max == auc_min:
        lwidth = np.full_like(auc, lw_range[0])
    else:
        lwidth = linmap(auc, auc_min, auc_max, lw_range[0], lw_range[1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 5); ax.set_ylim(0, 5); ax.set_aspect("equal")

    ax.set_xticks(np.arange(5) + 0.5)
    ax.set_yticks(np.arange(5) + 0.5)
    ax.set_xticklabels([f"{v:g}" for v in taus_sorted])
    ax.set_yticklabels([f"{v:g}" for v in alphas_sorted[::-1]])
    ax.set_xlabel("τ"); ax.set_ylabel("α")

    if show_grid:
        for g in range(36):
            a = g / 6
            color = "gray" if g % 6 != 0 else "black"
            ax.plot([0, 6], [a, a], linewidth=0.6, color=color)
            ax.plot([a, a], [0, 6], linewidth=0.6, color=color)

    for i in range(5):
        for j in range(5):
            x, y = j + 0.5, i + 0.5
            color = cmap(acc_norm(acc[i, j]))
            c = Circle((x, y), float(radii[i, j]), facecolor=color,
                       edgecolor="black", linewidth=float(0.05))
            ax.add_patch(c)
            c = Circle((x, y), float(radii[i, j]+0.05), facecolor="none",
                       edgecolor="red", linewidth=float(lwidth[i, j]))
            ax.add_patch(c)

    sm = ScalarMappable(norm=acc_norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Top-1 Accuracy")

    fig.tight_layout()
    plt.style.use('default')




    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 18
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return save_path


def load_pickle_file(filename):
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

running_path = os.getcwd()
path = "fedalaq_hyperparams_compare"
filepath = os.path.join(running_path, "output")
filepath = os.path.join(filepath, path)
files = [f for f in os.listdir(filepath) if f.endswith(".data")]
print(f"{path}:")


features = []
features2 = []
features3 = []

for file in files:
    if len(str.split(file, "_probes_data.data")[0].split("FedALAQ_")) == 1:
        continue
    
    combination = str.split(file, "_probes_data.data")[0].split("FedALAQ_")[1]
    

    alpha = float(combination.split("_")[-1])
    tau = float(file.split("_"+str(int(alpha))+"_probes_data.data")[0].split("FedALAQ_")[1].replace("_", "."))
        
    file = os.path.join(filepath, file)
    data = load_pickle_file(file)

    probes_vars = data["var_values"]
    value_acc = probes_vars["EvaluationAccuracy"]
    value_upl = probes_vars["ServerTotalRecvBytes"]
    value_mia = probes_vars["MIA_TPRS_0_01"]
    
    acc = sum([value_acc[i][2].to("cpu") for i in range(-4,-1)]) / 3
    upl = sum([value_upl[i][2] for i in range(-4,-1)]) / 3
    mia = value_mia[-1][2]
    
    features.append((alpha, tau, acc, upl, mia))
    
    value_mia = probes_vars["MIA_TPRS_0_1"]
    mia = value_mia[-1][2]
    features2.append((alpha, tau, acc, upl, mia))

    value_mia = probes_vars["MIA_AUC"]
    mia = value_mia[-1][2]
    features3.append((alpha, tau, acc, upl, mia))
    print(f"\t{file} loaded.")

# file = os.path.join(filepath, "FedAvg_probes_data.data")
# data = load_pickle_file(file)

# probes_vars = data["var_values"]
# value_acc = probes_vars["EvaluationAccuracy"]
# value_upl = probes_vars["ServerTotalRecvBytes"]
# value_mia = probes_vars["MIA_AUC"]

# acc = value_acc[-2][2].to("cpu")
# upl = value_upl[-2][2]
# mia = value_mia[-1][2]

fedavg_features = (acc, upl, mia)
plot_5x5_from_tuples(features, fedavg_features, path + "_fpr1p" + ".pdf", show_grid=True)
plot_5x5_from_tuples(features2, fedavg_features, path + "_fpr10p" + ".pdf", show_grid=True)
plot_5x5_from_tuples(features3, fedavg_features, path + "_auc" + ".pdf", show_grid=True)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""