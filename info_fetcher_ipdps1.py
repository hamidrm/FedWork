import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_pickle_file(filename):
    """Load a dictionary from a pickle file."""
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def _to_numpy_1d(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().reshape(-1).numpy()

    if isinstance(a, (list, tuple)):
        out = []
        for v in a:
            if isinstance(v, torch.Tensor):
                out.append(v.detach().cpu().item() if v.numel()==1
                        else v.detach().cpu().numpy().reshape(-1)[0])
            else:
                out.append(float(v))
        return np.asarray(out, dtype=float).reshape(-1)

    return np.asarray(a).reshape(-1)
    
def pareto_front_2d_indices(x, y):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size != y.size:
            raise ValueError(f"x and y must have same length, got {x.size} and {y.size}")

        sx = 1
        sy = -1

        xt = x * sx
        yt = y * sy

        order = np.argsort(xt, kind="mergesort")
        y_sorted = yt[order]
        running_best = np.minimum.accumulate(y_sorted)
        keep_sorted = y_sorted <= np.r_[np.inf, running_best[:-1]]
        return order[keep_sorted]

def get_acc_on_fixed_budget(x, y, budget):
        x = _to_numpy_1d(x)
        y = _to_numpy_1d(y)

        min = len(y) if len(x) > len(y) else len(x)
        x = x[:min]
        y = y[:min]
        

        idx = pareto_front_2d_indices(x, y)
        xf, yf = x[idx], y[idx]
        order = np.argsort(xf * 1)
        
        xf = xf[order]
        yf = yf[order]
        
        
        return np.interp(budget, xf, yf)


    
def main():
    number_of_clients = 10
    budget = 250000000
    #budget = 50000000
    running_path = os.getcwd()
    
    pathmnist_paths = ["fedalaq_cifar100_resnet18_comm_1_0"]
    #pathmnist_paths = ["fedalaq_pathmnist_d1_0", "fedalaq_pathmnist_d100_0"]
    for path in pathmnist_paths:
        filepath = os.path.join(running_path, "output")
        filepath = os.path.join(filepath, path)
        files = [f for f in os.listdir(filepath) if f.endswith(".data")]
        print(f"{path}:")
        for file in files:
            print(f"\t{file}:")
            file = os.path.join(filepath, file)
            data = load_pickle_file(file)

            probes_vars = data["var_values"]
            value_acc = probes_vars["EvaluationAccuracy"]
            value_upl = probes_vars["ServerTotalRecvBytes"]
            
            y = [ye[2] for ye in value_acc]
            x = [xe[2] for xe in value_upl]
                        
            acc = get_acc_on_fixed_budget(x, y, budget * number_of_clients)
            print(f"\t\tAcc. on budget {budget} is: {acc}")

if __name__ == "__main__":
    main()
