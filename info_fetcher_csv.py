import io
import os
import pickle
import torch
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


class DeviceUnpickler(pickle.Unpickler):
    def __init__(self, file_obj, map_location="cpu"):
        super().__init__(file_obj)
        self.map_location = map_location

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data),
                map_location=self.map_location,
                weights_only=False,
            )

        return super().find_class(module, name)


def load_pickle_with_device(file_path, map_location="cpu"):
    with open(file_path, "rb") as file_obj:
        return DeviceUnpickler(
            file_obj,
            map_location=map_location,
        ).load()
MOVING_AVERAGE_WINDOW = 50


def to_scalar(value):
    """Convert a scalar tensor or numeric value to a Python number."""
    if torch.is_tensor(value):
        return value.detach().cpu().item()

    return value


def trailing_moving_average(values, window):
    """
    Calculate a trailing moving average while retaining the original length.

    For the first window-1 entries, all available preceding values are used.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return values

    window = min(window, len(values))

    cumulative_sum = np.cumsum(
        np.insert(values, 0, 0.0)
    )

    result = np.empty_like(values, dtype=float)

    # Initial partial windows
    result[:window - 1] = (
        cumulative_sum[1:window]
        / np.arange(1, window)
    )

    # Full-sized windows
    result[window - 1:] = (
        cumulative_sum[window:]
        - cumulative_sum[:-window]
    ) / window

    return result


def extract_accuracy_data(data):
    """
    Extract:
        tuple[1] -> communication round
        tuple[2] -> evaluation accuracy
    """
    evaluation_accuracy = data["var_values"]["EvaluationAccuracy"]

    x_values = []
    y_values = []

    for entry in evaluation_accuracy:
        if not isinstance(entry, (tuple, list)) or len(entry) < 3:
            print(f"Skipping unexpected entry: {entry}")
            continue

        communication_round = to_scalar(entry[1])
        accuracy = to_scalar(entry[2])

        x_values.append(float(communication_round))
        y_values.append(float(accuracy))

    x_values = np.asarray(x_values)
    y_values = np.asarray(y_values)

    # Sort all points by communication round.
    order = np.argsort(x_values)

    return x_values[order], y_values[order]


def main():
    running_path = os.getcwd()

    pathmnist_paths = [
        "fedalaq_pathmnist_comm_300_0_4P_1000",
    ]

    # Replace the filenames here if the other two names differ.
    plot_configs = [
        {
            "filename": "FedAvgQ8_probes_data.data",
            "label": "FedAvgQ8",
            "color": "black",
        },
        {
            "filename": "FedAvg_probes_data.data",
            "label": "FedAvg",
            "color": "blue",
        },
        {
            "filename": "FedALAQ_0_3_100_probes_data.data",
            "label": "FedALAQ",
            "color": "green",
        },
    ]

    for path in pathmnist_paths:
        filepath = os.path.join(
            running_path,
            "output",
            path,
        )

        print(f"{path}:")

        fig, ax = plt.subplots(figsize=(6, 4.6))

        for config in plot_configs:
            filename = config["filename"]
            label = config["label"]
            color = config["color"]

            file_path = os.path.join(filepath, filename)

            if not os.path.isfile(file_path):
                print(f"\tFile not found: {filename}")
                continue

            print(f"\tLoading {filename}")

            data = load_pickle_with_device(
                file_path,
                map_location="cuda:0",
            )

            x_values, y_values = extract_accuracy_data(data)

            if len(x_values) == 0:
                print(f"\tNo accuracy data found in {filename}")
                continue

            moving_average = trailing_moving_average(
                y_values,
                window=MOVING_AVERAGE_WINDOW,
            )

            # Raw accuracy values
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=0.7,
                alpha=0.1,
                label="_nolegend_",
                zorder=1,
            )

            # Moving-average curve
            ax.plot(
                x_values,
                moving_average,
                color=color,
                linewidth=1.0,
                alpha=1.0,
                label=label,
                zorder=2,
            )

        ax.set_xlabel(
            "Communication Round",
            fontsize=15,
            fontfamily="serif",
        )

        ax.set_ylabel(
            "Top-1 Accuracy",
            fontsize=15,
            fontfamily="serif",
        )

        ax.tick_params(
            axis="both",
            labelsize=13,
        )

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.7,
            alpha=0.7,
        )

        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            fontsize=12,
            frameon=True,
        )

        fig.tight_layout()

        output_file = os.path.join(
            filepath,
            "evaluation_accuracy_moving_average.pdf",
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"\tPlot saved to: {output_file}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()