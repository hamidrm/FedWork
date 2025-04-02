import matplotlib.pyplot as plt
import numpy as np
import torch

class Plotter:

    def plot_hypervolume2d(self, x, y, label, reference_point, style_str, style_index):
        

        if len(y) > len(x):
            y_o_x = int(len(y) / len(x))
            y = [y[yi * y_o_x].cpu().numpy() for yi in range(int(len(y) / y_o_x))]
        

        if len(x) > len(y):
            x_o_y = int(len(x) / len(y))
            x = [x[xi * x_o_y].cpu().numpy() for xi in range(int(len(x) / x_o_y))]

        y = [1.0-y[yi] for yi in range(int(len(y)))]

        x = np.array(x)
        y = np.array(y)
        
        sorted_indices = np.argsort(x)
        x_cor = x[sorted_indices]
        y_cor = y[sorted_indices]

        # Hypervolume calculation
        hv = 0.0
        last_x = reference_point[0]
        for xi, yi in zip(x_cor, y_cor):
            width = abs(xi - last_x)
            height = abs(yi - reference_point[1])
            hv += width * height
            last_x = xi

        # Extract style parameters safely
        colors = linestyles = linewidths = markers = fill_colors = None
        alpha = 0.1  # default alpha
        
        if "colors=" in style_str:
            colors = style_str.split("colors=")[1].split(";")[0].strip().split(",")

        if "alpha=" in style_str:
            alpha_vals = style_str.split("alpha=")[1].split(";")[0].strip().split(",")
            if len(alpha_vals) > style_index:
                alpha = float(alpha_vals[style_index])

        if "fill_color=" in style_str:
            fill_colors = style_str.split("fill_color=")[1].split(";")[0].strip().split(",")

        if "linestyles=" in style_str:
            linestyles = style_str.split("linestyles=")[1].split(";")[0].strip().split(",")

        if "linewidths=" in style_str:
            linewidths = list(map(float, style_str.split("linewidths=")[1].split(";")[0].strip().split(",")))

        if "markers=" in style_str:
            markers = style_str.split("markers=")[1].split(";")[0].strip().split(",")

        # Safe extraction with defaults
        c = colors[style_index] if colors and len(colors) > style_index else 'blue'
        ls = linestyles[style_index] if linestyles and len(linestyles) > style_index else '-'
        lw = linewidths[style_index] if linewidths and len(linewidths) > style_index else 2
        m = markers[style_index] if markers and len(markers) > style_index else None
        fc = fill_colors[style_index] if fill_colors and len(fill_colors) > style_index else c

        # Plot Pareto front
        #plt.plot(x_cor, y_cor, color=c, linestyle=ls, linewidth=lw, marker=m, label=label)

        # Fill hypervolume area
        plt.fill_between(
            x_cor,
            y_cor,
            reference_point[0],
            step='post',
            color=fc,
            alpha=alpha,
            label=f"HV {label}: {hv:.4f}"
        )

        # Display hypervolume (placed adaptively)
        text_x = np.mean(x_cor)
        text_y = np.mean(y_cor)
        #plt.text(text_x, text_y, f"HV {label}: {hv:.4f}", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))


    def plot_begin(self, style_str):
        plt.figure()

        plt.style.use('default')
        if "style=" in style_str:
            style = style_str.split("style=")[1].split(";")[0].strip()
            plt.style.use(style)
        plt.rcParams["font.family"] = "Noto Mono"


    def plot_end(self, x_axis_title, y_axis_title, fig_caption, output_path):

        plt.xlabel(x_axis_title,fontsize=10, family='Noto Mono')
        plt.ylabel(y_axis_title,fontsize=10, family='Noto Mono')
        plt.title(fig_caption,fontsize=10, family='Noto Mono')
        plt.legend(frameon=True,fontsize="small")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()
        
    def plot(self, x, y, label, style_str, style_index):
        
        colors = None
        linestyles = None
        linewidths = None
        markers = None
        
        if "colors=" in style_str:
            colors = style_str.split("colors=")[1].split(";")[0].strip().split(",")
        
        if "linestyles=" in style_str:
            linestyles = style_str.split("linestyles=")[1].split(";")[0].strip().split(",")
        
        if "linewidths=" in style_str:
            linewidths = list(map(float, style_str.split("linewidths=")[1].split(";")[0].strip().split(",")))
    
        if "markers=" in style_str:
            markers = style_str.split("markers=")[1].split(";")[0].strip().split(",")
        
        c = None
        ls = None
        lw = None
        m = None

        if colors is not None:
            if len(colors) > style_index:
                c = colors[style_index]

        if linestyles is not None:
            if len(linestyles) > style_index:
                ls = linestyles[style_index]

        if linewidths is not None:
            if len(linewidths) > style_index:
                lw = linewidths[style_index]
        
        if markers is not None:
            if len(markers) > style_index:
                m = markers[style_index]
                

        # Convert x and y element-wise to ensure they are NumPy arrays
        def convert_to_numpy(data):
            if isinstance(data, torch.Tensor):  # Handle tensor directly
                if data.is_cuda:
                    data = data.cpu()  # Move to CPU
                return data.detach().numpy()  # Convert to NumPy
            elif isinstance(data, (list, tuple)):  # Handle list or tuple recursively
                return np.array([convert_to_numpy(item) for item in data])
            return np.asarray(data)  # Fallback for other types

        # Ensure x and y are fully converted
        x = convert_to_numpy(x)
        y = convert_to_numpy(y)


        plt.plot(x, y, label=label, color=c, linestyle=ls, linewidth=lw, marker=m)

