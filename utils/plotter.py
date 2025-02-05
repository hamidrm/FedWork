import matplotlib.pyplot as plt
import numpy as np
import torch

class Plotter:

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

