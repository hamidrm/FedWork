import matplotlib.pyplot as plt
import numpy as np
import torch

class Plotter:
    def _sense_to_sign(self, sense):
        """'min' -> +1, 'max' -> -1 (so we can turn any objective into 'min')."""
        s = str(sense).lower()
        if s.startswith("min"): return 1.0
        if s.startswith("max"): return -1.0
        raise ValueError("sense must be 'min' or 'max'")

    def pareto_front_2d_indices(self, x, y, senses=("min", "max"), strict=True):
        """
        Return indices of the non-dominated points for two objectives.

        By default: x is minimized, y is maximized (typical 'cost vs quality').
        Set strict=False to keep ties on the front.
        """
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        if x.size != y.size:
            raise ValueError(f"x and y must have same length, got {x.size} and {y.size}")

        sx = self._sense_to_sign(senses[0])  # +1=min, -1=max
        sy = self._sense_to_sign(senses[1])

        # Transform so BOTH objectives become "minimize"
        xt = x * sx
        yt = y * sy

        # Sort by xt ascending; scan, keeping new best (lowest) yt
        order = np.argsort(xt, kind="mergesort")
        y_sorted = yt[order]
        running_best = np.minimum.accumulate(y_sorted)
        if strict:
            keep_sorted = y_sorted < np.r_[np.inf, running_best[:-1]]
        else:
            keep_sorted = y_sorted <= np.r_[np.inf, running_best[:-1]]
        return order[keep_sorted]

    def _to_numpy_1d(self, a):
        # torch tensor -> numpy
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().reshape(-1).numpy()
        # list/tuple (possibly of tensors) -> numpy
        if isinstance(a, (list, tuple)):
            out = []
            for v in a:
                if isinstance(v, torch.Tensor):
                    out.append(v.detach().cpu().item() if v.numel()==1
                            else v.detach().cpu().numpy().reshape(-1)[0])
                else:
                    out.append(float(v))
            return np.asarray(out, dtype=float).reshape(-1)
        # everything else -> numpy
        return np.asarray(a).reshape(-1)

    def plot_tradeoff_2d(self, x, y, label, style_str, style_index, 
                         senses=("min","max"), extend=False, number_of_rounds=-1, extend_samples=False):
        """
        Scatter all points + overlay the Pareto front.
        senses: ('min'|'max', 'min'|'max') for (x, y).
        """

        x = self._to_numpy_1d(x)
        y = self._to_numpy_1d(y)

        
        
        
        if extend_samples:
            if len(x) > len(y):
                n, m = len(x), len(y)
                y = np.repeat(y, n // m)
            else:
                n, m = len(y), len(x)
                x = np.repeat(x, n // m)
            
        min = len(y) if len(x) > len(y) else len(x)
        min = min if number_of_rounds == -1 else number_of_rounds
        x = x[:min]
        y = y[:min]
        

        # front
        idx = self.pareto_front_2d_indices(x, y, senses=senses)
        xf, yf = x[idx], y[idx]

        # draw the front ordered along x (respecting its sense)
        sx = self._sense_to_sign(senses[0])
        order = np.argsort(xf * sx)

        # Extract style parameters safely
        colors = linestyles = linewidths = markers = fill_colors = None
        alpha = 0.1  # default alpha
        
        if "colors=" in style_str:
            colors = style_str.split("colors=")[1].split(";")[0].strip().split(",")

        if "alpha=" in style_str:
            alpha_vals = style_str.split("alpha=")[1].split(";")[0].strip().split(",")
            if len(alpha_vals) > style_index:
                alpha = float(alpha_vals[style_index])

        if "linestyles=" in style_str:
            linestyles = style_str.split("linestyles=")[1].split(";")[0].strip().split(",")

        if "linewidths=" in style_str:
            linewidths = list(map(float, style_str.split("linewidths=")[1].split(";")[0].strip().split(",")))

        if "markers=" in style_str:
            markers = style_str.split("markers=")[1].split(";")[0].strip().split(",")
        show_points_str = None
        if "show_points=" in style_str:
            show_points_str = style_str.split("show_points=")[1].split(";")[0].strip()
        
        show_points = False
        if show_points_str is not None:
            if show_points_str == "on":
                show_points = True
            elif show_points_str != "off":
                logger.warninig("Invalid value for 'show_points' in the defined style.")
    
        # Safe extraction with defaults
        c = colors[style_index] if colors and len(colors) > style_index else 'blue'
        ls = linestyles[style_index] if linestyles and len(linestyles) > style_index else '-'
        lw = linewidths[style_index] if linewidths and len(linewidths) > style_index else 2
        m = markers[style_index] if markers and len(markers) > style_index else None

        # scatter all points (faint)
        if show_points:
            plt.scatter(x, y, s=1, color=c, alpha=0.2)


        plt.plot(xf[order], yf[order], color=c, linestyle=ls, linewidth=lw, marker=m, label=label)
    
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
        figure_size_wh = None
        

        if "figure_size=" in style_str:
            figure_size_wh = style_str.split("figure_size=")[1].split(";")[0].strip().split(",")

        if figure_size_wh is None:
            plt.figure()
        else:
            plt.figure(figsize=(float(figure_size_wh[0]), float(figure_size_wh[1])))
        if "style=" in style_str:
            style = style_str.split("style=")[1].split(";")[0].strip()
            plt.style.use(style)
        
        
        plt.style.use('default')




        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12

    def plot_end(self, x_axis_title, y_axis_title, fig_caption, style_str, output_path):

        legend_pos = "best"
        y_axis_scale = None
        legend_font_size = "small"
        title_font_size = 10
        axis_font_size = 10
        font_family = 'serif'
        legend_status = None
        legend_ncol = 1
        bbox_to_anchor_xy = None
        
        if "legend_status=" in style_str:
            legend_status = style_str.split("legend_status=")[1].split(";")[0].strip()

        if "legend_ncol=" in style_str:
            legend_ncol = int(style_str.split("legend_ncol=")[1].split(";")[0].strip())
        
        if "bbox_to_anchor=" in style_str:
            bbox_to_anchor_xy = style_str.split("bbox_to_anchor=")[1].split(";")[0].strip().split(",")
                
        if "legend_pos=" in style_str:
            legend_pos = style_str.split("legend_pos=")[1].split(";")[0].strip()
            legend_pos = legend_pos.replace('_', ' ')
        
        if "y_axis_scale=" in style_str:
            y_axis_scale = style_str.split("y_axis_scale=")[1].split(";")[0].strip().split(",")
      
        if "legend_font_size=" in style_str:
            legend_font_size = style_str.split("legend_font_size=")[1].split(";")[0].strip()
              
        if "title_font_size=" in style_str:
            title_font_size = style_str.split("title_font_size=")[1].split(";")[0].strip()

        if "axis_font_size=" in style_str:
            axis_font_size = style_str.split("axis_font_size=")[1].split(";")[0].strip()

        if "font_family=" in style_str:
            font_family = style_str.split("font_family=")[1].split(";")[0].strip()


        plt.xlabel(x_axis_title,fontsize=axis_font_size, family=font_family)
        plt.ylabel(y_axis_title,fontsize=axis_font_size, family=font_family)
        plt.title(fig_caption,fontsize=title_font_size, family=font_family)
        plt.xticks(fontsize=axis_font_size)
        plt.yticks(fontsize=axis_font_size)        


        if legend_status is not None:
            if legend_status == "on":
                if bbox_to_anchor_xy is None:
                    plt.legend(frameon=True,fontsize=legend_font_size, loc=legend_pos, ncol=legend_ncol)
                else:
                    plt.legend(frameon=True,fontsize=legend_font_size, bbox_to_anchor=(float(bbox_to_anchor_xy[0]), float(bbox_to_anchor_xy[1])), loc=legend_pos, ncol=legend_ncol)
            elif legend_status != "off":
                logger.warninig("Invalid value for 'legend_status' in the defined style.")
        else:
            if bbox_to_anchor_xy is None:
                plt.legend(frameon=True,fontsize=legend_font_size, loc=legend_pos, ncol=legend_ncol)
            else:
                plt.legend(frameon=True,fontsize=legend_font_size, bbox_to_anchor=(bbox_to_anchor_xy[0], bbox_to_anchor_xy[1]), loc=legend_pos, ncol=legend_ncol)
            
        if y_axis_scale is not None:
            plt.ylim(float(y_axis_scale[0]), float(y_axis_scale[1]))

        

        plt.grid(True, which='major', linestyle='--', linewidth=0.4, alpha=0.6)
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()
        
    def plot(self, x, y, label, style_str, style_index):
        
        colors = None
        linestyles = None
        linewidths = None
        markers = None
        colors_map = None
        points_radius = None
        points_alpha = None
        show_points_str = None
        if "linewidths=" in style_str:
            linewidths = list(map(float, style_str.split("linewidths=")[1].split(";")[0].strip().split(",")))
    
        if "points_radius=" in style_str:
            points_radius = list(map(float, style_str.split("points_radius=")[1].split(";")[0].strip().split(",")))
            
        if "points_alpha=" in style_str:
            points_alpha = list(map(float, style_str.split("points_alpha=")[1].split(";")[0].strip().split(",")))
            
        if "colors_map=" in style_str:
            colors_map = style_str.split("colors_map=")[1].split(";")[0].strip()
            num_colors = len(linewidths)
            if colors_map == "tab10":
                colors = plt.cm.tab10(np.linspace(0, 1, num_colors))
                c = colors[style_index]
            elif colors_map == "Set2":
                colors = plt.cm.Set2(np.linspace(0, 1, num_colors))
                c = colors[style_index]
            elif colors_map == "husl":
                colors = plt.cm.husl(np.linspace(0, 1, num_colors))
                c = colors[style_index]
            else:
                logger.warning("Invalid color map!")
            
        
        if "colors=" in style_str:
            colors = style_str.split("colors=")[1].split(";")[0].strip().split(",")
        
        if "linestyles=" in style_str:
            linestyles = style_str.split("linestyles=")[1].split(";")[0].strip().split(",")
        
        if "markers=" in style_str:
            markers = style_str.split("markers=")[1].split(";")[0].strip().split(",")
        
        if "show_points=" in style_str:
            show_points_str = style_str.split("show_points=")[1].split(";")[0].strip()
        
        show_points = False
        if show_points_str is not None:
            if show_points_str == "on":
                show_points = True
            elif show_points_str != "off":
                logger.warninig("Invalid value for 'show_points' in the defined style.")
                
        c = None
        ls = None
        lw = None
        m = None
        r = None
        a = None
        
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
                
        if points_radius is not None:
            if len(points_radius) > style_index:
                r = points_radius[style_index]
        
        if points_alpha is not None:
            if len(points_alpha) > style_index:
                a = points_alpha[style_index]
                
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

        if show_points:
            plt.scatter(x, y, s=r, color=c, alpha=a)
        plt.plot(x, y, alpha=0.9, label=label, color=c, linestyle=ls, linewidth=lw, marker=m)
        
    def plot_envelope(self, x, Y, label, style_str, style_index):

        # ---------- Parse style_str ----------
        colors = None
        linestyles = None
        linewidths = None
        markers = None
        colors_map = None

        alpha_fill = 0.25
        show_minmax = True
        sort_x = False
        fill_intervals = None

        def _get_val(key, conv=str):
            if f"{key}=" in style_str:
                return conv(style_str.split(f"{key}=")[1].split(";")[0].strip())
            return None

        if "linewidths=" in style_str:
            linewidths = list(map(float, style_str.split("linewidths=")[1].split(";")[0].strip().split(",")))

        colors_map = _get_val("colors_map")
        # We'll set `c` later after we know how many styles we need; init to None
        c = None
        if "colors=" in style_str:
            colors = style_str.split("colors=")[1].split(";")[0].strip().split(",")
        if "linestyles=" in style_str:
            linestyles = style_str.split("linestyles=")[1].split(";")[0].strip().split(",")
        if "markers=" in style_str:
            markers = style_str.split("markers=")[1].split(";")[0].strip().split(",")

        v = _get_val("alpha_fill", float)
        if v is not None:
            alpha_fill = float(v)

        v = _get_val("show_minmax")
        if v is not None:
            show_minmax = str(v).lower() in ("1", "true", "yes", "y")

        v = _get_val("sort_x")
        if v is not None:
            sort_x = str(v).lower() in ("1", "true", "yes", "y")

        # Parse fill_range= a:b|c:d (or commas inside)
        v = _get_val("fill_range")
        if v:
            s = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
            parts = re.split(r"\|", s)
            intervals = []
            for p in parts:
                if not p:
                    continue
                if ":" in p:
                    a, b = p.split(":")[:2]
                elif "," in p:
                    a, b = p.split(",")[:2]
                else:
                    continue
                a, b = float(a), float(b)
                if b < a:
                    a, b = b, a
                intervals.append((a, b))
            if intervals:
                fill_intervals = intervals

        # Resolve per-style items
        ls = linestyles[style_index] if (linestyles is not None and len(linestyles) > style_index) else None
        lw = linewidths[style_index] if (linewidths is not None and len(linewidths) > style_index) else None
        m  = markers[style_index]    if (markers    is not None and len(markers)    > style_index) else None

        # ---------- Convert inputs ----------
        def _to_np_1d(a):
            if isinstance(a, torch.Tensor):
                a = a.detach()
                if a.is_cuda:
                    a = a.cpu()
                return a.flatten().numpy()
            elif isinstance(a, (list, tuple)):
                return np.asarray(a, dtype=float).ravel()
            else:
                return np.asarray(a, dtype=float).ravel()

        def _curve_to_np_1d(a):
            # each curve in Y
            if isinstance(a, torch.Tensor):
                a = a.detach()
                if a.is_cuda:
                    a = a.cpu()
                return a.flatten().numpy()
            elif isinstance(a, (list, tuple, np.ndarray)):
                return np.asarray(a, dtype=float).ravel()
            else:
                raise TypeError("Each element of Y must be a 1D tensor/list/array.")

        x_np = _to_np_1d(x)
        Y_np_list = [_curve_to_np_1d(y) for y in Y]
        n = x_np.size
        for i, y_arr in enumerate(Y_np_list):
            if y_arr.size != n:
                raise ValueError(f"Y[{i}] length {y_arr.size} != len(x) {n}")

        Y_np = np.vstack(Y_np_list)  # (k, n)

        # ---------- Optional sort ----------
        if sort_x:
            order = np.argsort(x_np)
            x_np = x_np[order]
            Y_np = Y_np[:, order]

        # ---------- Envelope computations ----------
        y_min = np.nanmin(Y_np, axis=0)
        y_max = np.nanmax(Y_np, axis=0)
        y_avg = (y_min + y_max) / 2.0

        # ---------- Color resolution ----------
        # If explicit colors list provided, pick by style_index
        if colors is not None:
            if isinstance(colors, np.ndarray):
                # colormap already sampled into array
                if colors.shape[0] > style_index:
                    c = colors[style_index]
            else:
                if len(colors) > style_index:
                    c = colors[style_index]

        # If still no color and a colors_map is requested, sample it
        if c is None and colors_map is not None:
            try:
                num_colors = (len(linewidths) if linewidths is not None else max(1, style_index + 1))
                if colors_map in ("tab10", "Set2") or hasattr(plt.cm, colors_map):
                    cmap = getattr(plt.cm, colors_map) if hasattr(plt.cm, colors_map) else plt.get_cmap(colors_map)
                    sampled = cmap(np.linspace(0, 1, num_colors))
                    if sampled.shape[0] > style_index:
                        c = sampled[style_index]
                elif colors_map == "husl" and hasattr(plt.cm, "husl"):
                    sampled = plt.cm.husl(np.linspace(0, 1, num_colors))
                    c = sampled[style_index]
                else:
                    logger.warning("Invalid color map!")
            except Exception:
                logger.warning("Invalid color map!")
                c = None

        base_color = c if c is not None else 'tab:blue'
        avg_lw = lw if lw is not None else 2.5

        # ---------- Plot ----------
        # Faint min/max outlines
        if show_minmax:
            plt.plot(x_np, y_min, color='0.65', lw=1, label=f'{label} min')
            plt.plot(x_np, y_max, color='0.65', lw=1, label=f'{label} max')

        # Build mask for filling
        if fill_intervals is None:
            mask = np.ones_like(x_np, dtype=bool)
            fill_lbl = f'{label} envelope'
        else:
            mask = np.zeros_like(x_np, dtype=bool)
            for x0, x1 in fill_intervals:
                mask |= (x_np >= x0) & (x_np <= x1)
            fill_lbl = f'{label} envelope (restricted)'

        # Fill between min/max only where mask is True
        plt.fill_between(x_np, y_min, y_max, where=mask, interpolate=True,
                        alpha=alpha_fill, color=base_color, label=fill_lbl)

        # Average line (primary)
        plt.plot(x_np, y_avg, label=f'{label} avg',
             color=base_color, linestyle=ls, linewidth=avg_lw, marker=m, alpha=0.95)

