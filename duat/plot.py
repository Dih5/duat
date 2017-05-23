# -*- coding: UTF-8 -*-
"""Plot OSIRIS-generated data."""

from __future__ import print_function
from __future__ import division

from glob import glob
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter

from duat.common import ensure_dir_exists, human_order_key, MPCaller, Call, logger


def _is_latex(s):
    """

    Decide if a string is a LaTeX expression.

    Args:
        s (str): The string to test.

    Returns:
        bool: True if it seems a LaTeX expression, False otherwise.

    """
    if not s:  # Empty string is not LaTeX
        return False
    return re.match("^.(_.)?$", s) or "\\" in s  # If in the form x, x_i, or if there is a backlash (command)


def _improve_latex(s):
    """Improve an OSIRIS-generated latex string using common rules."""
    # Descriptive subindexes in roman
    s2 = s.replace(r"\omega_p", r"\omega_{\mathrm{p}}")
    s2 = s2.replace("m_e", r"m_{\mathrm{e}}")
    # "Arbitrary units" in roman
    s2 = s2.replace("a.u.", r"\mathrm{a.u.}")
    return s2


class Diagnostic:
    """
    A OSIRIS diagnostic.

    Attributes:
        data_path (str): Path to the directory were the data is stored.
        data_name (str): A friendly name for the data.
        units (str): The name of the unit the magnitude is measured in.
        dt (float): The time step between snapshots of a consecutive number.
        t_0 (float): The time of the first snapshot.
        time_units (str): The name of the unit of time.
        file_list (list of str): List of h5 files, one per time snapshot.
        snapshot_list (list of int): List of integers identifying the snapshots. Multiply by dt to get time.
        keys (list of str): Names of the datasets in the Diagnostic, given in human order.
        axes (list of dict): Info of each axis in the Diagnostic.
        datasets_as_axis(dict): Info of datasets if treated as axes. WARNING: Only used with energy bins.
        shape (tuple): A tuple with:

            * list: The number of grid dimensions.
            * int: The number of datasets (excluding axes definition).
            * int: The number of snapshots in time.
            
    Note:
        The axes list is provided in the order of the numpy convention for arrays. This is the opposite of order used
        to label the axes in the hdf5 files. For example in a 2d array the first axes will be the labeled as AXIS2, and
        the second will be AXIS1. Unless the user makes use of other external tools to read the data, he/she can safely
        ignore this note.
            
    """

    def __init__(self, data_path):
        """
        Create a Diagnostic instance.

        Args:
            data_path (str): Path of the directory containing the diagnostic data.
        
        Raises:
            ValueError: If there is no data in `data_path`.
            
        """
        self.data_path = data_path
        self.file_list = glob(os.path.join(data_path, "*.h5"))
        if not self.file_list:
            raise ValueError("No diagnostic data found in %s" % data_path)
        self.file_list.sort(key=human_order_key)
        self.snapshot_list = list(
            map(lambda x: int((os.path.split(x)[1]).split(".h5")[0].split("-")[-1]), self.file_list))

        # Get info from first time snapshot
        with h5py.File(self.file_list[0], "r") as f:
            self.t_0 = f.attrs["TIME"][0]
            self.time_units = f.attrs["TIME UNITS"][0].decode('UTF-8')
            self.keys = self._get_keys(f)
            self.axes = self._get_axes(f, self.keys[0])
            self.units = f[self.keys[0]].attrs["UNITS"][0].decode('UTF-8')
            if len(self.keys) > 1:
                # Take a general name from the global attribute
                self.data_name = f.attrs["NAME"][0].decode('UTF-8')
                # Construct an axes-like object representing the dataset
                self.datasets_as_axis = {}
                # Guess name
                name = f[self.keys[0]].attrs["TAG"][0].decode('UTF-8')  # E.g., "Energy <=    0.00000000"
                name = re.match("(.*?) <", name).group(1)  # TODO: Catch errors, check syntax
                self.datasets_as_axis["NAME"] = name
                self.datasets_as_axis["LONG_NAME"] = name
                # Guess values
                dataset_axes = []
                number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'  # A pattern matching floats
                for d in self.keys:
                    tag = f[d].attrs["TAG"][0].decode('UTF-8')
                    # tag is, e.g., '   0.00000000      < Energy <=    5.00000007E-02'
                    values = list(map(float, re.findall(number_pattern, tag)))
                    dataset_axes.append(sum(values) / len(values))  # So simple inequalities are represented somehow
                    # TODO: First and last datasets are not bounded intervals. Their only finite point is used as the x coordinate, which must be understood with caution
                self.datasets_as_axis["LIST"] = dataset_axes
                self.datasets_as_axis["MIN"] = dataset_axes[0]
                self.datasets_as_axis["MAX"] = dataset_axes[-1]
                self.datasets_as_axis["UNITS"] = "?"
            else:
                # Take a specific name from the dataset
                self.data_name = f[self.keys[0]].attrs["LONG_NAME"][0].decode('UTF-8')
                # No axes-like object in this case
                self.datasets_as_axis = None

        if len(self.file_list) < 2:
            self.dt = 0
        else:
            with h5py.File(self.file_list[1], "r") as f:
                self.dt = f.attrs["TIME"][0] / self.snapshot_list[1] - self.t_0

        self.shape = ([len(x["LIST"]) for x in self.axes], len(self.keys), len(self.snapshot_list))

    def __repr__(self):
        return "Diagnostic<%s %s>" % (self.data_name, str(self.shape))

    @staticmethod
    def _get_keys(file):
        """Get the dataset keys from an opened file."""
        keys = list(file.keys())
        if "AXIS" not in keys:
            raise ValueError("AXIS group not found.")
        keys.remove("AXIS")
        keys.sort(key=human_order_key)
        return keys

    @staticmethod
    def _get_axes(file, dataset_key=None):
        """Get the axes info."""
        if dataset_key is None:
            dataset_key = Diagnostic._get_keys(file)[0]
        axes = []
        for i, axis in enumerate(reversed(list(file["AXIS"].keys()))):
            ax = file["AXIS"][axis]
            data = {}
            for d in ["LONG_NAME", "UNITS", "NAME", "TYPE"]:
                data[d] = ax.attrs[d][0].decode('UTF-8')
            data["MIN"], data["MAX"] = ax[:]
            # TODO: Non linear axis support
            data["LIST"] = np.linspace(data["MIN"], data["MAX"], num=file[dataset_key].shape[i])
            axes.append(data)

        return axes

    def _clean_dataset_key(self, dataset_key):
        """Return the given dataset key as `str`, using human order if `int`. Might raise error or warning."""
        if isinstance(dataset_key, int):
            dataset_key = self.keys[dataset_key]
        elif isinstance(dataset_key, str):
            if dataset_key not in self.keys:
                raise ValueError("Dataset %s does not exist in the file." % dataset_key)
        elif dataset_key is None:
            if len(self.keys) != 1:  # Warn if implicitly selecting one among others.
                logger.warning("No dataset selected when multiple are available. Plotting the first one.")
            dataset_key = self.keys[0]
        else:
            raise TypeError("Unknown dataset type: %s", type(dataset_key))
        return dataset_key

    def get_generator(self, dataset_selector=None, axes_selector=None, time_selector=None):
        """
        Get a generator providing data from the file.

        Calling this method returns a generator which, when called, will provide data for increasing times (unless
        modified by time_selector parameter). The data might be reduced either by selecting a position in an axis (or
        a dataset) or by using a function along some axis (or datasets), e.g., a sum.

        This data is provided as numpy arrays where the first axis refers to dataset coordinate (if present) and next
        to (non-reduced) axis in the order they are found in the files.

        Args:
            dataset_selector (str, int or callable): Instructions to reduce datasets. An int selects a dataset in human
                order, a str selects it by name. A function taking a list and returning a scalar can be used to reduce
                the data, e.g., sum, mean...

            axes_selector (tuple): Instructions to reduce axes data. It must be
                a tuple of the same length of the number axes or None to perform no reduction.
                Each element can be of the following types:

                    * int: Select the item in the given position.
                    * None: No reduction is performed in this axis.
                    * callable (default): Reduce the data along this axes using the given function (e.g., mean, max, sum...).


            time_selector (slice): A slice instance selecting the points in time to take.

        Returns:
            generator: A generator which provides the data.

        """
        multiple_datasets = False  # If a dataset list is going to be returned
        if dataset_selector is not None:
            if self.shape[1] == 1:
                logger.warning("Single dataset found. Ignoring the provided dataset_selector.")

                def f_dataset_selector(f):
                    return f[self.keys[0]][:]
            else:
                if isinstance(dataset_selector, int):
                    dataset_selector = self.keys[dataset_selector]

                if isinstance(dataset_selector, str):  # If it was int or str
                    def f_dataset_selector(f):
                        return f[dataset_selector][:]
                else:  # Assumed function
                    def f_dataset_selector(f):
                        return np.apply_along_axis(dataset_selector, 0, [f[key][:] for key in self.keys])

        else:
            if self.shape[1] > 1:
                multiple_datasets = True

                def f_dataset_selector(f):
                    return np.array([f[key][:] for key in self.keys])
            else:
                def f_dataset_selector(f):
                    return f[self.keys[0]][:]

        if axes_selector is not None:
            def f_axes_selector(x):
                offset = 1 if multiple_datasets else 0  # If multiple dataset, do not count its axis for reduction
                for i, sel in enumerate(axes_selector):
                    if sel is not None:
                        if isinstance(sel, int):
                            x = np.take(x, sel, axis=i - offset)
                        else:  # Assumed function
                            x = np.apply_along_axis(sel, i - offset, x)
                        offset += 1
                return x
        else:
            def f_axes_selector(x):
                return x

        if time_selector is not None and not isinstance(time_selector, slice):
            logger.warning("Invalid time_selector parameter ignored. Use a slice instead.")
            time_selector = None

        def gen():
            for file_name in (self.file_list[time_selector] if time_selector else self.file_list):
                with h5py.File(file_name, "r") as f:
                    data = f_dataset_selector(f)
                # Make sure to exit the context manager before yielding
                # h5py might accuse you of murdering identifiers if you don't!
                yield f_axes_selector(data)

        return gen()

    def get_axes(self, dataset_selector=None, axes_selector=None):
        """
        Get a dictionary with the info of the axes obtained as result of a given reduction.
        
        Args:
            dataset_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            axes_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.

        Returns:
            list of dict: Ordered list of the axes left by the reduction.
            
        """
        axes = []
        if dataset_selector is not None:
            if self.shape[1] == 1:
                logger.warning("Single dataset found. Ignoring the provided dataset_selector.")
                # Anyhow, datasets are reduced, so skip

        elif self.shape[1] > 1:
            axes.append(self.datasets_as_axis)

        if axes_selector is not None:
            for i, sel in enumerate(axes_selector):
                if sel is None:
                    axes.append(self.axes[i])
        else:
            for a in self.axes:
                axes.append(a)
        return axes

    def get_time_list(self, time_selector=None):
        """
        Get the list of times obtained as a result of a given slice.
        
        Args:
            time_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method. 

        Returns:
            :obj:`list` of :obj:`float`: The times resulting as a consequence of the slice.

        """
        if time_selector:
            # This could be improved to avoid generating unneeded values
            return [self.t_0 + self.dt * i for i in self.snapshot_list][time_selector]
        else:
            return [self.t_0 + self.dt * i for i in self.snapshot_list]

    def time_1d_animation(self, output_path=None, dataset_selector=None, axes_selector=None, time_selector=None,
                          dpi=200, fps=1, scale_mode="expand",
                          latex_label=True, interval=200):
        """
        Generate a plot of 1d data animated in time.
        
        If an output path with a suitable extension is supplied, the method will export it. Available formats are mp4
        and gif. The returned objects allow for minimal customization and representation. For example in Jupyter you
        might use `IPython.display.HTML(animation.to_html5_video())`, where `animation` is the returned `FuncAnimation`
        instance.
        
        Note:
            Exporting a high resolution animated gif with many frames might eat your RAM.

        Args:
            output_path (str): The place where the plot is saved. If "" or None, the plot is shown in matplotlib.
            dataset_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            axes_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            time_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            interval (float): Delay between frames in ms. If exporting to mp4, the fps is used instead to generate the
                              file, although the returned objects do use this value.
            dpi (int): The resolution of the frames in dots per inch (only if exporting).
            fps (int): The frames per seconds (only if exporting to mp4).
            scale_mode (str): How the scale is changed through time. Available methods are:

                * "expand": The y limits increase when needed, but they don't decrease.
                * "adjust_always": Always change the y limits to those of the data.

            latex_label (bool): Whether for use LaTeX code for the plot.
            
        Returns:
            (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`, `matplotlib.animation.FuncAnimation`):
            Objects representing the generated plot and its animation.

        """
        if output_path:
            ensure_dir_exists(os.path.dirname(output_path))
        axes = self.get_axes(dataset_selector=dataset_selector, axes_selector=axes_selector)
        if len(axes) != 1:
            raise ValueError("Expected 1 axis plot, but %d were provided" % len(axes))
        axis = axes[0]

        gen = self.get_generator(dataset_selector=dataset_selector, axes_selector=axes_selector,
                                 time_selector=time_selector)

        # Set plot labels
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        x_name = axis["LONG_NAME"]
        x_units = axis["UNITS"]
        with h5py.File(self.file_list[0], "r") as f:
            y_name = f[self.keys[0]].attrs["LONG_NAME"][0].decode('UTF-8')
            y_units = f[self.keys[0]].attrs["UNITS"][0].decode('UTF-8')
        if latex_label:
            if x_units:
                x_units = "$" + _improve_latex(x_units) + "$"
            if y_units:
                y_units = "$" + _improve_latex(y_units) + "$"
            # Names might be text or LaTeX. Try to guess
            if _is_latex(x_name):
                x_name = "$" + _improve_latex(x_name) + "$"
            if _is_latex(y_name):
                y_name = "$" + _improve_latex(y_name) + "$"

        if x_units:
            ax.set_xlabel("%s (%s)" % (x_name, x_units))
        else:
            ax.set_xlabel("%s" % (x_name,))
        if y_units:
            ax.set_ylabel("%s (%s)" % (y_name, y_units))
        else:
            ax.set_ylabel("%s" % (y_name,))

        # Plot the points
        x_min, x_max = axis["MIN"], axis["MAX"]
        plot_data, = ax.plot(axis["LIST"], next(gen))
        ax.set_xlim(x_min, x_max)

        time_list = self.get_time_list(time_selector)

        # Prepare a function for the updates
        def update(i):
            """Update the plot, returning the artists which must be redrawn."""
            try:
                new_dataset = next(gen)
            except StopIteration:
                logger.warning("Tried to add a frame to the animation, but all data was used.")
                return
            label = 't = {0}'.format(time_list[i])
            plot_data.set_ydata(new_dataset[:])
            ax.set_title(label)
            if not scale_mode:
                pass
            elif scale_mode == "expand":
                prev = ax.get_ylim()
                data_limit = [min(new_dataset), max(new_dataset)]
                ax.set_ylim(min(prev[0], data_limit[0]), max(prev[1], data_limit[1]))
            elif scale_mode == "adjust_always":
                ax.set_ylim(min(new_dataset), max(new_dataset))
            return plot_data, ax

        anim = FuncAnimation(fig, update, frames=range(1, len(time_list) - 2), interval=interval)

        if not output_path:  # "" or None
            pass
        else:
            filename = os.path.basename(output_path)
            if "." in filename:
                extension = output_path.split(".")[-1].lower()
            else:
                extension = None
            if extension == "gif":
                anim.save(output_path, dpi=dpi, writer='imagemagick')
            elif extension == "mp4":
                metadata = dict(title=os.path.split(self.data_path)[-1], artist='duat', comment=self.data_path)
                writer = FFMpegWriter(fps=fps, metadata=metadata)
                with writer.saving(fig, output_path, dpi):
                    # Iterate over frames
                    for i in range(1, len(time_list) - 1):
                        update(i)
                        writer.grab_frame()
                    # Keep showing the last frame for the fixed time
                    writer.grab_frame()
            else:
                logger.warning("Unknown extension in path %s. No output produced." % output_path)

        plt.close()

        return fig, ax, anim

    def time_1d_colormap(self, output_path=None, dataset_selector=None, axes_selector=None, time_selector=None,
                         dpi=200, latex_label=True, cmap=None, log_map=False, show=True, rasterized=True,
                         contour_plot=False, z_min=None, z_max=None):
        """
        Generate a colormap in an axis and the time.
    
        This function plots a magnitude depending on ONE spatial coordinate (hence the name) and on time as a colormap
        in the cartesian product of such a magnitude and the time.
        
        Note:
            For simple manipulation like labels or title you can make use of the returned tuple or a
            `matplotlib.pyplot.style.context`. More advanced manipulation can be done extracting the data with the
            :func:`~duat.osiris.plot.Diagnostic.get_generator` method instead.
    
        Args:
            output_path (str): The place where the plot is saved. If "" or None, the figure is not saved.
            dataset_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            axes_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            time_selector: See :func:`~duat.osiris.plot.Diagnostic.get_generator` method.
            dpi (int): The resolution of the file in dots per inch.
            latex_label (bool): Whether for use LaTeX code for the plot.
            cmap (str or `matplotlib.colors.Colormap`): The Colormap to use in the plot.
            log_map (bool): Whether the map is plot in log scale. If the log scale is too wide and values are not
                            displayed in the bar, use the z_min or z_max parameters to fix it.
            show (bool): Whether to show the plot. This is blocking if matplotlib is in non-interactive mode.
            rasterized (bool): Whether the map is rasterized. This does not apply to axes, title... Note non-rasterized
                               images with large amount of data exported to PDF might challenging to handle.
            contour_plot (bool): Whether contour lines are plot instead of the density map.
            z_min (float): Minimum value for the colormap. If None it is automatically set.
            z_max (float): Maximum value for the colormap. If None it is automatically set.
            
        Returns:
            (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`): Objects representing the generated plot.
    
        """
        if output_path:
            ensure_dir_exists(os.path.dirname(output_path))
        axes = self.get_axes(dataset_selector=dataset_selector, axes_selector=axes_selector)
        if len(axes) != 1:
            raise ValueError("Expected 1 axis plot, but %d were provided" % len(axes))
        if len(self.file_list) < 2:
            raise ValueError("Unable to plot a colormap with only one time snapshot")
        axis = axes[0]

        gen = self.get_generator(dataset_selector=dataset_selector, axes_selector=axes_selector,
                                 time_selector=time_selector)

        # Set plot labels
        fig, ax = plt.subplots()

        x_name = axis["LONG_NAME"]
        x_units = axis["UNITS"]
        y_name = "t"
        y_units = self.time_units
        title_name = self.data_name
        title_units = self.units

        if latex_label:
            if x_units:
                x_units = "$" + _improve_latex(x_units) + "$"
            if y_units:
                y_units = "$" + _improve_latex(y_units) + "$"
            if title_units:
                title_units = "$" + _improve_latex(title_units) + "$"
            # Names might be text or LaTeX. Try to guess
            if _is_latex(x_name):
                x_name = "$" + _improve_latex(x_name) + "$"
            if _is_latex(y_name):
                y_name = "$" + _improve_latex(y_name) + "$"
            if _is_latex(title_name):
                title_name = "$" + _improve_latex(title_name) + "$"

        if x_units:
            ax.set_xlabel("%s (%s)" % (x_name, x_units))
        else:
            ax.set_xlabel("%s" % (x_name,))
        if y_units:
            ax.set_ylabel("%s (%s)" % (y_name, y_units))
        else:
            ax.set_ylabel("%s" % (y_name,))

        time_list = self.get_time_list(time_selector)

        # Gather the points
        x_min, x_max = axis["MIN"], axis["MAX"]
        z = np.asarray(list(gen))

        # Options are different for contourf and pcolormesh methods. If the number of cases to distinguish increases
        # procedurally build the call instead of further adding if clauses.

        if contour_plot:
            # Rasterizing in contourf is a bit tricky
            # Cf. http://stackoverflow.com/questions/33250005/size-of-matplotlib-contourf-image-files
            if log_map:
                # Mask manually to prevent a UserWarning
                if rasterized:
                    plot = ax.contourf(axis["LIST"], time_list, np.ma.masked_where(z <= 0, z),
                                       locator=ticker.LogLocator(), cmap=cmap, zorder=-9)
                    ax.set_rasterization_zorder(-1)
                else:
                    plot = ax.contourf(axis["LIST"], time_list, np.ma.masked_where(z <= 0, z),
                                       locator=ticker.LogLocator(), cmap=cmap)
            else:
                if rasterized:
                    plot = ax.contourf(axis["LIST"], time_list, z, cmap=cmap, zorder=-9)
                    ax.set_rasterization_zorder(-1)
                else:
                    plot = ax.contourf(axis["LIST"], time_list, z, cmap=cmap)
        else:
            # Although pcolormesh works with the rasterize option, the tricky method can also be used
            if log_map:
                # Mask manually to prevent a UserWarning
                masked_z = np.ma.masked_where(z <= 0, z)
                if z_min is None:
                    z_min = masked_z.min()
                if z_max is None:
                    z_max = masked_z.max()
                # TODO: Sometimes z_min is not an interesting value. Perhaps add a parameter to set it.
                if rasterized:
                    plot = ax.pcolormesh(axis["LIST"], time_list, masked_z, norm=LogNorm(vmin=z_min, vmax=z_max),
                                         cmap=cmap, zorder=-9)
                    ax.set_rasterization_zorder(-1)
                else:
                    plot = ax.pcolormesh(axis["LIST"], time_list, masked_z, norm=LogNorm(vmin=z_min, vmax=z_max),
                                         cmap=cmap)
            else:
                if rasterized:
                    plot = ax.pcolormesh(axis["LIST"], time_list, z, cmap=cmap, zorder=-9)
                    ax.set_rasterization_zorder(-1)
                else:
                    plot = ax.pcolormesh(axis["LIST"], time_list, z, cmap=cmap)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(time_list[0], time_list[-1])

        ax.set_title("%s (%s)" % (title_name, title_units))

        fig.colorbar(plot)

        if output_path:  # "" or None
            plt.savefig(output_path, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax


def get_diagnostic_list(run_dir="."):
    """
    Create a list with the diagnostic found in the given run directory.

    Args:
        run_dir (str): The run directory.

    Returns:
        :obj:`list` of :obj:`Diagnostic`: List of the diagnostic found.

    """
    diagnostic_list = []
    # This might be not compatible with Windows due to the meaning of \ in patterns, but who cares...
    ms_folder = os.path.join(run_dir, "MS", "")  # Notice the empty string forces the route ending in the separator...
    for root, dirs, files in os.walk(ms_folder):
        if not dirs and files:  # Terminal directory with files in it
            diagnostic_list.append(Diagnostic(root))
    return diagnostic_list
