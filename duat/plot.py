# -*- coding: UTF-8 -*-
"""Plot OSIRIS-generated data."""

from __future__ import print_function
from __future__ import division

from glob import glob
import os
import re
from math import floor

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as colors
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


def _create_label(name, units, latex_label=False):
    """
    Prepare a label for a plot
    
    Args:
        name (str): Text describing the magnitude 
        units (str): Text describing the units 
        latex_label (bool): Whether to use LaTeX 

    Returns:
        str: The desired label

    """
    if latex_label:
        if units:
            units = "$" + _improve_latex(units) + "$"
        # Names might be text or LaTeX. Try to guess
        if _is_latex(name):
            name = "$" + _improve_latex(name) + "$"

    if units:
        return "%s (%s)" % (name, units)
    else:
        return name


def _autonorm(norm, z):
    """
    Automatic options for color plot normalization.

    Args:
        norm (str or other): Description of the norm.
        z (matrix of numbers): Data.

    Returns:
        (`matplotlib.colors.Normalize`): A suitable normalize option (or None)

    """
    if isinstance(norm, str):
        norm = norm.lower()
    if norm == "lin":
        norm = None
    if norm == "log":
        masked_z = np.ma.masked_where(z <= 0, z)
        z_max = masked_z.max()
        z_min = masked_z.min()
        z_min = max(z_min, z_max / 1E9)
        norm = colors.LogNorm(vmin=z_min, vmax=z_max)
    return norm


def _fix_colorbar(cbar):
    # Manual fix for matplotlib's 8358 issue
    if isinstance(cbar.norm, colors.LogNorm) or isinstance(cbar.norm, colors.SymLogNorm):
        cbar.ax.minorticks_off()


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

    def _scaled_slice_to_slice(self, scaled_slice):
        return scaled_slice._get_slice(self.t_0, self.dt * len(self.snapshot_list), len(self.snapshot_list))

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


            time_selector (slice or ScaledSlice): A slice or ScaledSlice instance selecting the points in time to take.
                A slice selects times from the list returned by :func:`~duat.osiris.plot.Diagnostic.get_time_list`.
                A ScaledSlice chooses a slice that best represents a choice in terms of time units.

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
            if len(axes_selector) != len(self.axes):
                raise ValueError(
                    "Invalid axes_selector parameter. Length must be %d. Check the axes of the Diagnostic instance." % len(
                        self.axes))

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

        if time_selector is not None:
            if isinstance(time_selector, ScaledSlice):
                time_selector = self._scaled_slice_to_slice(time_selector)
            elif not isinstance(time_selector, slice):
                logger.warning("Invalid time_selector parameter ignored. Use a slice or a ScaledSlice instead.")
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
            if len(axes_selector) != len(self.axes):
                raise ValueError(
                    "Invalid axes_selector parameter. Length must be %d. Check the axes of the Diagnostic instance." % len(
                        self.axes))
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
            if isinstance(time_selector, slice):
                # This could be improved to avoid generating unneeded values
                return [self.t_0 + self.dt * i for i in self.snapshot_list][time_selector]
            elif isinstance(time_selector, ScaledSlice):
                return self.get_time_list(self._scaled_slice_to_slice(time_selector))
            else:
                raise TypeError("time_selector must be a slice or a ScaledSlice")
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
                * "max": Use the maximum range from the beginning.

            latex_label (bool): Whether for use LaTeX code for the plot.
            
        Returns:
            (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`, `matplotlib.animation.FuncAnimation`):
            Objects representing the generated plot and its animation.
            
        Raises:
            FileNotFoundError: If tried to export to mp4 but ffmpeg is not found in the system.

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
        y_name = self.data_name
        y_units = self.units

        ax.set_xlabel(_create_label(x_name, x_units, latex_label))
        ax.set_ylabel(_create_label(y_name, y_units, latex_label))

        # Plot the points
        x_min, x_max = axis["MIN"], axis["MAX"]
        plot_data, = ax.plot(axis["LIST"], next(gen))
        ax.set_xlim(x_min, x_max)

        if scale_mode == "max":
            # Get a list (generator) with the mins and maxs in each time step
            min_max_list = map(lambda l: [min(l), max(l)],
                               self.get_generator(dataset_selector=dataset_selector, axes_selector=axes_selector,
                                                  time_selector=time_selector))
            f = lambda mins, maxs: (min(mins), max(maxs))
            y_min, y_max = f(*zip(*min_max_list))
            ax.set_ylim(y_min, y_max)

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
            if not scale_mode or scale_mode == "max":
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
                         dpi=200, latex_label=True, cmap=None, norm=None, show=True, rasterized=True,
                         contour_plot=False):
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
            norm (str or `matplotlib.colors.Normalize`): How to scale the colormap. For advanced manipulation, use some
                           Normalize subclass, e.g., colors.SymLogNorm(0.01). Automatic scales can be selected with
                           the following strings:

                           * "lin": Linear scale from minimum to maximum.
                           * "log": Logarithmic scale from minimum to maximum up to vmax/vmin>1E9, otherwise increasing vmin.


            show (bool): Whether to show the plot. This is blocking if matplotlib is in non-interactive mode.
            rasterized (bool): Whether the map is rasterized. This does not apply to axes, title... Note non-rasterized
                               images with large amount of data exported to PDF might challenging to handle.
            contour_plot (bool): Whether contour lines are plot instead of the density map.
                           
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

        ax.set_xlabel(_create_label(x_name, x_units, latex_label))
        ax.set_ylabel(_create_label(y_name, y_units, latex_label))

        time_list = self.get_time_list(time_selector)

        # Gather the points
        x_min, x_max = axis["MIN"], axis["MAX"]
        z = np.asarray(list(gen))

        norm = _autonorm(norm, z)

        plot_function = ax.contourf if contour_plot else ax.pcolormesh
        if rasterized:
            # Rasterizing in contourf is a bit tricky
            # Cf. http://stackoverflow.com/questions/33250005/size-of-matplotlib-contourf-image-files
            plot = plot_function(axis["LIST"], time_list, z, norm=norm, cmap=cmap, zorder=-9)
            ax.set_rasterization_zorder(-1)
        else:
            plot = plot_function(axis["LIST"], time_list, z, norm=norm, cmap=cmap)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(time_list[0], time_list[-1])

        ax.set_title(_create_label(title_name, title_units, latex_label))

        _fix_colorbar(fig.colorbar(plot))

        if output_path:  # "" or None
            plt.savefig(output_path, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    def axes_2d_colormap(self, output_path=None, dataset_selector=None, axes_selector=None, time_selector=None,
                         dpi=200, latex_label=True, cmap=None, norm=None, show=True, rasterized=True,
                         contour_plot=False):
        """
        Generate a colormap in two axes.
        
        A single time snapshot must be selected with the time_selector parameter. For an animated version in time see 
        the :func:`~duat.osiris.plot.Diagnostic.time_2d_animation` method.


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
            norm (str or `matplotlib.colors.Normalize`): How to scale the colormap. For advanced manipulation, use some
                           Normalize subclass, e.g., colors.SymLogNorm(0.01). Automatic scales can be selected with
                           the following strings:

                           * "lin": Linear scale from minimum to maximum.
                           * "log": Logarithmic scale from minimum to maximum up to vmax/vmin>1E9, otherwise increasing vmin.


            show (bool): Whether to show the plot. This is blocking if matplotlib is in non-interactive mode.
            rasterized (bool): Whether the map is rasterized. This does not apply to axes, title... Note non-rasterized
                               images with large amount of data exported to PDF might challenging to handle.
            contour_plot (bool): Whether contour lines are plot instead of the density map.

        Returns:
            (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`): Objects representing the generated plot.

        """
        if output_path:
            ensure_dir_exists(os.path.dirname(output_path))
        axes = self.get_axes(dataset_selector=dataset_selector, axes_selector=axes_selector)
        if len(axes) != 2:
            raise ValueError("Expected 2 axes plot, but %d were provided" % len(axes))

        gen = self.get_generator(dataset_selector=dataset_selector, axes_selector=axes_selector,
                                 time_selector=time_selector)

        # Set plot labels
        fig, ax = plt.subplots()

        x_name = axes[0]["LONG_NAME"]
        x_units = axes[0]["UNITS"]
        y_name = axes[1]["LONG_NAME"]
        y_units = axes[1]["UNITS"]
        title_name = self.data_name
        title_units = self.units

        ax.set_xlabel(_create_label(x_name, x_units, latex_label))
        ax.set_ylabel(_create_label(y_name, y_units, latex_label))

        time_list = self.get_time_list(time_selector)

        if len(time_list) == 0:
            raise ValueError("No time snapshot selected")
        if len(time_list) != 1:
            raise ValueError("A single time snapshot must be selected for this plot")

        # Gather the points
        x_min, x_max = axes[0]["MIN"], axes[0]["MAX"]
        y_min, y_max = axes[1]["MIN"], axes[1]["MAX"]
        z = np.transpose(np.asarray(list(gen)[0]))

        norm = _autonorm(norm, z)

        plot_function = ax.contourf if contour_plot else ax.pcolormesh
        if rasterized:
            # Rasterizing in contourf is a bit tricky
            # Cf. http://stackoverflow.com/questions/33250005/size-of-matplotlib-contourf-image-files
            plot = plot_function(axes[0]["LIST"], axes[1]["LIST"], z, norm=norm, cmap=cmap, zorder=-9)
            ax.set_rasterization_zorder(-1)
        else:
            plot = plot_function(axes[0]["LIST"], axes[1]["LIST"], z, norm=norm, cmap=cmap)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_title(_create_label(title_name, title_units, latex_label))

        _fix_colorbar(fig.colorbar(plot))

        if output_path:  # "" or None
            plt.savefig(output_path, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close()

        return fig, ax

    def time_2d_animation(self, output_path=None, dataset_selector=None, axes_selector=None, time_selector=None,
                          dpi=200, fps=1, cmap=None, norm=None, rasterized=True, z_min=None,
                          z_max=None, latex_label=True, interval=200):
        """
        Generate a plot of 2d data as a color map which animated in time.

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
            latex_label (bool): Whether for use LaTeX code for the plot.
            cmap (str or `matplotlib.colors.Colormap`): The Colormap to use in the plot.
            norm (str or `matplotlib.colors.Normalize`): How to scale the colormap. For advanced manipulation, use some
                           Normalize subclass, e.g., colors.SymLogNorm(0.01). Automatic scales can be selected with
                           the following strings:

                           * "lin": Linear scale from minimum to maximum.
                           * "log": Logarithmic scale from minimum to maximum up to vmax/vmin>1E9, otherwise increasing vmin.


            rasterized (bool): Whether the map is rasterized. This does not apply to axes, title... Note non-rasterized
                               images with large amount of data exported to PDF might challenging to handle.
        Returns:
            (`matplotlib.figure.Figure`, `matplotlib.axes.Axes`, `matplotlib.animation.FuncAnimation`):
            Objects representing the generated plot and its animation.
            
        Raises:
            FileNotFoundError: If tried to export to mp4 but ffmpeg is not found in the system.

        """
        if output_path:
            ensure_dir_exists(os.path.dirname(output_path))
        axes = self.get_axes(dataset_selector=dataset_selector, axes_selector=axes_selector)
        if len(axes) != 2:
            raise ValueError("Expected 2 axes plot, but %d were provided" % len(axes))

        gen = self.get_generator(dataset_selector=dataset_selector, axes_selector=axes_selector,
                                 time_selector=time_selector)

        # Set plot labels
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)

        x_name = axes[0]["LONG_NAME"]
        x_units = axes[0]["UNITS"]
        y_name = axes[1]["LONG_NAME"]
        y_units = axes[1]["UNITS"]
        title_name = self.data_name
        title_units = self.units

        ax.set_xlabel(_create_label(x_name, x_units, latex_label))
        ax.set_ylabel(_create_label(y_name, y_units, latex_label))

        # Gather the points
        x_min, x_max = axes[0]["MIN"], axes[0]["MAX"]
        y_min, y_max = axes[1]["MIN"], axes[1]["MAX"]
        z = np.transpose(np.asarray(next(gen)))

        time_list = self.get_time_list(time_selector)
        if len(time_list) < 2:
            raise ValueError("At least two time snapshots are needed to make an animation")

        norm = _autonorm(norm, z)

        plot_function = ax.pcolormesh
        if rasterized:
            # Rasterizing in contourf is a bit tricky
            # Cf. http://stackoverflow.com/questions/33250005/size-of-matplotlib-contourf-image-files
            plot = plot_function(axes[0]["LIST"], axes[1]["LIST"], z, norm=norm, cmap=cmap, zorder=-9)
            ax.set_rasterization_zorder(-1)
        else:
            plot = plot_function(axes[0]["LIST"], axes[1]["LIST"], z, norm=norm, cmap=cmap)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_title(_create_label(title_name, title_units, latex_label))

        _fix_colorbar(fig.colorbar(plot))

        # Prepare a function for the updates
        def update(i):
            """Update the plot, returning the artists which must be redrawn."""
            try:
                new_dataset = np.transpose(np.asarray(next(gen)))
            except StopIteration:
                logger.warning("Tried to add a frame to the animation, but all data was used.")
                return
            label = 't = {0}'.format(time_list[i])
            # BEWARE: The set_array syntax is rather problematic. Depending on the shading used in pcolormesh, the
            #         following might not work.
            plot.set_array(new_dataset[:-1, :-1].ravel())
            # For more details, check lumbric's answer to
            # https://stackoverflow.com/questions/18797175/animation-with-pcolormesh-routine-in-matplotlib-how-do-i-initialize-the-data
            ax.set_title(label)
            return plot, ax

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


def _pos_in_bin(x, a, b, n):
    """Find the bin where x is found in a mesh from a to b with n points (including both).
    The index starts at 0 and if x is in the mesh the interval to the right will be returned.
    The returned value can be outside [0, n-1] if x is not in [a, b).
    Note that for x=b, the returned value is n"""
    return floor((x - a) / (b - a) * n)


class ScaledSlice:
    """
    A slice described in simulation units (instead of list position).
    
    This object can be used to describe a time_selector parameter.
        
    """

    def __init__(self, start, stop, step=None):
        """
        Create a ScaledSlice instance.
        
        Args:
            start(float): Where the slice should start. Actual start will be before if needed. 
            stop(float): Where the slice should stop. The point is in general excluded, as usual in Python.
            step (float): The desired step of the slice. Actual step will be the biggest multiple of the mesh step which
                          is less than this one.
            
        """
        self.start = start
        self.stop = stop
        self.step = step

    def __repr__(self):
        if self.step:
            return "ScaledSlice<(%g, %g, %g)>" % (self.start, self.stop, self.step)
        else:
            return "ScaledSlice<(%g, %g)>" % (self.start, self.stop)

    def _get_slice(self, mesh_min, mesh_max, n_points):
        """Return a slice best approximating the instance in the given partition"""
        a = _pos_in_bin(self.start, mesh_min, mesh_max, n_points)
        if a < 0:
            a = 0
        if a >= n_points:
            a = n_points - 1
        b = _pos_in_bin(self.stop, mesh_min, mesh_max, n_points)
        if b < 0:
            b = 0
        if b >= n_points:
            b = n_points - 1
        if self.step:
            c = floor(self.step / ((mesh_max - mesh_min) / n_points))
            if c < 1:
                c = 1
            return slice(a, b, c)
        else:
            return slice(a, b)
