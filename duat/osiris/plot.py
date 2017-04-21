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
from matplotlib.animation import FuncAnimation, FFMpegWriter

from ..common import ensure_dir_exists, human_order_key, MPCaller, Call


def _is_latex(s):
    """

    Decide if a string is a LaTeX expression.

    Args:
        s (str): The string to test.

    Returns: (bool) True if it seems a LaTeX expression, False otherwise

    """
    if not s:  # Empty string is not LaTeX
        return False
    return re.match("^.(_.)?$", s) or "\\" in s  # If in the form x, x_i, or if there is a backlash (command)


class Diagnostic:
    """
    A OSIRIS diagnostic.

    Attributes:
        data_path (str): Path to the directory were the data is stored.
        file_list (`list` of `str`): List of h5 files, one per time snapshot.
        time_list (`list` of `str`): List of times in each snapshot.
        keys (`list` of `str`): Names of the datasets in the Diagnostic, given in human order.
        axes (`list` of `dict`): Info of each axis in the Diagnostic.
        dim (`tuple` of `int`): A tuple with:

            * The number of grid dimensions.
            * The number of datasets (excluding axes definition).
            * The number of snapshots in time.
    """

    def __init__(self, data_path):
        """
        Create a Diagnostic instance.

        Args:
            data_path: Path of the directory containing the diagnostic data
        """
        self.data_path = data_path
        self.file_list = glob(os.path.join(data_path, "*.h5"))
        self.file_list.sort(key=human_order_key)
        self.time_list = list(
            map(lambda x: float((os.path.split(x)[1]).split(".h5")[0].split("-")[-1]), self.file_list))

        # Get info from first time snapshot
        with h5py.File(self.file_list[0], "r") as f:
            self.keys = self._get_keys(f)
            self.axes = self._get_axes(f, self.keys[0])

        self.dim = (len(self.axes), len(self.keys), len(self.time_list))

    def _clean_dataset_key(self, dataset_key):
        """Return the given key as str, using human order if int. Might rise error or warning"""
        if isinstance(dataset_key, int):
            dataset_key = self.keys[dataset_key]
        elif isinstance(dataset_key, str):
            if dataset_key not in self.keys:
                raise ValueError("Dataset %s does not exist in the file." % dataset_key)
        elif dataset_key is None:
            if len(self.keys) != 1:  # Warn if implicitly selecting one among others.
                print("No dataset selected when multiple are available. Plotting the first one.")
            dataset_key = self.keys[0]
        else:
            raise TypeError("Unknown dataset type: %s", type(dataset_key))
        return dataset_key

    def get_generator(self, dataset_selector=None, axes_selector=None, time_selector=None):
        """
        Get a generator providing data from the file.

        Calling this method returns a generator which, when called, will provide data for increasing times (unless
        modified by time_selector parameter). The data might be reduced either by selecting a position in an axis (or
        a dataset) or by using a function along some axis (or datasets), e.g. a sum.

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

        Returns: (generator): A generator which provides the data.

        """
        multiple_datasets = False  # If a dataset list is going to be returned
        if dataset_selector:
            if self.dim[1] == 1:
                print("Single dataset found. Ignoring the provided dataset_selector.")

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
            if self.dim[1] > 1:
                multiple_datasets = True

                def f_dataset_selector(f):
                    return np.array([f[key][:] for key in self.keys])
            else:
                def f_dataset_selector(f):
                    return f[self.keys[0]][:]

        if axes_selector:
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
            print("Invalid time_selector parameter ignored. Use a slice instead.")
            time_selector = None

        def gen():
            for file_name in (self.file_list[time_selector] if time_selector else self.file_list):
                with h5py.File(file_name, "r") as f:
                    data = f_dataset_selector(f)
                # Make sure to exit the context manager before yielding
                # h5py might accuse you of murdering identifiers if you don't!
                yield f_axes_selector(data)

        return gen()

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
        for i, axis in enumerate(file["AXIS"]):
            ax = file["AXIS"][axis]
            data = {}
            for d in ["LONG_NAME", "UNITS", "NAME", "TYPE"]:
                data[d] = ax.attrs[d][0].decode('UTF-8')
            data["MIN"], data["MAX"] = ax[:]
            # TODO: Non linear axis
            data["LIST"] = np.linspace(data["MIN"], data["MAX"], num=file[dataset_key].shape[i])
            axes.append(data)

        return axes

    def time_1d_animation(self, output_path=None, dataset_selector=None, axes_selector=None, time_selector=None, dpi=200, fps=1, scale_mode="expand",
                          latex_label=True):
        """
        Generate a plot in an axis animated in time.

        Args:
            output_path (str): The place where the plot is saved. If "" or None, the plot is shown in matplotlib.
            dataset_selector: See `get_generator` method.
            axes_selector: See `get_generator` method.
            time_selector: See `get_generator` method.
            dpi (int): The resolution of the file in dots per inch.
            fps (float): The frames per seconds.
            scale_mode (str): How the scale is changed thorough time. Available methods are:

                * "expand": The y limits increase when needed, but they don't decrease.
                * "adjust_always": Always change the y limits to those of the data.

            latex_label (bool): Whether for use LaTeX code for the plot.

        Returns:

        """
        gen = self.get_generator(dataset_selector=dataset_selector, axes_selector=axes_selector,
                                 time_selector=time_selector)

        # Set plot labels
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        x_name = self.axes[0]["LONG_NAME"]
        x_units = self.axes[0]["UNITS"]
        with h5py.File(self.file_list[0], "r") as f:
            y_name = f[self.keys[0]].attrs["LONG_NAME"][0].decode('UTF-8')
            y_units = f[self.keys[0]].attrs["UNITS"][0].decode('UTF-8')
        if latex_label:
            if x_units:
                x_units = "$" + x_units + "$"
            if y_units:
                y_units = "$" + y_units + "$"
            # Names might be text or LaTeX. Try to guess
            if _is_latex(x_name):
                x_name = "$" + x_name + "$"
            if _is_latex(y_name):
                y_name = "$" + y_name + "$"
        ax.set_xlabel("%s (%s)" % (x_name, x_units))
        ax.set_ylabel("%s (%s)" % (y_name, y_units))

        # Plot the points
        x_min, x_max = self.axes[0]["MIN"], self.axes[0]["MAX"]
        plot_data, = ax.plot(self.axes[0]["LIST"], next(gen))
        ax.set_xlim(x_min, x_max)

        # Prepare a function for the updates
        def update(i):
            """Update the plot, returning the artists which must be redrawn"""
            new_dataset = next(gen)
            label = 't = {0}'.format(self.time_list[i])
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

        if not output_path:  # "" or None
            # FIXME: Plot in matplotlib window seems not to be working now.
            FuncAnimation(fig, update, frames=np.arange(0, len(self.time_list)), interval=1)
            plt.show()
        elif output_path.split(".")[-1].lower() == "gif":
            anim = FuncAnimation(fig, update, frames=np.arange(0, len(self.time_list)), interval=200)
            anim.save(output_path, dpi=dpi, writer='imagemagick')
        else:
            metadata = dict(title=os.path.split(self.data_path)[-1], artist='duat', comment=self.data_path)
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            with writer.saving(fig, output_path, dpi):
                # Iterate over frames
                for i in np.arange(1, len(self.time_list)):
                    update(i)
                    writer.grab_frame()
                # Keep showing the last frame for the fixed time
                writer.grab_frame()
        plt.close()


# TODO: Recode. DEPRECATED
def time_dataset_animation(data_path, output_path=None, position=None, dpi=200, fps=1, scale_mode="expand",
                           latex_label=True):
    """
    Generate a dataset-distributed magnitude animated in time.

    Args:
        data_path(str): The folder containing the files with the slices in time.
        output_path (str): The place where the plot is saved. If "" or None, the plot is shown in matplotlib.
        position (int): Number of the position in the axis to plot.
        dpi (int): The resolution of the file in dots per inch.
        fps (float): The frames per seconds.
        scale_mode (str): How the scale is changed thorough time. Available methods are:

            * "expand": The y limits increase when needed, but they don't decrease.
            * "adjust_always": Always change the y limits to those of the data.

        latex_label (bool): Whether for use LaTeX code for the plot.

    Returns:

    """
    file_list = glob(os.path.join(data_path, "*.h5"))
    file_list.sort(key=human_order_key)
    time_list = list(map(lambda x: float((os.path.split(x)[1]).split(".h5")[0].split("-")[-1]), file_list))

    # Plot the first frame
    f = h5py.File(file_list[0], "r")
    keys = list(f.keys())

    # Choose the dataset
    if "AXIS" not in keys:
        raise ValueError("AXIS group not found in file %s." % file_list[0])
    keys.remove("AXIS")
    keys.sort(key=human_order_key)

    # Set plot labels
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    x_name = f[keys[0]].attrs["TAG"][0].decode('UTF-8')  # "Energy <=    0.00000000"
    x_name = re.match("(.*?) <", x_name).group(1)  # TODO: Catch errors, check syntax
    y_name = f[keys[0]].attrs["LONG_NAME"][0].decode('UTF-8')
    y_units = f[keys[0]].attrs["UNITS"][0].decode('UTF-8')
    if latex_label:
        y_units = "$" + y_units + "$"
        # Names might be text or LaTeX. Try to guess
        if _is_latex(x_name):
            x_name = "$" + x_name + "$"
        if _is_latex(y_name):
            y_name = "$" + y_name + "$"
    ax.set_xlabel("%s" % x_name)
    ax.set_ylabel("%s (%s)" % (y_name, y_units))

    # Plot the points
    x = []
    y = []
    if position is None:
        position = 0
        # Warn if implicitly selecting one among others. (Not sure if this scoring even exists, but here it goes)
        if len(f[keys[0]]) > 1:
            print("No position selected when multiple are available. Plotting the first one.")
    # b'   0.00000000      < Energy <=    5.00000007E-02'
    number_pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'  # A pattern matching floats
    for d in keys:
        y.append(f[d][position])
        tag = f[d].attrs["TAG"][0].decode('UTF-8')  # E.g., '   0.00000000      < Energy <=    5.00000007E-02'
        values = list(map(float, re.findall(number_pattern, tag)))
        x.append(sum(values) / len(values))
    # TODO: First and last datasets are not bounded intervals. Their only finite point is used as the x coordinate, which must be understood with caution

    plot_data, = ax.plot(x, y)
    ax.set_xlim(min(x), max(x))

    # Prepare a function for the updates
    def update(i):
        """Update the plot, returning the artists which must be redrawn"""
        f = h5py.File(file_list[i], "r")
        y = []
        for d in keys:
            y.append(f[d][position])
        label = 't = {0}'.format(time_list[i])
        plot_data.set_ydata(y)
        ax.set_title(label)
        if not scale_mode:
            pass
        elif scale_mode == "expand":
            prev = ax.get_ylim()
            data_limit = [min(y), max(y)]
            ax.set_ylim(min(prev[0], data_limit[0]), max(prev[1], data_limit[1]))
        elif scale_mode == "adjust_always":
            ax.set_ylim(min(y), max(y))
        return plot_data, ax

    if not output_path:  # "" or None
        # TODO: Plot in matplotlib window seems not to be working now. Perhaps jupyter-related.
        FuncAnimation(fig, update, frames=np.arange(0, len(file_list)), interval=1)
        plt.show()
    elif output_path.split(".")[-1].lower() == "gif":
        anim = FuncAnimation(fig, update, frames=np.arange(0, len(file_list)), interval=200)
        anim.save(output_path, dpi=dpi, writer='imagemagick')
    else:
        metadata = dict(title=os.path.split(data_path)[-1], artist='duat', comment=data_path)
        writer = FFMpegWriter(fps=fps, metadata=metadata)
        with writer.saving(fig, output_path, dpi):
            # Iterate over frames
            for i in np.arange(0, len(file_list)):
                update(i)
                writer.grab_frame()
            # Keep showing the last frame for the fixed time
            writer.grab_frame()
    plt.close()


# TODO: Recode. DEPRECATED
def time_1d_colormap(data_path, output_path=None, dataset=None, dpi=200, latex_label=True, cmap=None):
    """
    Generate a colormap in an axis and the time.

    This function plots a magnitude depending on ONE spatial coordinate (hence the name) and on time as a colormap in
    the cartesian product of such a magnitude and the time.

    Args:
        data_path(str): The folder containing the files with the slices in time.
        output_path (str): The place where the plot is saved. If "" or None, the plot is shown in matplotlib.
        dataset (int or str): The dataset to use if multiple are available. Either an int with its position in human
                              order or a string with its name.
        dpi (int): The resolution of the file in dots per inch.
        latex_label (bool): Whether for use LaTeX code for the plot.
        cmap (str or `matplotlib.colors.Colormap`): The Colormap to use in the plot.

    """
    file_list = glob(os.path.join(data_path, "*.h5"))
    file_list.sort(key=human_order_key)
    time_list = list(map(lambda x: float((os.path.split(x)[1]).split(".h5")[0].split("-")[-1]), file_list))

    f = h5py.File(file_list[0], "r")
    keys = list(f.keys())

    # Choose the dataset
    if "AXIS" not in keys:
        raise ValueError("AXIS group not found in file %s." % file_list[0])
    keys.remove("AXIS")
    if isinstance(dataset, int):
        keys.sort(key=human_order_key)
        data_key = keys[dataset]
    elif isinstance(dataset, str):
        data_key = dataset
        if data_key not in dataset:
            raise ValueError("Dataset %s does not exist in the file." % dataset)
    elif dataset is None:
        if len(keys) != 1:  # Warn if implicitly selecting one among others.
            print("No dataset selected when multiple are available. Plotting the first one.")
            keys.sort(key=human_order_key)
            data_key = keys[0]
        else:
            data_key = keys[0]
    else:
        raise TypeError("Unknown dataset type: %s", type(dataset))
    selected_dataset = f[data_key]

    # Set plot labels
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    x_name = f["AXIS"]["AXIS1"].attrs["LONG_NAME"][0].decode('UTF-8')
    x_units = f["AXIS"]["AXIS1"].attrs["UNITS"][0].decode('UTF-8')
    y_name = "t"
    y_units = r"1 / \omega_p"  # Consistent with outputs, do not change
    if latex_label:
        x_units = "$" + x_units + "$"
        y_units = "$" + y_units + "$"
        # Names might be text or LaTeX. Try to guess
        if _is_latex(x_name):
            x_name = "$" + x_name + "$"
        y_name = "$" + y_name + "$"  # Y is the time, which is LaTeX here
    ax.set_xlabel("%s (%s)" % (x_name, x_units))
    ax.set_ylabel("%s (%s)" % (y_name, y_units))

    # Gather the points
    x_min, x_max = f["AXIS"]["AXIS1"][:]
    z = [selected_dataset[:]]
    for file in file_list[1:]:
        f = h5py.File(file, "r")
        new_dataset = f[data_key]
        z.append(new_dataset[:])
    z = np.asarray(z)
    contour_plot = ax.contourf(np.linspace(x_min, x_max, num=len(selected_dataset)), time_list, z, cmap=cmap)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(min(time_list), max(time_list))
    fig.colorbar(contour_plot)

    if not output_path:  # "" or None
        plt.show()
    else:
        plt.savefig(output_path, dpi=dpi)

    plt.close()


def auto_process(run_dir=".", file_format="mp4", output_dir=None, verbose=None, kwargs_1d=None, num_threads=None):
    """
    Automatically process the files generated by a Osiris run.

    Args:
        run_dir (str): The run directory.
        file_format (str): The preferred format for the generated files (mp4, gif...).
        output_dir (str): Where to output the plots. Default: run_dir/plot.
        verbose (bool): Whether to print to stdout a message when a file is generated.
        kwargs_1d (dict): Additional keywords arguments passed to `time_1d_animation` and `time_dataset_animation`.
        num_threads (int): Number of threads to use to process the files. If None or less than 2, no multiprocessing is
                           used.

    Returns:
        (int) Number of files generated.

    """
    # TODO: Pass kwargs
    # TODO: Choose if prefer video or colormap
    if not output_dir:
        output_dir = os.path.join(run_dir, "plot")
    ensure_dir_exists(output_dir)

    if not kwargs_1d:
        kwargs_1d = {}

    generated = 0

    v_print = print if verbose else lambda *args, **kwargs: None

    if isinstance(num_threads, int) and num_threads < 2:
        num_threads = 0
    if num_threads:
        t = MPCaller(num_threads)

    # This might be not compatible with Windows due to the meaning of \ in patterns, but who cares...
    ms_folder = os.path.join(run_dir, "MS", "")  # Notice the empty string forces the route ending in the separator...
    for root, dirs, files in os.walk(ms_folder):
        if not dirs and files:  # Terminal directory with files in it
            route = re.match(ms_folder + "(.*)", root).group(1)  # ... so there is no initial separator here
            route = route.replace(os.sep, "_")
            filename_base = os.path.join(output_dir, route)
            d = Diagnostic(root)
            d_spatial, d_datasets, d_time = d.dim
            v_print("Generating file(s) for " + root + "\n- Dimensions: " + str((d_spatial, d_datasets, d_time)))
            if d_spatial == 1 and d_datasets == 1:
                v_print("- Generating: " + filename_base + "." + file_format)
                if num_threads:
                    t.add_call(Call(d.time_1d_animation, filename_base + "." + file_format, **kwargs_1d))
                else:
                    d.time_1d_animation(filename_base + "." + file_format, **kwargs_1d)
                generated += 1
            elif d_spatial == 1:
                chosen_datasets = [0, 1] if d_datasets == 2 else [0, d_datasets // 2, d_datasets - 1]
                for c in chosen_datasets:
                    v_print("- Generating: " + filename_base + "_" + str(c) + "." + file_format)
                    if num_threads:
                        t.add_call(
                            Call(d.time_1d_animation, filename_base + "_" + str(c) + "." + file_format, dataset=c,
                                 **kwargs_1d))
                    else:
                        d.time_1d_animation(filename_base + "_" + str(c) + "." + file_format, dataset=c, **kwargs_1d)
                    generated += 1
    if num_threads:
        t.wait_calls()
    return generated
