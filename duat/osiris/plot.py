# -*- coding: UTF-8 -*-
from __future__ import print_function
from __future__ import division

from glob import glob
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from ..common import ensure_dir_exists, human_order_key


def _dim_hdf5(file):
    """
    get the dimensions of an opened hdf5 file.

    Args:
        file(`h5py.File`): The opened file object.

    Returns:
        (tuple): A tuple with

            *(int): The number of grid dimensions.
            *(int): The number of datasets (excluding axes definition).
    """
    keys = list(file.keys())
    if "AXIS" not in keys:
        raise ValueError("AXIS group not found.")
    return len(file["AXIS"]), len(keys) - 1


def dim_hdf5_dir(data_path):
    """
    get the dimensions of the files in a diagnostic directory.

    Args:
        data_path: The directory with the files.

    Returns:
        (tuple): A tuple with:

            *(int): The number of grid dimensions.
            *(int): The number of datasets (excluding axes definition).
            *(int): The number of snapshots in time.
    """
    file_list = glob(os.path.join(data_path, "*.h5"))
    if file_list:
        f = h5py.File(file_list[0], "r")
        d1, d2 = _dim_hdf5(f)
        return d1, d2, len(file_list)
    else:
        return 0, 0, 0


def time_1d(data_path, output_path=None, dataset=None, dpi=200, fps=1, scale_mode="expand", latex_label=True):
    """
    Generate a time-dependent 1d plot.

    Args:
        data_path(str): The folder containing the files with the slices in time.
        output_path (str): The place where the plot is saved. If "" or None, the plot is shown in matplotlib.
        dataset (str or int): The dataset used to plot. It can be a string with the name or a int with its position in human-order among the datasets.
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
    y_name = f[data_key].attrs["LONG_NAME"][0].decode('UTF-8')
    y_units = f[data_key].attrs["UNITS"][0].decode('UTF-8')
    if latex_label:
        x_units = "$" + x_units + "$"
        y_units = "$" + y_units + "$"
        # Names might be text or LaTeX. Try to guess
        if re.match("^.(_.)?$", x_name):  # If in the form x or x_i
            x_name = "$" + x_name + "$"
        if re.match("^.(_.)?$", y_name):
            y_name = "$" + y_name + "$"
    ax.set_xlabel("%s (%s)" % (x_name, x_units))
    ax.set_ylabel("%s (%s)" % (y_name, y_units))

    # Plot the points
    x_min, x_max = f["AXIS"]["AXIS1"][:]
    plot_data, = ax.plot(np.linspace(x_min, x_max, num=len(selected_dataset)), selected_dataset[:])
    ax.set_xlim(x_min, x_max)

    # Prepare a function for the updates
    def update(i):
        """Update the plot, returning the artists which must be redrawn"""
        f = h5py.File(file_list[i], "r")
        new_dataset = f[data_key]
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


# FIXME: HACK: unify with previous (?)
def time_1d_across(data_path, output_path=None, position=None, dpi=200, fps=1, scale_mode="expand", latex_label=True):
    """
    Generate a time-dependent 1d plot.

    Args:
        data_path(str): The folder containing the files with the slices in time.
        output_path (str): The place where the plot is saved. If "" or None, the plot is shown in matplotlib.
        dataset (str or int): The dataset used to plot. It can be a string with the name or a int with its position in human-order among the datasets.
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
        if re.match("^.(_.)?$", x_name):  # If in the form x or x_i
            x_name = "$" + x_name + "$"
        if re.match("^.(_.)?$", y_name):
            y_name = "$" + y_name + "$"
    ax.set_xlabel("%s" % x_name)
    ax.set_ylabel("%s (%s)" % (y_name, y_units))

    # Plot the points
    x = []
    y = []
    # TODO: If position is none
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


def auto_process(run_dir=".", file_format="mp4", output_dir=None, verbose=None, kwargs_1d=None):
    """
    Automatically process the files generated by a Osiris run.

    Args:
        run_dir (str): The run directory.
        file_format (str): The preferred format for the generated files (mp4, gif...).
        output_dir (str): Where to output the plots. Default: run_dir/plot.
        verbose (bool): Whether to print to stdout a message when a file is generated.
        kwargs_1d (dict): Aditional keywords arguments passed to `time_1d` and `time_1d_across`

    Returns:
        (int) Number of files generated.

    """
    # TODO: Pass kwargs
    if not output_dir:
        output_dir = os.path.join(run_dir, "plot")
    ensure_dir_exists(output_dir)

    if not kwargs_1d:
        kwargs_1d = {}

    generated = 0

    v_print = print if verbose else lambda *args, **kwargs: None

    # This might be not compatible with Windows due to the meaning of \ in patterns, but who cares...
    ms_folder = os.path.join(run_dir, "MS", "")  # Notice the empty string forces the route ending in the separator...
    for root, dirs, files in os.walk(ms_folder):
        if not dirs and files:  # Terminal directory with files in it
            route = re.match(ms_folder + "(.*)", root).group(1)  # ... so there is no initial separator here
            route = route.replace(os.sep, "_")
            filename_base = os.path.join(output_dir, route)
            d_spatial, d_datasets, d_time = dim_hdf5_dir(root)
            v_print("Generating file(s) for " + root + "\n- Dimensions: " + str((d_spatial, d_datasets, d_time)))
            if d_spatial == 1 and d_datasets == 1:
                v_print("- Generating: " + filename_base + "." + file_format)
                time_1d(root, filename_base + "." + file_format, **kwargs_1d)
                generated += 1
            elif d_spatial == 1:
                chosen_datasets = [0, 1] if d_datasets == 2 else [0, d_datasets // 2, d_datasets - 1]
                for c in chosen_datasets:
                    v_print("- Generating: " + filename_base + "_" + str(c) + "." + file_format)
                    time_1d(root, filename_base + "_" + str(c) + "." + file_format, dataset=c, **kwargs_1d)
                    generated += 1

    return generated
