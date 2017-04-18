# -*- coding: UTF-8 -*-
"""Run configuration files with OSIRIS."""
from __future__ import print_function


from os import path, remove, walk
from shutil import copyfile
import subprocess
from time import sleep
import sys
import re

from ..common import ensure_dir_exists, ensure_executable, ifd

# Path to osiris executables - guessed later in the code
osiris_1d = ""
"""Path to the osiris-1D.e file"""
osiris_2d = ""
"""Path to the osiris-2D.e file"""
osiris_3d = ""
"""Path to the osiris-3D.e file"""


def set_osiris_path(folder, warn=True):
    global osiris_1d, osiris_2d, osiris_3d
    if not path.isdir(folder):
        if warn:
            print("Warning: %s is not an existing folder." % folder, file=sys.stderr)
        return

    r = path.join(folder, "osiris-1D.e")
    if path.isfile(r):
        osiris_1d = r
    elif warn:
        print("Warning: osiris-1D not found in %s" % folder, file=sys.stderr)
    r = path.join(folder, "osiris-2D.e")
    if path.isfile(r):
        osiris_2d = r
    elif warn:
        print("Warning: osiris-2D not found in %s" % folder, file=sys.stderr)
    r = path.join(folder, "osiris-3D.e")
    if path.isfile(r):
        osiris_3d = r
    elif warn:
        print("Warning: osiris-3D not found in %s" % folder, file=sys.stderr)


def run_mono(config, run_dir, prefix=None, clean_dir=True, blocking=None):
    """
    Run a config file with Osiris.

    Args:
        config (`ConfigFile`): the instance describing the configuration file.
        run_dir (str): Folder where the run is carried.
        prefix (str): A prefix to run the command (e.g., "qsub", ...).
        clean_dir (bool): Whether to remove the files in the directory before execution.
        blocking: Whether to wait for the run to finish.

    Returns:
        (tuple): A tuple with:

            * (bool): Whether the started run found an error in the very first ms.
            * (`subprocess.Popen`): A Popen instance of the started run.

    """
    # TODO: Daemonization
    if clean_dir:
        for root, dirs, files in walk(run_dir):
            for f in files:
                remove(path.join(root, f))
    ensure_dir_exists(run_dir)
    config.write(path.join(run_dir, "os-stdin"))
    osiris_path = path.abspath(path.join(run_dir, "osiris"))
    osiris = ifd(config.get_d(), osiris_1d, osiris_2d, osiris_3d)
    copyfile(osiris, osiris_path)
    ensure_executable(osiris_path)

    if not prefix:  # None or ""
        prefix = ""
    elif prefix[-1] != " ":
        prefix += " "

    proc = subprocess.Popen(prefix + osiris_path + " > out.txt 2> err.txt", shell=True, cwd=path.abspath(run_dir))
    if blocking:
        proc.wait()
    else:  # Sleep a little to check for quickly appearing errors
        sleep(0.2)
    # Try to detect errors checking the output
    with open(path.join(run_dir, "out.txt"), "r") as f:
        text = f.read()
    # TODO: Optimize this search
    if "(*error*)" in text or re.search("Error reading .* parameters", text) or re.search("MPI_ABORT was invoked",
                                                                                          text):
        print(
            "Error detected while launching %s.\nCheck out.txt there for more information or re-run in console." % run_dir,
            file=sys.stderr)
        success = False
    else:
        success = True
    return success, proc


# Try to guess the OSIRIS location:
for t in [path.join(path.expanduser("~"), "osiris", "bin"),
          path.join("usr", "local", "osiris", "bin")]:
    set_osiris_path(t, warn=False)
    if osiris_1d and osiris_2d and osiris_3d:
        break

if not (osiris_1d and osiris_2d and osiris_3d):
    if not (osiris_1d or osiris_2d or osiris_3d):
        print("Warning: no OSIRIS executables were found.", file=sys.stderr)
    else:
        if not osiris_1d:
            print("Warning: osiris-1D.e not found.", file=sys.stderr)
        if not osiris_2d:
            print("Warning: osiris-2D.e not found.", file=sys.stderr)
        if not osiris_3d:
            print("Warning: osiris-3D.e not found.", file=sys.stderr)
    print("Use the function set_osiris_path or set the variables run.osiris_1d and so to allow the run module to work.",
          file=sys.stderr)
