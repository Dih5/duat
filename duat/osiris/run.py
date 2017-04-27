# -*- coding: UTF-8 -*-
"""Run configuration files with OSIRIS."""


from os import path, remove, walk
from shutil import copyfile
import subprocess
from time import sleep, time
import re

from ..common import ensure_dir_exists, ensure_executable, ifd, tail, logger, get_dir_size

import psutil

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
            logger.warning("%s is not an existing folder." % folder)
        return

    r = path.join(folder, "osiris-1D.e")
    if path.isfile(r):
        osiris_1d = r
    elif warn:
        logger.warning("osiris-1D not found in %s" % folder)
    r = path.join(folder, "osiris-2D.e")
    if path.isfile(r):
        osiris_2d = r
    elif warn:
        logger.warning("osiris-2D not found in %s" % folder)
    r = path.join(folder, "osiris-3D.e")
    if path.isfile(r):
        osiris_3d = r
    elif warn:
        logger.warning("osiris-3D not found in %s" % folder)


def _find_running_exe(exe):
    """Return the list of the pid of the processes of the argument executable (absolute path)"""
    candidates = []
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'exe'])
        except psutil.NoSuchProcess:
            pass
        else:
            if pinfo["exe"] and pinfo['exe'] == exe:
                candidates.append(pinfo['pid'])
    return candidates


class Run:
    """
    An osiris run.
    
    Attributes:
        run_dir (str): Directory where the run takes place.
        total_steps (int): Amount of time steps in the simulation.
        process (psutil.Process): Representation of the process running the simulation.
        
    Notes:
        Only single-process runs are supported at the moment.
        
    """

    def __init__(self, run_dir):
        self.run_dir = run_dir
        # TODO: Handle exceptions
        with open(path.join(run_dir, "os-stdin"), "r") as f:
            text = f.read()
        dt = float(re.match(r".*time_step(.*?){(.*?)dt(.*?)=(.*?),(.*?)}", text, re.DOTALL + re.MULTILINE).group(4))
        tmin = float(re.match(r".*time(.*?){(.*?)tmin(.*?)=(.*?),(.*?)}", text, re.DOTALL + re.MULTILINE).group(4))
        tmax = float(re.match(r".*time(.*?){(.*?)tmax(.*?)=(.*?),(.*?)}", text, re.DOTALL + re.MULTILINE).group(4))

        self.total_steps = int((tmax - tmin) // dt) + 1

        candidates = _find_running_exe(path.join(self.run_dir, "osiris"))

        try:
            if not candidates:  # No process running found
                self.process = None
            elif len(candidates) > 1:
                logger.warning("More than one pid was found for the run.\n"
                              "Multiple processes are not really handled by duat yet, do not trust what you see.")
                self.process = psutil.Process(candidates[0])
            else:
                self.process = psutil.Process(candidates[0])
        except psutil.NoSuchProcess:
            # If the process have died before processing was completed.
            self.process = None

    def __repr__(self):
        if self.is_running():  # Process has not finished yet
            return "Run<%s (%s/%d)>" % (self.run_dir, self.current_step(), self.total_steps)
        else:
            # Badly configured runs also return 0, so do not display the useless return code
            if self.has_error():
                return "Run<%s [FAILED]>" % (self.run_dir,)
            else:
                return "Run<%s>" % (self.run_dir,)

    def current_step(self):
        """
        Find the current simulation step.
        
        Returns: (int) The simulation step or -1 if it could not be found.

        """
        last_line = tail(path.join(self.run_dir, "out.txt"), 8)
        if not last_line:  # Empty file
            return -1
        if re.search("now at  t", last_line[-1]):
            return int(re.match(r".* n = *(.*?)$", last_line[-1]).group(1))
        elif " Osiris run completed normally\n" in last_line:
            return self.total_steps
        else:
            return -1

    def is_running(self):
        """Return True if the simulation is known to be running, or False otherwise."""
        if self.process is None:
            return False
        return self.process.is_running()

    def terminate(self):
        """Terminate the OSIRIS process (if running)."""
        if self.process is not None:
            if self.process.is_running():
                try:
                    self.process.terminate()
                except psutil.NoSuchProcess:
                    # The process has just terminated
                    pass
            else:
                logger.warning("The process had already stopped")
        else:
            logger.warning("Asked for termination of a Run with no known process")

    def kill(self):
        """
        Abruptly terminate the OSIRIS process (if running).
        
        The :func:`~duat.osiris.run.Run.terminate` method should be used instead to perform a cleaner exit.
        """
        if self.process is not None:
            if self.process.is_running():
                try:
                    self.process.kill()
                except psutil.NoSuchProcess:
                    # The process has just terminated
                    pass
            else:
                logger.warning("The process had already stopped")
        else:
            logger.warning("Asked for termination of a Run with no known process")

    def estimated_time(self):
        """
        Estimated time to end the simulation in seconds.
        
        The estimation uses a linear model and considers initialization negligible.
        The modification time of the os-stdin file is used in the calculation. If altered, estimation will be meaningless.
        
        Returns: (float) The estimation of the time to end the simulation or NaN if no estimation could be done.

        """
        if not self.is_running():  # Already finished
            return 0
        else:
            current = self.current_step()
            if current <= 0:  # If not started or error
                return float('nan')
            else:
                elapsed = time() - path.getmtime(path.join(self.run_dir, "os-stdin"))
                return elapsed * (self.total_steps / current - 1)

    def get_size(self):
        """Get the size of all run data in bytes."""
        return get_dir_size(self.run_dir)

    def has_error(self):
        """Search for common error messages in the output file."""
        # TODO: Cache result if reached execution with no error
        with open(path.join(self.run_dir, "out.txt"), "r") as f:
            text = f.read()
        # TODO: Optimize this search
        if "(*error*)" in text or re.search("Error reading .* parameters", text) or re.search("MPI_ABORT was invoked",
                                                                                              text):
            return True
        else:
            return False


def run_config(config, run_dir, prefix=None, clean_dir=True, blocking=None, force=None):
    """
    Initiate a OSIRIS run from a config instance.

    Args:
        config (`ConfigFile`): The instance describing the configuration file.
        run_dir (str): Folder where the run is carried.
        prefix (str): A prefix to run the command (e.g., "qsub", ...).
        clean_dir (bool): Whether to remove the files in the directory before execution.
        blocking (bool): Whether to wait for the run to finish.
        force (str): Set what to do if a running executable is found in the directory. Set to "ignore" to launch anyway,
                     possibly resulting in multiple instances running simultaneously; set to "kill" to terminate the
                     existing processes.

    Returns:
        (tuple): A Run instance describing the execution.

    """
    candidates = _find_running_exe(path.join(run_dir, "osiris"))
    if candidates:
        if force == "ignore":
            logger.warning("Ignored %d running exe found in %s" % (len(candidates), run_dir))
        elif force == "kill":
            logger.warning("Killing %d running exe found in %s" % (len(candidates), run_dir))
            for c in candidates:
                try:
                    psutil.Process(c).terminate()
                except psutil.NoSuchProcess:
                    pass  # If just ended
        else:
            logger.warning("Running exe found in %s. Aborting launch." % run_dir)
            return

    if clean_dir:
        for root, dirs, files in walk(run_dir):
            for f in files:
                remove(path.join(root, f))

        for root, dirs, files in walk(run_dir):
            for f in files:
                logger.warning("Could not remove file %s" % f)

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
    else:  # Sleep a little to check for quickly appearing errors and to allow the shell to start osiris
        sleep(0.2)

    # BEWARE: Perhaps under extreme circumstances, OSIRIS might have not started despite sleeping.
    # This could be solved reinstantiating RUN. Consider it a feature instead of a bug :P

    run = Run(run_dir)

    # Try to detect errors checking the output
    if run.has_error():
        logger.warning(
            "Error detected while launching %s.\nCheck out.txt there for more information or re-run in console." % run_dir)
    return run


# Try to guess the OSIRIS location:
for t in [path.join(path.expanduser("~"), "osiris", "bin"),
          path.join("usr", "local", "osiris", "bin")]:
    set_osiris_path(t, warn=False)
    if osiris_1d and osiris_2d and osiris_3d:
        break

if not (osiris_1d and osiris_2d and osiris_3d):
    if not (osiris_1d or osiris_2d or osiris_3d):
        logger.warning("Warning: no OSIRIS executables were found.")
    else:
        if not osiris_1d:
            logger.warning("Warning: osiris-1D.e not found.")
        if not osiris_2d:
            logger.warning("Warning: osiris-2D.e not found.")
        if not osiris_3d:
            logger.warning("Warning: osiris-3D.e not found.")
        logger.warning("Use the function set_osiris_path or set the variables run.osiris_1d and so to allow the run module to work.")
