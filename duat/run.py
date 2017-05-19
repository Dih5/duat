# -*- coding: UTF-8 -*-
"""Run configuration files with OSIRIS."""

import re
import subprocess
from glob import glob
from os import path, remove, walk, listdir, environ
from shutil import copyfile
from time import sleep, time

import psutil

from duat.plot import get_diagnostic_list as _get_diagnostic_list
from duat.common import ensure_dir_exists, ensure_executable, ifd, tail, logger, get_dir_size, human_order_key, MPCaller, \
    Call

# Path to osiris executables - guessed later in the code
osiris_1d = ""
"""str: Path to the osiris-1D.e file"""
osiris_2d = ""
"""str: Path to the osiris-2D.e file"""
osiris_3d = ""
"""str: Path to the osiris-3D.e file"""


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
        process (psutil.Process): Representation of the process running the simulation. If no process is found it will
                                  be None. Only in that case, methods that update the state of simulation will check if
                                  a process has spawned when called.
        
    Notes:
        Only single-process runs are supported at the moment. Resuming runs are neither supported yet.
        
    """

    def __init__(self, run_dir):
        """
        Create a Run instance.
        
        Args:
            run_dir (str): Path where the OSIRIS run takes place. An os-stdin file must exist there.
             
        Raises:
            ValueError: If no os-stdin is found.
            
        """
        self.run_dir = run_dir
        self._run_dir_name = path.basename(run_dir)
        try:
            with open(path.join(run_dir, "os-stdin"), "r") as f:
                text = f.read()
            r = re.match(r".*time_step(.*?){(.*?)dt(.*?)=(.*?),(.*?)}", text, re.DOTALL + re.MULTILINE)
            if not r:
                raise ValueError("No dt found in os-stdin.")
            dt = float(r.group(4))
            r = re.match(r".*time(.*?){(.*?)tmin(.*?)=(.*?),(.*?)}", text, re.DOTALL + re.MULTILINE)
            t_min = float(r.group(4)) if r else 0.0
            r = re.match(r".*time(.*?){(.*?)tmax(.*?)=(.*?),(.*?)}", text, re.DOTALL + re.MULTILINE)
            if not r:
                raise ValueError("No tmax found in os-stdin. Default value 0.0 is trivial.")
            t_max = float(r.group(4))

            self.total_steps = int((t_max - t_min) // dt) + 1
        except FileNotFoundError:
            raise ValueError("No os-stdin file in %s" % run_dir)

        self.update()

    def __repr__(self):
        if self.is_running():
            return "Run<%s [RUNNING (%s/%d)]>" % (self._run_dir_name, self.current_step(), self.total_steps)
        elif self.has_error():
            # The run started but failed
            return "Run<%s [FAILED]>" % (self._run_dir_name,)
        elif self.is_finished():
            # The run was finished
            return "Run<%s> [FINISHED]" % (self._run_dir_name,)
        elif self.current_step() >= 0:
            # The run started at some point but was not completed
            return "Run<%s> [INCOMPLETE (%s/%d)]" % (self._run_dir_name, self.current_step(), self.total_steps)
        else:
            # The run did not start
            return "Run<%s> [NOT STARTED]" % (self._run_dir_name,)

    def update(self):
        """Update the process info using what is found at the moment."""
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

    def current_step(self):
        """
        Find the current simulation step.
        
        Returns:
            int: The simulation step or -1 if it could not be found.

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
            # Try to find a process only if none was found when the instance was created
            self.update()
            if self.process is None:
                return False
            else:
                return self.process.is_running()
        return self.process.is_running()

    def is_finished(self):
        if self.is_running():
            return False
        else:
            if path.isfile(path.join(self.run_dir, "TIMINGS", "timings.001")):
                return True
            else:
                return False

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
        
        Returns: 
            float: The estimation of the time to end the simulation or NaN if no estimation could be done.

        """
        if not self.is_running():
            return 0 if self.is_finished() else float("nan")
        else:
            current = self.current_step()
            if current <= 0:  # If not dumped yet or error
                return float('nan')
            else:
                elapsed = time() - path.getmtime(path.join(self.run_dir, "os-stdin"))
                return elapsed * (self.total_steps / current - 1)

    def real_time(self):
        """Find the total time in seconds taken by the simulation if it has finished, otherwise returning nan."""
        try:
            # TODO: Update for resuming runs
            with open(path.join(self.run_dir, "TIMINGS", "timings.001"), "r") as f:
                text = f.read()
            r = re.match(r" Total time for loop was(?: *)(.*?)(?: *)seconds", text, re.DOTALL + re.MULTILINE)
            if not r:
                logger.warning("Bad format in timings file. The real time could not be read.")
                return float("nan")
            else:
                return float(r.group(1))
        except FileNotFoundError:
            return float("nan")

    def get_size(self):
        """Get the size of all run data in bytes."""
        return get_dir_size(self.run_dir)

    def has_error(self):
        """Search for common error messages in the output file."""
        # TODO: Cache result if reached execution with no error
        try:
            with open(path.join(self.run_dir, "out.txt"), "r") as f:
                text = f.read()
            # TODO: Optimize this search
            if "(*error*)" in text or re.search("Error reading .* parameters", text) or re.search(
                    "MPI_ABORT was invoked",
                    text):
                return True
            else:
                return False
        except FileNotFoundError:
            return False

    def get_diagnostic_list(self):
        """
        Create a list with the diagnostic found in the given Run.

        Returns:
            :obj:`list` of :obj:`Diagnostic`: List of the diagnostic found.

        """
        return _get_diagnostic_list(self.run_dir)


def open_run_list(base_path, filter=None):
    """
    Create a Run instance for each of the subdirectories in the given path.
    
    Args:
        base_path (str): Path where the runs are found.
        filter (str): Filter the directories using a UNIX-like pattern.

    Returns:
        list of `Run`: A list with the Run instances, ordered so their paths are in human order.

    """
    dir_list = listdir(base_path)
    if not dir_list:
        return []
    if filter is not None:
        filter_list = glob(path.join(base_path, filter))
        filter_list = [path.basename(x) for x in filter_list]
        dir_list = [x for x in dir_list if x in filter_list]
        if not dir_list:
            return []
    dir_list.sort(key=human_order_key)
    return [Run(x) for x in [path.join(base_path, y) for y in dir_list]]


def _execute_run(prefix, osiris_path, run_dir, run_object=None):
    """Execute and wait for a run to finish, optionally updating a Run instance when the call is made."""
    # Cf. run_config
    p = subprocess.Popen(prefix + osiris_path + " > out.txt 2> err.txt", shell=True, cwd=path.abspath(run_dir))
    if run_object is not None:
        # TODO: The run_object updates in the process created by MPCaller, where it is useless.
        # Probably it is not worth sharing the object. User may call the Run.update method if interested.
        sleep(0.2)
        run_object.update()
    p.wait()


def run_config(config, run_dir, prefix=None, clean_dir=True, blocking=None, force=None, mpcaller=None):
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
        mpcaller (MPCaller): An instance controlling multithreaded calls. If supplied, all calls will be handled by this
                     instance and the blocking parameter will be ignored.

    Returns:
        tuple: A Run instance describing the execution.

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

    if mpcaller is not None:
        run = Run(run_dir)
        # Set the run instance to update the process info when the call is made.
        mpcaller.add_call(Call(_execute_run, prefix, osiris_path, run_dir, run_object=run))
        return run
    else:
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


def run_variation(config, variation, run_base, caller=None, **kwargs):
    """
    Make consecutive calls to :func:`~duat.osiris.run.run_config` with ConfigFiles generated from a variation.
    
    Args:
        config (`ConfigFile`): Base configuration file.
        variation (`Variation`): Description of the variations to apply.
        run_base (str): Path to the directory where the runs will take place, each in a folder named var_number.
        caller (int or `MPCaller`): If supplied, the calls will be managed by a MPCaller instance. If an int is provided
                                    an MPCaller with such a number of threads will be created. Provide an instance if
                                    interested in further controlling.
        **kwargs: Keyword arguments to pass to :func:`~duat.osiris.run.run_config`

    Returns:
        list: List with the return values of each call.

    """
    r_list = []

    if caller is None:
        for i, c in enumerate(variation.get_generator(config)):
            r = run_config(c, path.join(run_base, "var_" + str(i)), **kwargs)
            r_list.append(r)
    else:
        if isinstance(caller, int):
            _caller = MPCaller(caller)
        else:
            # Otherwise assume it was a MPCaller instance
            _caller = caller

        for i, c in enumerate(variation.get_generator(config)):
            r = run_config(c, path.join(run_base, "var_" + str(i)), mpcaller=_caller, **kwargs)
            r_list.append(r)

        if isinstance(caller, int):
            # If the MPCaller was created in this method, threads should die after execution
            _caller.wait_calls(blocking=False)
            # Nevertheless, processes seems not to be discarded until a new call to this method is made
    return r_list


# Try to guess the OSIRIS location:
_candidates = []
if "OSIRIS_PATH" in environ:
    _candidates.append(environ["OSIRIS_PATH"])
_candidates.append(path.join(path.expanduser("~"), "osiris", "bin"))
_candidates.append(path.join("usr", "local", "osiris", "bin"))

for t in _candidates:
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

    logger.warning("Set the environment variable OSIRIS_PATH to a folder where the OSIRIS executables with names "
                   "osiris-1D.e and so on are found.\n"
                   "You can also use run.set_osiris_path or set the variables run.osiris_1d (and so on).")
