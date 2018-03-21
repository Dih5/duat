# -*- coding: UTF-8 -*-
"""Run configuration files with OSIRIS."""

import re
import subprocess
from glob import glob
from os import path, remove, walk, listdir, environ
from shutil import copyfile
from time import sleep, time
from xml.etree import ElementTree

import psutil

from duat.plot import get_diagnostic_list as _get_diagnostic_list
from duat.common import ensure_dir_exists, ensure_executable, ifd, head, tail, logger, get_dir_size, human_order_key, \
    MPCaller, \
    Call
from duat.config import ConfigFile

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


class _memoized(object):
    def __init__(self, calls_to_live=5, time_to_live=2):
        self.cache = {}
        self.calls_to_live = calls_to_live
        self.time_to_live = time_to_live
        self._call_count = 1

    def __call__(self, func):
        def _memoized_function(*args):
            self.func = func
            now = time()
            try:
                value, last_update = self.cache[args]
                if self._call_count >= self.calls_to_live or now - last_update > self.time_to_live:
                    self._call_count = 1
                    raise AttributeError

                self._call_count += 1
                return value

            except (KeyError, AttributeError):
                value = self.func(*args)
                self.cache[args] = (value, now)
                return value

            except TypeError:
                return self.func(*args)

        return _memoized_function


# Memoize the process list so consecutive calls to _find_running_exe are faster (e.g., working with a list of Run).
@_memoized(calls_to_live=50, time_to_live=2)
def _get_process_list():
    return list(psutil.process_iter())


def _find_running_exe(exe):
    """Return the list of the pid of the processes of the argument executable"""
    candidates = []
    exe = path.abspath(exe)
    for proc in _get_process_list():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'exe'])
        except psutil.NoSuchProcess:
            pass
        else:
            if pinfo["exe"] and pinfo['exe'] == exe:
                candidates.append(pinfo['pid'])
    return candidates


_qstat_available = True


# Memoize the process list so consecutive calls to _get_grid_jobs are faster (e.g., working with a list of Run).
# Also, if qstat is not available, stop trying to call it.
@_memoized(calls_to_live=50, time_to_live=2)
def _general_qstat():
    global _qstat_available
    if not _qstat_available:
        return None
    try:
        return subprocess.check_output("qstat -xml 2> /dev/null", shell=True)
    except subprocess.CalledProcessError:
        _qstat_available = False
        return None  # qstat not available


def _get_job_tree_text(job_tree, property):
    """Get some parameter of an ElementTree representing a job """
    # Check to prevent AttributeError: 'NoneType' object has no attribute 'text'
    try:
        return job_tree.find(property).text
    except AttributeError:
        return ""


def _get_grid_jobs():
    """
    Get information of active jobs in qstat
    
    Returns:
        list of dict: A list of dictionaries, each with info of a running process. Available keys include:
            job_number (int): The number identifying the job.
            script (str): Path to script launching the job.
            submission_time (int): Unix time when the job was submitted.
            cwd (str): Path of the current working directory.
            
            
    """
    output = _general_qstat()
    if not output:
        return None
    tree = ElementTree.fromstring(output)
    jobs = []
    for job in tree.iter('job_list'):
        job_number = job[0].text
        output = subprocess.check_output("qstat -j %s -xml" % job[0].text, shell=True)
        job_tree = ElementTree.fromstring(output)[0][0]  # First index is djob_info, second is element
        time_str = _get_job_tree_text(job_tree, "JB_submission_time")
        try:
            start_time = int(job_tree.find("JB_ja_tasks")[0].find("JAT_start_time").text)
        except (TypeError, AttributeError):
            # TypeError if JB_ja_tasks not in the tree (which will happen if not started)
            # AttributeError if JAT_start_time not in the subtree
            start_time = 0
        jobs.append({
            "job_number": int(job_number),
            "script": _get_job_tree_text(job_tree, "JB_script_file"),
            "submission_time": int(time_str) if time_str else 0,
            "start_time": start_time,
            "cwd": _get_job_tree_text(job_tree, "JB_cwd"),
        })
    return jobs


class Run:
    """
    An osiris run.
    
    Attributes:
        run_dir (str): Directory where the run takes place.
        total_steps (int): Amount of time steps in the simulation.
        running_mode (str): Can be "local" if a local process was found, "grid" if a grid job was found, or ""
                            otherwise. A simulation launch in the grid but running in the local machine will be tagged
                            as "local" (since you can access its process).
        processes (list of psutil.Process): If running_mode is "local", representation of the processes running the
                                            simulation.
        job (dict): If running_mode is "grid", information about the job of the simulation.
        
    """

    def __init__(self, run_dir):
        """
        Create a Run instance.
        
        Args:
            run_dir (str): Path where the OSIRIS run takes place. An os-stdin file must exist there.
             
        Raises:
            FileNotFoundError: If no os-stdin is found.
            
        """
        self.run_dir = run_dir
        self._run_dir_name = path.basename(run_dir)

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
        self.running_mode = ""
        self.processes = None
        self.job = None

        self._update()

    def get_config(self):
        """Return a ConfigFile instance parsing the os-stdin file"""
        return ConfigFile.from_file(path.join(self.run_dir, "os-stdin"))

    def get_status(self):
        """
        Return the status of the run.

        Returns:
            str: The status of the run. Possible values are:

                * "RUNNING": Either a process is running the simulation (running_mode="local") or a qstat job was found (running_mode="grid").
                * "FAILED": An error was detected in the OSIRIS output.
                * "FINISHED": The simulation has successfully finished its execution.
                * "INCOMPLETE": The simulation appears to have started but not running now. You should check the output to try to understand what happened. If restart information was sent, you might want to run "continue.sh" to resume the simulation.
                * "NOT STARTED": The files were created, but the simulation is not running. Note you will see this state if the Run is queued in a MPCaller instance.

        """
        if self._is_running():
            return "RUNNING"
        elif self._has_error():
            # The run started but failed
            return "FAILED"
        elif self._is_finished():
            # The run was finished
            return "FINISHED"
        elif self.current_step() >= 0:
            # The run started at some point but was not completed
            return "INCOMPLETE"
        else:
            # The run did not start
            return "NOT STARTED"

    def __repr__(self):
        status = self.get_status()
        if status == "RUNNING":
            return "Run<%s [RUNNING (%s/%d)]>" % (self._run_dir_name, self.current_step(), self.total_steps)
        elif status == "FAILED":
            # The run started but failed
            return "Run<%s [FAILED]>" % (self._run_dir_name,)
        elif status == "FINISHED":
            # The run was finished
            return "Run<%s> [FINISHED]" % (self._run_dir_name,)
        elif status == "INCOMPLETE":
            # The run started at some point but was not completed
            return "Run<%s> [INCOMPLETE (%s/%d)]" % (self._run_dir_name, self.current_step(), self.total_steps)
        elif status == "NOT STARTED":
            # The run did not start
            return "Run<%s> [NOT STARTED]" % (self._run_dir_name,)
        else:
            return "Run<%s> [???]" % (self._run_dir_name,)

    def _update(self):
        """Update the running info using what is found at the moment."""
        candidates = _find_running_exe(path.join(self.run_dir, "osiris"))

        try:
            if not candidates:  # No process running found
                self.processes = None
                # Try to find a job in queue
                jobs = _get_grid_jobs()
                if not jobs:  # Either no qstat or empty list
                    self.running_mode = ""
                else:
                    script_path = path.abspath(path.join(self.run_dir, "start.sh"))
                    valid_jobs = list(filter(lambda j: j["script"] == script_path, jobs))
                    if valid_jobs:
                        if len(valid_jobs) > 1:
                            logger.warning("More than one grid job was found for the run.")
                        self.job = valid_jobs[0]
                        self.running_mode = "grid"
                    else:  # No queued job
                        self.running_mode = ""

            else:
                self.processes = list(map(psutil.Process, candidates))
                self.running_mode = "local"

        except psutil.NoSuchProcess:
            # If the processes have died before processing was completed.
            self.processes = None
            self.running_mode = ""

    def current_step(self):
        """
        Find the current simulation step.
        
        Returns:
            int: The simulation step or -1 if it could not be found.

        """
        try:
            last_line = tail(path.join(self.run_dir, "out.txt"), 8)
        except FileNotFoundError:
            return -1
        if not last_line:  # Empty file
            return -1
        if re.search("now at  t", last_line[-1]):
            # Unless the line was incomplete, there should be a match with:
            a = re.match(r".* n = *(.*?)$", last_line[-1])
            if a:
                return int(a.group(1))
            # Otherwise, try the previous one
            a = re.match(r".* n = *(.*?)$", last_line[-2])
            if a:
                return int(a.group(1))
            else:
                return -1  # Some error exists in the file

        elif " Osiris run completed normally\n" in last_line:
            return self.total_steps
        else:
            return -1

    def _is_running(self):
        """Return True if the simulation is known to be running, or False otherwise."""
        # Public interface is given by get_status instead.
        self._update()
        return True if self.running_mode else False

    def _is_finished(self):
        # Public interface is given by get_status instead.
        if path.isfile(path.join(self.run_dir, "TIMINGS", "timings.001")):
            return True
        else:
            return False

    def _has_error(self):
        """Search for common error messages in the output file."""
        # Public interface is given by get_status instead.
        # TODO: Cache result if reached execution with no error
        try:
            # If there is something in the error file:
            if path.getsize(path.join(self.run_dir, "err.txt")) > 0:
                return True
        except FileNotFoundError:
            pass
        try:
            with open(path.join(self.run_dir, "out.txt"), "r") as f:
                text = f.read()
            # TODO: Depending on the file size, the following might be better. Investigate this.
            # text = "".join(head(path.join(self.run_dir, "out.txt"), 300)+tail(path.join(self.run_dir, "out.txt"), 10))

            # TODO: The commented option is slower (even if compiled) than this one. Investigate.
            if "(*error*)" in text or re.search("Error reading .* parameters", text) or re.search(
                    "MPI_ABORT was invoked", text):
                # if re.search("\(\*error\*\)|Error reading .* parameters|MPI_ABORT was invoked",text):
                return True
            else:
                return False
        except FileNotFoundError:
            return False

    def terminate(self):
        """
        Terminate the OSIRIS processes (if running).

        If runnning is "local", sends SIGINT to the processes. If "grid", calls qdel.

        Raises:
            subprocess.CalledProcessError: If using a grid and qdel fails.

        """
        self._update()
        if self.running_mode == "local":
            for process in self.processes:
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    # The process has just terminated
                    # In multiprocess run this is likely to happen when other processes stops.
                    pass
        elif self.running_mode == "grid":
            subprocess.check_call("qdel %d" % self.job["job_number"], shell=True)
            pass
        else:
            logger.warning("Asked for termination of a Run not known to be running.")

    def kill(self):
        """
        Abruptly terminate the OSIRIS processes (if running).
        
        The :func:`~duat.osiris.run.Run.terminate` method should be used instead to perform a cleaner exit.

        If runnning is "local", sends SIGKILL to the processes. If "grid", calls qdel.

        Raises:
            subprocess.CalledProcessError: If using a grid and qdel fails.

        """
        self._update()
        if self.running_mode == "local":
            for process in self.processes:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    # The process has just terminated
                    # In multiprocess run this is likely to happen when other processes stops.
                    pass
        elif self.running_mode == "grid":
            subprocess.check_call("qdel %d" % self.job["job_number"], shell=True)
            pass
        else:
            logger.warning("Asked for termination of a Run not known to be running.")

    def estimated_time(self):
        """
        Estimated time to end the simulation in seconds.
        
        The estimation uses a linear model and considers initialization negligible.
        For local runs, the start time of a process is used in the calculation.
        For grid runs, the start time of the job is used instead.

        If the run was resumed, the estimation will be wrong.
        
        Returns: 
            float: The estimation of the time to end the simulation or NaN if no estimation could be done.

        """
        self._update()
        if not self.running_mode:
            return 0 if self._is_finished() else float("nan")
        elif self.running_mode == "local":
            start = self.processes[0].create_time()
        elif self.running_mode == "grid":
            start = self.job["start_time"]
            if start == 0:
                # Queued, but not started
                return float("nan")
        else:
            logger.warning("Invalid running_mode attribute")
            return float("nan")
        current = self.current_step()
        if current <= 0:  # If not dumped yet or error
            return float('nan')
        else:
            elapsed = time() - start
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


def _execute_run(prefix, osiris_path, run_dir):
    """Execute and wait for a run to finish"""
    # Cf. run_config
    p = subprocess.Popen(prefix + osiris_path + " > out.txt 2> err.txt", shell=True, cwd=path.abspath(run_dir))
    p.wait()


def run_config(config, run_dir, prefix=None, clean_dir=True, blocking=None, force=None, mpcaller=None):
    """
    Initiate a OSIRIS run from a config instance.

    Args:
        config (`ConfigFile`): The instance describing the configuration file.
        run_dir (str): Folder where the run is carried.
        prefix (str): A prefix to run the command. If None, "mpirun -np X " will be used when a config with multiple
                      nodes is provided.
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
    # Automatic mpirun# Find the needed amount of nodes
    n = config.get_nodes()
    if prefix is None:
        prefix = "mpirun -np %d " % n if n > 1 else ""
    elif prefix[-1] != " ":
        prefix += " "

    # Search for possibly running processes
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

    # Clean if needed
    if clean_dir:
        for root, dirs, files in walk(run_dir):
            for f in files:
                remove(path.join(root, f))

        for root, dirs, files in walk(run_dir):
            for f in files:
                logger.warning("Could not remove file %s" % f)

    # If the run is restartable, make the if_restart variable explicit
    if "restart" in config and "ndump_fac" in config["restart"] and config["restart"]["ndump_fac"]:
        if "if_restart" not in config["restart"]:
            config["restart"]["if_restart"] = False  # This is the default value

    # copy the input file
    ensure_dir_exists(run_dir)
    config.write(path.join(run_dir, "os-stdin"))

    # Copy the osiris executable
    osiris_path = path.abspath(path.join(run_dir, "osiris"))
    osiris = ifd(config.get_d(), osiris_1d, osiris_2d, osiris_3d)
    copyfile(osiris, osiris_path)
    ensure_executable(osiris_path)

    # Create a start.sh file to ease manual launch
    with open(path.join(run_dir, "start.sh"), 'w') as f:
        f.write("#!/bin/bash\n%s./osiris > out.txt 2> err.txt" % prefix)
    ensure_executable(path.join(run_dir, "start.sh"))

    # Create a continue.sh file to ease manual relaunch of aborted executions
    with open(path.join(run_dir, "continue.sh"), 'w') as f:
        f.write("#!/bin/bash"
                "\nsed -i -e \"s/if_restart = .false./if_restart = .true./g\" os-stdin"
                "\n./%s osiris >> out.txt 2>> err.txt" % prefix)
    ensure_executable(path.join(run_dir, "continue.sh"))

    if mpcaller is not None:
        run = Run(run_dir)
        # Set the run instance to update the process info when the call is made.
        mpcaller.add_call(Call(_execute_run, prefix, osiris_path, run_dir))
        return run
    else:
        proc = subprocess.Popen(prefix + osiris_path + " > out.txt 2> err.txt", shell=True, cwd=path.abspath(run_dir))
        if blocking:
            proc.wait()
        else:  # Sleep a little to check for quickly appearing errors and to allow the shell to start osiris
            sleep(0.2)

        # BEWARE: Perhaps under extreme circumstances, OSIRIS might have not started despite sleeping.
        # This could be solved calling the update method of the Run instance.
        # Consider this a feature instead of a bug :P

        run = Run(run_dir)

        # Try to detect errors checking the output
        if run._has_error():
            logger.warning(
                "Error detected while launching %s.\nCheck out.txt and err.txt for more information or re-run in console." % run_dir)
        return run


def run_config_grid(config, run_dir, prefix=None, run_name="osiris_run", remote_dir=None, clean_dir=True, prolog="",
                    epilog=""):
    """
    Queue a OSIRIS run in a compatible grid (e.g., Sun Grid Engine).

    Args:
        config (`ConfigFile`): The instance describing the configuration file.
        run_dir (str): Folder where the run will be carried.
        prefix (str): A prefix to run the command. If None, "mpirun -np X " will be used when a config with multiple
                      nodes is provided.
        run_name (str): Name of the job in the engine.
        remote_dir (str): If provided, a remote directory where the run will be carried, which might be only available
                          in the node selected by the engine. Note that if this option is used, the returned Run
                          instance will not access the remote_dir, but the run_dir. If the remote node is unable to
                          access the path (trying to create it if needed), OSIRIS will be started in the run dir and
                          errors will be logged by the queue system.
        clean_dir (bool): Whether to remove the files in the directory before execution.
        prolog (str): Shell code to run before calling OSIRIS (but once in the remote_dir if asked for).
        epilog (str): Shell code to run after calling OSIRIS.

    Returns:
        Run: A Run instance describing the execution.

    """
    # Clean if needed
    if clean_dir:
        for root, dirs, files in walk(run_dir):
            for f in files:
                remove(path.join(root, f))

        for root, dirs, files in walk(run_dir):
            for f in files:
                logger.warning("Could not remove file %s" % f)

    # Find the needed amount of nodes
    n = config.get_nodes()
    if prefix is None:
        prefix = "mpirun -np %d " % n if n > 1 else ""
    elif prefix[-1] != " ":
        prefix += " "

    # copy the input file
    ensure_dir_exists(run_dir)
    config.write(path.join(run_dir, "os-stdin"))

    # Copy the osiris executable
    osiris_path = path.abspath(path.join(run_dir, "osiris"))
    osiris = ifd(config.get_d(), osiris_1d, osiris_2d, osiris_3d)
    copyfile(osiris, osiris_path)
    ensure_executable(osiris_path)

    # Create a start.sh file with the launch script
    s = "".join(["#!/bin/bash\n#\n#$ -cwd\n#$ -S /bin/bash\n#$ -N %s\n" % run_name,
                 "#$ -pe smp %d\n" % n if n > 1 else "",
                 "#\n",
                 "NEW_DIR=%s\nmkdir -p $NEW_DIR\ncp -r . $NEW_DIR\ncd $NEW_DIR\n" % remote_dir if remote_dir else "",
                 prolog + "\n",
                 "\n%s./osiris > out.txt 2> err.txt\n" % prefix,
                 epilog + "\n"])

    with open(path.join(run_dir, "start.sh"), 'w') as f:
        f.write(s)
    ensure_executable(path.join(run_dir, "start.sh"))

    subprocess.Popen("qsub " + path.abspath(path.join(run_dir, "start.sh")), shell=True, cwd=path.abspath(run_dir))

    return Run(run_dir)


def run_variation(config, variation, run_base, caller=None, on_existing=None, **kwargs):
    """
    Make consecutive calls to :func:`~duat.osiris.run.run_config` with ConfigFiles generated from a variation.
    
    Args:
        config (`ConfigFile`): Base configuration file.
        variation (`Variation`): Description of the variations to apply.
        run_base (str): Path to the directory where the runs will take place, each in a folder named var_number.
        caller (int or `MPCaller`): If supplied, the calls will be managed by a MPCaller instance. If an int is provided
                                    an MPCaller with such a number of threads will be created. Provide an instance if
                                    interested in further controlling.
        on_existing (str): Action to do if a run of the variation exists. Only the names of the subfolders are used for
                           this purpose, which means the run could be different if the variation or the path have
                           changed. Set to "ignore" to leave untouched existing runs or set to "overwrite" to delete the
                           data and run a new instance. Default is like "ignore" but raising a warning.
        **kwargs: Keyword arguments to pass to :func:`~duat.osiris.run.run_config`

    Returns:
        list of Run: List with the Run instances in the variation directory.

    """
    r_list = []

    if on_existing is not None:
        if not isinstance(on_existing, str):
            raise ValueError("Invalid on_existing parameter")
        on_existing = on_existing.lower()
        if on_existing not in ["ignore", "overwrite"]:
            raise ValueError("Invalid on_existing parameter")

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
            var_dir = path.join(run_base, "var_" + str(i))
            if path.isfile(path.join(var_dir, "os-stdin")):
                # If the item existed
                if on_existing is None:
                    logger.warning("Skipping existing variation item " + var_dir)
                elif on_existing == "ignore":
                    pass
                else:  # overwrite
                    run_config(c, var_dir, mpcaller=_caller, **kwargs)
            else:
                # The item did not exist
                run_config(c, var_dir, mpcaller=_caller, **kwargs)
            r_list.append(Run(var_dir))

        if isinstance(caller, int):
            # If the MPCaller was created in this method, threads should die after execution
            _caller.wait_calls(blocking=False)
            # Nevertheless, processes seems not to be discarded until a new call to this method is made
    return r_list


def run_variation_grid(config, variation, run_base, run_name="os-var_", remote_dir=None, on_existing=None, **kwargs):
    """
    Make consecutive calls to :func:`~duat.osiris.run.run_config_grid` with ConfigFiles generated from a variation.

    Args:
        config (`ConfigFile`): Base configuration file.
        variation (`Variation`): Description of the variations to apply.
        run_base (str): Path to the directory where the runs will take place, each in a folder named var_number.
        run_name (str): Prefix to the name to use in the grid system.
        remote_dir (str): If provided, a remote directory where the runs will be carried, which might be only available
                          in the node selected by the engine. See :func:`~duat.osiris.run.run_config_grid`.
        on_existing (str): Action to do if a run of the variation exists. Only the names of the subfolders are used for
                           this purpose, which means the run could be different if the variation or the path have
                           changed. Set to "ignore" to leave untouched existing runs or set to "overwrite" to delete the
                           data and run a new instance. Default is like "ignore" but raising a warning.
        **kwargs: Keyword arguments to pass to :func:`~duat.osiris.run.run_config_grid`.

    Returns:
        list of Run: List with the Run instances in the variation directory.

    """
    r_list = []

    if on_existing is not None:
        if not isinstance(on_existing, str):
            raise ValueError("Invalid on_existing parameter")
        on_existing = on_existing.lower()
        if on_existing not in ["ignore", "overwrite"]:
            raise ValueError("Invalid on_existing parameter")

    for i, c in enumerate(variation.get_generator(config)):
        var_name = "var_" + str(i)
        var_dir = path.join(run_base, var_name)
        if path.isfile(path.join(var_dir, "os-stdin")):
            # If the item existed
            if on_existing is None:
                logger.warning("Skipping existing variation item " + var_dir)
            elif on_existing == "ignore":
                pass
            else:  # overwrite
                if remote_dir:
                    run_config_grid(c, var_dir, run_name=run_name + str(i),
                                    remote_dir=path.join(remote_dir, var_name), **kwargs)
                else:
                    run_config_grid(c, var_dir, run_name=run_name + str(i), **kwargs)
        else:
            # The item did not exist
            if remote_dir:
                run_config_grid(c, var_dir, run_name=run_name + str(i), remote_dir=path.join(remote_dir, var_name),
                                **kwargs)
            else:
                run_config_grid(c, var_dir, run_name=run_name + str(i), **kwargs)
        r_list.append(Run(var_dir))

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
