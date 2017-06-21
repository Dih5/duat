# -*- coding: UTF-8 -*-
"""Common tools."""

import os
import re
from multiprocessing import Process, Queue
import logging


def ifd(d, v1, v2, v3):
    """
    Conditionally select a value depending on the given dimension number.

    Args:
        d (int): Dimension number (1, 2, or 3).
        v1: The value if d = 1.
        v2: The value if d = 2.
        v3: The value if d = 3.

    Returns:
        v1, v2 or v3.

    """
    if d == 1:
        return v1
    elif d == 2:
        return v2
    elif d == 3:
        return v3
    else:
        raise ValueError("Invalid dimension: %s." % d)


def human_order_key(text):
    """
    Key function to sort in human order.

    """
    # This is based in http://nedbatchelder.com/blog/200712/human_sorting.html
    return [int(c) if c.isdigit() else c for c in re.split('(\d+)', text)]


def ensure_dir_exists(path):
    """
    Ensure a directory exists, creating it if needed.

    Args:
        path (str): The path to the directory.

    Raises:
        OSError: An error occurred when creating the directory.

    """
    try:
        # Will fail either if exists or unable to create it
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            # Directory did [probably] exist
            pass
        else:
            # There was an error on creation, so make sure we know about it
            raise OSError("Unable to create directory " + path)


def get_dir_size(dir_path):
    """Get the size of a directory in bytes."""
    total_size = 0
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    return total_size


def ensure_executable(path, all_users=None):
    """
    Ensure a file is executable.

    Args:
        path (str): the path to the file.
        all_users (bool): whether it should be make executable for the user or for all users.

    """
    st = os.stat(path)
    os.chmod(path, st.st_mode | (0o111 if all_users else 0o100))


class Call:
    """Objectization of a call to be made. When an instance is called, such a call will be made."""

    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.fn(*self.args, **self.kwargs)


def _caller(q):
    """Execute calls from a Queue until its value is 'END'."""
    while True:
        data = q.get()
        if data == "END":
            break
        else:
            data()  # Call the supplied argument (lambda o Call instance)


class MPCaller:
    """
    MultiProcessing Caller. Makes calls using multiple subprocesses.
    
    Attributes:
        processes (list of multiprocessing.Process): Processes managed by the instance.
        
    """

    def __init__(self, num_threads=2):
        self._queue = Queue()
        self.processes = []
        self.spawn_threads(num_threads)

    def __repr__(self):
        return "MPCaller<%d threads, %d tasks in _queue>" % (len(self.processes), self._queue.qsize())

    def spawn_threads(self, num_threads):
        """Create the required number of processes and add them to the caller.
        
        This does not remove previously created processes.
        
        """
        for _ in range(num_threads):
            t = Process(target=_caller, args=(self._queue,))
            t.daemon = True
            t.start()
            self.processes.append(t)

    def add_call(self, call):
        """
        Add a call to the instance's stack.
        
        Args:
            call (Callable): A function whose call method will be invoked by the processes. Consider using lambda
                             functions or a :class:`Call` instance.

        """
        self._queue.put(call)

    def wait_calls(self, blocking=True, respawn=False):
        """
        Ask all processes to consume the queue and stop after that.
        
        Args:
            blocking (bool): Whether to block the call, waiting for processes termination.
            respawn (bool): If blocking is True, this indicates whether to respawn the threads after the calls finish.
                            If blocking is not True this is ignored (no automatic respawn if non-blocking).

        """
        num_threads = len(self.processes)
        for _ in range(num_threads):
            self._queue.put("END")

        if blocking:
            for t in self.processes:
                t.join()
            self.processes = []
            if respawn:
                self.spawn_threads(num_threads)

    def abort(self, interrupt=False):
        """
        Remove all queued calls and ask processes to stop.

        Args:
            interrupt: If True, terminate all processes.

        """
        for _ in range(len(self.processes)+1):
            self._queue.put("END")
        while True:
            data = self._queue.get()
            if data == "END":
                break
            # Else do nothing
        if interrupt:
            for t in self.processes:
                t.terminate()
            # If the killed process was trying to use the Queue it could have corrupted
            # Just in case, create a new one
            self._queue = Queue()


def tail(path, lines=1, _step=4098):
    """
    Get the last lines of a file.
    
    Args:
        path (str): Path to the file to read.
        lines (int): Number of lines to read.
        _step (int): Size of the step used in the search.

    Returns:
        :obj:`list` of :obj:`str`: The lines found.

    """
    # Adapted from glenbot's answer to:
    # http://stackoverflow.com/questions/136168/get-last-n-lines-of-a-file-with-python-similar-to-tail
    f = open(path, "r")
    lines_found = []
    block_counter = -1
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _step, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            # read all and exit loop
            f.seek(0)
            lines_found = f.readlines()
            break

        lines_found = f.readlines()

        if len(lines_found) > lines:
            break
        block_counter -= 1

    return lines_found[-lines:]


logging.basicConfig(level=logging.INFO)
logger = logging.Logger("duat")
