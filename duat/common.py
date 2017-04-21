# -*- coding: UTF-8 -*-
"""Common tools."""

import os
import re
from multiprocessing import Process, Queue


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
        path: The path to the directory.

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


def ensure_executable(path, all_users=None):
    """
    Ensure a file is executable.

    Args:
        path (str): the path to the file.
        all_users (bool): whether it should be make executable for the user or for all users.

    """
    st = os.stat(path)
    os.chmod(path, st.st_mode | (0o111 if all_users else 0o100))


def daemonize(func):
    """
    Run a function as a daemon.

    Args:
        func (`Callable`): The function to run as a daemon.

    Raises:
        OSError: An error occurred performing a fork.

    """
    # Create a son
    pid = os.fork()
    if pid > 0:  # Parent
        return
    # We are the son
    os.setsid()

    # Create a grandson

    pid = os.fork()
    if pid > 0:  # Son (not grandson)
        os._exit(os.EX_OK)  # So no cleaning is performed

    # We are the grandson
    func()
    os._exit(os.EX_OK)


class Call:
    """Objectization of a call to be made"""

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
    """MultiProcessing Caller. Makes calls using multiple subprocesses."""

    def __init__(self, num_threads=2):
        self.q = Queue()
        self.processes = []
        self.spawn_threads(num_threads)

    def spawn_threads(self, num_threads):
        """Create the required number of threads"""
        for _ in range(num_threads):
            t = Process(target=_caller, args=(self.q,))
            t.daemon = True
            t.start()
            self.processes.append(t)

    def add_call(self, call):
        """Add a call to its stack"""
        self.q.put(call)

    def wait_calls(self):
        """Ask all processes to consume the queue and end.

        After this method is called no threads will remain. Create another instance or call spawn_threads if needed.
        """
        for _ in range(len(self.processes)):
            self.q.put("END")
        for t in self.processes:
            t.join()
        self.processes = []
