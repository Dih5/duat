# -*- coding: UTF-8 -*-

import os
import re
import sys


def ifd(d, v1, v2, v3):
    """
    Conditionally select a value depending on the given dimension number.

    Args:
        d (int): Dimension number (1, 2, or 3).
        v1: The value if d = 1
        v2: The value if d = 2
        v3: The value if d = 3

    Returns:
        v1, v2 or v3

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
