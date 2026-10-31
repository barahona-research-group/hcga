"""Utils functions."""

import logging
import multiprocessing
import multiprocessing.pool
import signal
import threading
from contextlib import contextmanager

import pandas as pd

from hcga.graph import Graph

L = logging.getLogger(__name__)


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(multiprocessing.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


class FeatureTimeout(multiprocessing.TimeoutError, TimeoutError):
    """Raised when a feature evaluation exceeds its time limit.

    Subclasses both multiprocessing.TimeoutError and the builtin TimeoutError.
    """


def _can_use_alarm():
    """SIGALRM-based time limits need setitimer (POSIX) and the main thread."""
    return hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()


@contextmanager
def time_limit(seconds):
    """Raise FeatureTimeout in the enclosed block after the given number of seconds.

    Uses SIGALRM in the current process, so it costs a few microseconds per call. Without
    setitimer (Windows) or outside the main thread no limit is applied. A single long call
    into C code is only interrupted once it returns to the Python interpreter.
    """
    if not seconds or not _can_use_alarm():
        yield
        return

    def _handler(signum, frame):  # pylint: disable=unused-argument
        raise FeatureTimeout(f"timed out after {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def timeout_eval(func, args, timeout=None):
    """Evaluate a function within a given timeout period.

    Args:
        func: The function to call.
        args: Arguments to pass to the function.
        timeout: The timeout period in seconds (None or 0 for no limit).

    Returns:
        The function's result. Exceptions raised by the function propagate, and a timeout
        raises FeatureTimeout (a subclass of multiprocessing.TimeoutError).
    """
    with time_limit(timeout):
        return func(*args)


def get_trivial_graph(n_node_features=0):
    """Generate a trivial graph for internal purposes."""
    nodes = pd.DataFrame([0, 1, 2])
    if n_node_features > 0:
        nodes["features"] = 3 * [n_node_features * [0.0]]
    edges = pd.DataFrame()
    edges["start_node"] = [0, 1, 2]
    edges["end_node"] = [1, 2, 0]
    return Graph(nodes, edges, 0)
