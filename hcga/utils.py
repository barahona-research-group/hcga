"""Utils functions."""
import logging
import multiprocessing as mp
import multiprocessing.queues as mpq
import traceback

import dill
import pandas as pd

from hcga.graph import Graph

L = logging.getLogger(__name__)


class NoDaemonProcess(mp.Process):
    """Class that represents a non-daemon process"""

    def _get_daemon(self):  # pylint: disable=no-self-use
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(mp.pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


def _lemmiwinks(func, args, kwargs, q):
    """lemmiwinks crawls into the unknown"""
    q.put(dill.loads(func)(*args, **kwargs))


class Process(mp.Process):
    """
    Class which returns child Exceptions to Parent.
    https://stackoverflow.com/a/33599967/4992248
    """

    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._parent_conn, self._child_conn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._child_conn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._child_conn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._parent_conn.poll():
            self._exception = self._parent_conn.recv()
        return self._exception


def timeout_eval(func, args, timeout=None):
    """Evaluate a function and kill it is it takes longer than timeout.

    If timeout is None, a simple evaluation will take place.
    """
    if timeout is None:
        return func(*args)
    q_worker = mp.Queue()
    proc = Process(target=_lemmiwinks, args=(dill.dumps(func), args, {}, q_worker))
    proc.start()

    if proc.exception:
        exc = proc.exception[0]
        proc.terminate()
        q_worker.close()
        raise exc
    try:
        return q_worker.get(timeout=timeout)
    except mpq.Empty:
        raise TimeoutError
    finally:
        proc.terminate()
        q_worker.close()


class TimeoutError(Exception):
    """Timeout exception."""


def get_trivial_graph(n_node_features=0):
    """Generate a trivial graph for internal purposes."""
    nodes = pd.DataFrame([0, 1, 2])
    if n_node_features > 0:
        nodes["features"] = 3 * [n_node_features * [0.0]]
    edges = pd.DataFrame()
    edges["start_node"] = [0, 1, 2]
    edges["end_node"] = [1, 2, 0]
    return Graph(nodes, edges, 0)
