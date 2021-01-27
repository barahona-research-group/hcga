"""Utils functions."""
import logging

import dill
import pandas as pd

import multiprocessing as mp
import multiprocessing.queues as mpq
from hcga.graph import Graph

L = logging.getLogger(__name__)


class TimeoutError2(Exception):
    def __init__(self, func, timeout):
        self.t = timeout
        self.fname = func.__name__

    def __str__(self):
        return f"function '{self.fname}' timed out after {self.t}s"


def _lemmiwinks(func, args, kwargs, q):
    """lemmiwinks crawls into the unknown"""
    print(func, dill.loads(func))
    q.put(dill.loads(func)(*args, **kwargs))


def _eval(func, args, timeout=1):

    q_worker = mp.Manager().Queue()
    proc = mp.Process(target=_lemmiwinks, args=(dill.dumps(func), args, {}, q_worker))
    proc.start()
    try:
        out = q_worker.get(timeout=timeout)
        return out
    except mpq.Empty:
        raise TimeoutError(func, timeout)
    finally:
        try:
            proc.terminate()
            q_worker.close()
        except:
            pass


class TimeoutError(Exception):
    """Timeout exception."""


def timeout_handler(signum, frame):
    """Function to raise timeout exception."""
    raise TimeoutError


def get_trivial_graph(n_node_features=0):
    """Generate a trivial graph for internal purposes."""
    nodes = pd.DataFrame([0, 1, 2])
    if n_node_features > 0:
        nodes["features"] = 3 * [n_node_features * [0.0]]
    edges = pd.DataFrame()
    edges["start_node"] = [0, 1, 2]
    edges["end_node"] = [1, 2, 0]
    return Graph(nodes, edges, 0)
