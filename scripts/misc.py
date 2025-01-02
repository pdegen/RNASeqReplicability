import glob
import random
import pickle
import warnings
import sys
import os
import pandas as pd
import numpy as np
import cProfile as profile


def open_table(file):
    """Returns table, agnostic whether file is csv or feather format"""

    ext = file.split(".")[-1]
    if ext == "feather":
        tab = pd.read_feather(file)

        if "Row" in tab:
            tab.set_index("Row", inplace=True)
            #synthetic data: index should be stored as dtype int
            if tab.sort_index().index[0] == "1":
                tab.index = tab.index.astype(int)
        elif "index" in tab:
            tab.set_index("index", inplace=True)
        elif "Term ID" in tab and not tab["Term ID"][0].startswith("hsa"):
            tab.set_index("Term ID", inplace=True)
        elif "Term" in tab:
            tab.set_index("Term", inplace=True)

    elif ext == "csv":
        tab = pd.read_csv(file, index_col=0)

    else:
        try:
            file += ".feather"
            tab = open_table(file)
        except FileNotFoundError:
            file = file.split(".feather")[0]
            file += ".csv"
            tab = open_table(file)
    return tab


def pickler(contents, filepath):
    """Store arbitrary Python object in filepath"""
    with open(filepath, "wb") as fp:
        pickle.dump(contents, fp)


def profile_func(func, kwargs):
    prof = profile.Profile()
    prof.enable()
    func(**kwargs)
    prof.disable()
    return prof


def add_metadata_to_multiindex(df, df_meta):
    """Takes a df of count data and adds metadata as a column multiindex.
    
    The two input frames must have the same column names.
    
    Parameters
    ----------
    df : pd.DataFrame, count data with m rows, n columns
    df_meta : pd.DataFrame, metadata with k rows, n columns
    """

    if np.any(df.columns != df_meta.columns):
        raise Exception("df and df_meta must have matching columns")

    if "Sample" in df_meta.index:
        warnings.warn("df_meta has a row named 'Sample',  \
                      this will result in a duplicated multiindex")

    col_arrays = np.vstack([df_meta.values, df.columns.to_numpy()])
    multi_cols = pd.MultiIndex.from_arrays(col_arrays,
                                           names=df_meta.index.to_list()
                                                 + ["Sample"])

    return pd.DataFrame(df.values, index=df.index, columns=multi_cols)


def get_matching_treatment_col_ix(df, control_cols) -> list:
    """Given list of control col names, return indices of control cols and matching treatment cols"""
    ix = [df.columns.get_loc(c) for c in control_cols]
    ix += [c + len(df.columns) // 2 for c in ix]
    return ix

def replicate_sampler(df: pd.DataFrame, n: int, ispaired: bool):
    if ispaired:
        return paired_replicate_sampler(df, n)
    else:
        return unpaired_replicate_sampler(df, n)

def paired_replicate_sampler(df, n):
    """
    Sample a subset of replicates from a paired-design count matrix
    df : pd.DataFrame, count data with k control columns followed by k case columns, for a total of k patients
    n : Number of patients to resample, must be <= k
    """
    if len(df.columns) % 2 != 0: raise Exception("Input df must have even number of columns (paired-design experiment)")
    patients = len(df.columns) // 2
    if patients < n: raise Exception(f"Number of samples must be smaller than total number of replicates! {len(df.columns)//2}")
    ind = np.array(random.sample(range(0, patients), n))
    ind = np.concatenate([ind, ind + patients])
    ind = np.sort(ind)
    return df.iloc[:, ind], ind


def unpaired_replicate_sampler(df, n):
    """
    Sample a subset of replicates from a count matrix
    df : pd.DataFrame, count data with k control columns followed by k case columns, for a total of k patients
    n : Number of patients to resample, must be <= k
    """
    if len(df.columns) % 2 != 0: raise Exception("Input df must have even number of columns")
    patients = len(df.columns) // 2
    if patients < n: raise Exception(f"Number of samples must be smaller than total number of replicates! {len(df.columns)//2}")
    ind_ctrl = np.array(random.sample(range(0, patients), n))
    ind_case = np.array(random.sample(range(patients, 2*patients), n))
    ind = np.array(list(ind_ctrl) + list(ind_case))
    return df.iloc[:, ind], ind

def get_grid_size(n, k=0, fill=False):
    """
    Given interger n, find grid of size l*m = n that comes closest to being a square
    For k > 0,  returns successively "less square" grids
    Retruns (l, m)
    Can be used e.g. for plotting grids
    If fill, prevents pairs of form (1, m) by adding +1 to n
    """
    if n == 1: return (1, 1)
    from sympy import isprime
    if fill and isprime(n): n += 1
    pairs = []
    for i in reversed(range(1, n)):
        rows, cols = 0, 0
        if n % i == 0:
            cols = i
            rows = n // i
            pairs.append((rows, cols))

    pairs = [tuple(sorted(p)) for p in pairs]
    pairs = list(set(pairs))
    pairs = sorted(pairs, key=lambda x: np.abs(x[0] - x[1]))
    try:
        return pairs[k]
    except IndexError:
        return pairs[0]


# Code below from https://github.com/realpython/codetiming

from contextlib import ContextDecorator
from dataclasses import dataclass, field
import time
from typing import Any, Callable, ClassVar, Dict, Optional


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[Dict[str, float]] = dict()
    name: Optional[str] = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: Optional[Callable[[str], None]] = print
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.logger:
            self.logger(self.text.format(elapsed_time))
        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
