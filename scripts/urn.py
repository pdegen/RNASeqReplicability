import numpy as np
from scipy.optimize import root

try:
    from scipy.stats import nchypergeom_wallenius
except ImportError:
    pass


def E_hyp(A, B, N):
    "Expected intersection of two uniform draws of size A, B (without replacement) out of N elements"
    # assert (A <= N) & (B <= N)
    return A * B / N


def E_nc_hyp(A, N, m1, w, method="fisher"):
    """
    Approximate expected intersection of m1 with one biased draw of size A (without replacement) out of N elements
    m1 elements have weight w1,
    m2 = N-m1 elements have weigt w2,
    w = w1/w2
    """
    # assert A <= N
    if method == "fisher":
        a = w - 1
        b = m1 + A - N - (m1 + A) * w
        c = m1 * A * w
        return -2 * c / (b - np.sqrt(b ** 2 - 4 * a * c))

    elif method == "wallenius":
        def fun(x, m1, m2, w, A):
            return x / m1 + (1 - (A - x) / m2) ** w - 1

        return root(fun, x0=m1, args=(m1, N - m1, w, A)).x

    elif method == "wallenius_scipy":
        # requires scipy >= 1.7.0
        mean = nchypergeom_wallenius.stats(M=N, n=m1, N=A, odds=w, moments='m')
        return mean


def E_nc_hyp2(A, B, N, m1, w, method="fisher"):
    """
    Approximate expected intersection of two biased draws of size A, B (without replaement) out of N elements
    m1 elements have weight w1,
    m2 = N-m1 elements have weigt w2,
    w = w1/w2
    """
    # assert (A <= N) & (B <= N)
    m2 = N - m1
    Am1 = E_nc_hyp(A, N, m1, w, method=method)  # expected overlap of A with m1
    Bm1 = E_nc_hyp(B, N, m1, w, method=method)  # expected overlap of B with m1
    ABm1 = E_hyp(Am1, Bm1, m1)  # expected overlap of A and B in m1

    Am2 = A - Am1
    Bm2 = B - Bm1
    ABm2 = E_hyp(Am2, Bm2, m2)  # expected overlap of A and B in m2

    return ABm1 + ABm2


def E_jacc(A, B, N, I):
    "Approximate expected jaccard index of two sets of size A, B drawn without replacement out of N, given expected intersection I"
    return I / (A + B - I)


def E_jacc2(A, w, N, method="fisher"):
    m1 = A
    I = E_nc_hyp2(A, A, N, m1, w, method=method)
    return I / (A + A - I)


def random_jacc_rep(frac):
    """Jaccard index of two uniformly sampled subsets of fractional size frac"""
    return frac / (2 - frac)


def random_metrics(A, N, m1, w=1):
    """MCC, precision, recall of randomly sampled subsets of size A out of N, given ground truth of size m1, and ground truth bias w (odds ratio)"""
    TP = E_nc_hyp(A, N, m1, w=w, method="wallenius")
    FP = A - TP
    FN = m1 - TP
    TN = N - TP - FP - FN
    prec = TP / (TP + FP) if TP + FP else np.nan
    rec = TP / (TP + FN) if TP + FN else np.nan
    squared = float((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = (TP * TN - FP * FN) / (np.sqrt(squared)) if squared else np.nan
    return mcc, prec, rec
