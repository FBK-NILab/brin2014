"""Logarithm of the intrinsic Bayes factor of the dependence model M1
vs. the independence model M0 for a 2 way contingency table with one
margin fixed.

THIS IS JUST THE SPECIAL CASE OF A 2x2 CONTINGENCY TABLE.

See: Casella and Moreno JASA 2009, Equation 6.
"""

from __future__ import division
import numpy as np
from numpy import log, logaddexp
from scipy.special import gammaln, binom

def log_binomial(a, b):
    """Log of the binomial coefficient.
    """
    return gammaln(a + 1.0) - gammaln(b + 1.0) - gammaln(a - b + 1.0)


def compute_B_10_omf(n1, n2, y1, y2, t1, t2):
    """Naive implementation of Equation 6 in Casella and Moreno JASA
    2009.
    """
    B_10 = (n1 + n2 + 1.0) / (n1 + t1 + 1.0) / (n2 + t2 + 1.0)
    B_10 *= (t1 + 1.0) * (t2 + 1.0) / (t1 + t2 + 1.0)
    B_10 *= binom(n1 + n2, y1 + y2)
    I = np.arange(0, t1 + 1.0, 1.0)
    J = np.arange(0, t2 + 1.0, 1.0)
    B_10_tot = 0.0
    for i in I:
        for j in J:
            B_10_ij = binom(t1, i)**2 * binom(t2, j)**2
            B_10_ij /= binom(t1 + t2, i + j)
            B_10_ij /= binom(n1 + t1, y1 + i)
            B_10_ij /= binom(n2 + t2, y2 + j)
            B_10_tot += B_10_ij
    B_10 *= B_10_tot
    return B_10

def compute_log_B_10_omf(n1, n2, y1, y2, t1, t2):
    """Log of BF_10 for 2x2 contingency table with one margin fixed.

    n1, n2 are the two fixed margins.

    y1, y2 are the first column of the confusion matrix.

    t1, t2 are the parameters of the intrinsic prior class.

    This implementation is numerically stable and follows that of the
    original paper.
    """
    log_B_10 = log(n1 + n2 + 1.0) - log(n1 + t1 + 1.0) - log(n2 + t2 + 1.0)
    log_B_10 += log(t1 + 1.0) + log(t2 + 1.0) - log(t1 + t2 + 1.0)
    log_B_10 += log_binomial(n1 + n2, y1 + y2)            
    log_total = []
    I = np.arange(0, t1 + 1.0, 1.0)
    J = np.arange(0, t2 + 1.0, 1.0)
    for i in I:
        for j in J:
            log_B_10_ij = 2.0 * log_binomial(t1, i) + 2.0 * log_binomial(t2, j)
            log_B_10_ij -= log_binomial(t1 + t2, i + j)
            log_B_10_ij -= log_binomial(n1 + t1, y1 + i)
            log_B_10_ij -= log_binomial(n2 + t2, y2 + j)    
            log_total.append(log_B_10_ij)
    log_B_10 += np.logaddexp.reduce(log_total)
    return log_B_10


def compute_log_B_10_omf_vectorized(n1, n2, y1, y2, t1, t2):
    """Log of BF_10 for 2x2 contingency table with one margin fixed.

    n1, n2 are the two fixed margins.

    y1, y2 are the first column of the confusion matrix.

    t1, t2 are the parameters of the intrinsic prior class.

    This implementation is numerically stable and follows that of the
    original paper.
    """
    log_B_10 = log(n1 + n2 + 1.0) - log(n1 + t1 + 1.0) - log(n2 + t2 + 1.0)
    log_B_10 += log(t1 + 1.0) + log(t2 + 1.0) - log(t1 + t2 + 1.0)
    log_B_10 += log_binomial(n1 + n2, y1 + y2)
    I = np.arange(0, t1 + 1.0)
    J = np.arange(0, t2 + 1.0)
    i, j = np.meshgrid(I, J)
    log_B_10_ij = 2.0 * log_binomial(t1, i) + 2.0 * log_binomial(t2, j)
    log_B_10_ij -= log_binomial(t1 + t2, i + j)
    log_B_10_ij -= log_binomial(n1 + t1, y1 + i)
    log_B_10_ij -= log_binomial(n2 + t2, y2 + j)
    log_B_10 += np.logaddexp.reduce(log_B_10_ij.flatten())
    return log_B_10


def log_B_10_omf(cm, t1=None, t2=None):
    """Convenience function to computer log BF_10 given the
    contingency matrix cm.
    """
    if t1 is None: t1 = int(np.sqrt(cm.sum()))
    if t2 is None: t2 = int(np.sqrt(cm.sum()))
    n1, n2 = cm.sum(1)
    y1, y2 = cm[:,0]
    return compute_log_B_10_omf_vectorized(n1, n2, y1, y2, t1, t2)


def log_B_10_omf_min(cm, steps=10):
    lb10 = []
    values1 = np.unique(np.round(np.linspace(0, cm.sum(1)[0], steps))).astype(np.int)
    values2 = np.unique(np.round(np.linspace(0, cm.sum(1)[1], steps))).astype(np.int)
    for t1 in values1:
        for t2 in values2:
            lb10.append(log_B_10_omf(cm, t1, t2))

    return np.min(lb10)


if __name__ == '__main__':

    print __doc__
    print
    print "Testing all methods to compute log_B_10."
    print "expected result: 6.84847922794"

    n1 = 10
    n2 = 10
    y1 = 10
    y2 = 0
    t1 = 10
    t2 = 10

    print "Naive:", np.log(compute_B_10_omf(n1, n2, y1, y2, t1, t2))
    print "Logscale:", compute_log_B_10_omf(n1, n2, y1, y2, t1, t2)
    print "Vectorized Logscale:", compute_log_B_10_omf_vectorized(n1, n2, y1, y2, t1, t2)
    print
    cm = np.array([[10,  0],
                   [ 0, 10]])
    print cm
    print "From contingency matrix:", log_B_10_omf(cm, t1, t2)
