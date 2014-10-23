import numpy as np
from casella_moreno_2009_one_margin_fixed import log_B_10_omf, log_B_10_omf_min
import csv


# Metrics from Fawcett 2006
def P(X):
    p = X[0].sum()
    return p
def N(X):
    n = X[1].sum()
    return n
def TP(X):
    tp = X[0][0]
    return tp
def TN(X):
    tn = X[1][1]
    return tn
def FP(X):
    fp = X[0][1]
    return fp
def FN(X):
    fn = X[1][0]
    return fn
# sensitivity
def TPR(X):
    tpr = TP(X) / (TP(X)+FN(X))
    return tpr
# false positive rate
def FPR(X):
    fpr = FP(X) / (FP(X) + TN(X))
    return fpr
# accuracy 
def ACC(X):
    acc = (TP(X) + TN(X)) / (P(X) + N(X))
    return acc
# specificity 
def SPC(X):
    spc = TN(X) / (FP(X) + TN(X))
    return spc
# precision - positive predictive value
def PPV(X):    
    ppv = TP(X) / (TP(X) + FP(X))
    return ppv
# negative predictive value
def NPV(X):
    npv = TN(X) / (TN(X) + FN(X))
    return npv
# false discovery rate
def FDR(X):
    fdr = FP(X) / (FP(X) + TP(X))
    return fdr
# Matthews correlation coefficient
def MCC(X):
    mcc = (TP(X)*TN(X) - FP(X)*FN(X)) / np.sqrt((TP(X) + FP(X))*(TP(X) + FN(X))*(TN(X) + FP(X))*(TN(X) + FN(X)))
    return mcc
# F1 score
def F1(X):
    f1 = 2*TP(X) / (2*TP(X) + FP(X) + FN(X))
    return f1

# logBayesFactor
def B_10_CM(X):
    b_10 = log_B_10_omf(X)
    return b_10

def CCF(X):
    np
# Kappa Coeficient
def KC(X):
    rACC = ( (TN(X) + FP(X))*(TN(X)+ FN(X))+(FN(X)+TP(X))*(FP(X)+TP(X)) )/X.sum()**2
    kc = (ACC(X) - rACC) / (1 - rACC)
    return kc

def Youden_J(X):
    return TPR(X) + SPC(X) - 1.0

if __name__ == '__main__':
    Xs_sim = np.array([
    [[90., 0.],
     [10., 0.]],
    [[80., 10.],
     [0., 10.]],
    [[90., 0.],
     [0., 10.]],
    [[45., 45.],
     [5., 5.]]
    ])

    Xs_real = np.array([
    [[739., 82.],
     [441., 77.]],
    [[713., 108.],
     [408., 110.]],
    [[750., 71.],
     [441., 77.]],
    [[651., 170.],
     [340., 178.]]
    ])

    for Xs in [Xs_sim, [X/5 for X in Xs_sim], Xs_real]:
        for X in Xs:
            print(X)
            print('Accuracy: %s' % ACC(X))
            print('Sensitivity: %s' % TPR(X))
            print('Specificity: %s' % SPC(X))
            print('Precision: %s' % PPV(X))
            print('Matthews correlation coefficient %s' % MCC(X))
            print('F1 score: %s' % F1(X))
            print('kappa : %s' % KC(X))
            print("Youden's J : %s" % Youden_J(X))
            print('OMF logB_10: %s' % log_B_10_omf(X))
            print('OMF min_t(logB_10): %s' % log_B_10_omf_min(X))
            print('CM logB_10: %s' % B_10_CM(X))
            print('-----')
            print('')
