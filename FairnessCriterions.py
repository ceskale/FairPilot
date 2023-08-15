import numpy as np
from sklearn.metrics import confusion_matrix

def fairness(X1, y1, X2, y2, model, isNN) :
    
    if isNN == True :
        pred1_NN = model.predict(X1, verbose=0) 
        pred1 = np.argmax(pred1_NN, axis=1)
        pred2_NN = model.predict(X2, verbose=0) 
        pred2 = np.argmax(pred2_NN, axis=1)
    else :
        pred1 = model.predict(X1)
        pred2 = model.predict(X2)

    m1 = confusion_matrix(y1, pred1)
    TN1, FP1, FN1, TP1 = m1.ravel()
    tot1 = TN1 + FP1 + FN1 + TP1
    acc1 = (TN1+TP1)/tot1
    ppv1 = TP1/(TP1+FP1)
    FPR1 = FP1/(TP1+FP1)
    NPV1 = TN1/(TN1+FN1)
    FOR1 = FN1/(TN1+FN1)
    TPR1 = TP1/(TP1+FN1)
    FNR1 = FN1/(TP1+FN1)
    TNR1 = TN1/(TN1+FP1)
    FPR1 = FP1/(TN1+FP1)
    stp1 = (TP1+FP1)/tot1
    GRAD1 = (FP1 + TP1)/tot1

    m2 = confusion_matrix(y2, pred2)
    TN2, FP2, FN2, TP2 = m2.ravel()
    tot2 = TN2 + FP2 + FN2 + TP2
    acc2 = (TN2+TP2)/tot2
    ppv2 = TP2/(TP2+FP2)
    FPR2 = FP2/(TP2+FP2)
    NPV2 = TN2/(TN2+FN2)
    FOR2 = FN2/(TN2+FN2)
    TPR2 = TP2/(TP2+FN2)
    FNR2 = FN2/(TP2+FN2)
    TNR2 = TN2/(TN2+FP2)
    FPR2 = FP2/(TN2+FP2)
    stp2 = (TP2+FP2)/tot2
    GRAD2 = (FP2 + TP2)/tot2
    
    prp = 1 - abs(ppv1 - ppv2)
    prE = 1 - abs(FPR1 - FPR2)
    Eop = 1 - abs(FNR1 - FNR2)
    Acc = 1 - abs(acc1 - acc2)
    Eq1 = 1 - 0.5*(abs(TPR1 - TPR2) + abs(FPR1 - FPR2))
    Pty = 1 - abs(GRAD1 - GRAD2)
    
    return prp, prE, Eop, Acc, Eq1, Pty