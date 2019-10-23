# GlaS metrics, translated from the official Matlab code:
# https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/evaluation_v6.zip
#

import numpy as np

from scipy import stats
from sklearn.neighbors import NearestNeighbors


def ObjectHausdorff(S=None, G=None):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaS = (S > 0).sum()
    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))

    temp1 = 0
    for iLabelS in range(len(listLabelS)):
        Si = (S == listLabelS[iLabelS])
        intersectlist = G[Si]
        if intersectlist.any():
            indexGi = stats.mode(intersectlist).mode
            Gi = (G == indexGi)
        else:
            tempDist = np.zeros((len(listLabelG), 1))
            for iLabelG in range(len(listLabelG)):
                Gi = (G == listLabelG[iLabelG])
                tempDist[iLabelG] = Hausdorff(Gi, Si)
            minIdx = np.argmin(tempDist)
            Gi = (G == listLabelG[minIdx])
        omegai = Si.sum() / totalAreaS
        temp1 = temp1 + omegai * Hausdorff(Gi, Si)

    temp2 = 0
    for iLabelG in range(len(listLabelG)):
        tildeGi = (G == listLabelG[iLabelG])
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = (S == indextildeSi)
        else:
            tempDist = np.zeros((len(listLabelS), 1))
            for iLabelS in range(len(listLabelS)):
                tildeSi = (S == listLabelS[iLabelS])
                tempDist[iLabelS] = Hausdorff(tildeGi, tildeSi)
            minIdx = np.argmin(tempDist)
            tildeSi = (S == listLabelS[minIdx])
        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 = temp2 + tildeOmegai * Hausdorff(tildeGi, tildeSi)

    objHausdorff = (temp1 + temp2) / 2
    return objHausdorff

def Hausdorff(S=None, G=None, *args, **kwargs):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))

    numS = len(listS)
    numG = len(listG)
    if numS == 0 and numG == 0:
        hausdorffDistance = 0
        return hausdorffDistance
    else:
        if numS == 0 or numG == 0:
            hausdorffDistance = np.Inf
            return hausdorffDistance

    y = np.where(S > 0)
    x = np.where(G > 0)

    x = np.vstack((x[0], x[1])).transpose()
    y = np.vstack((y[0], y[1])).transpose()

    nbrs = NearestNeighbors(n_neighbors=1).fit(x)
    distances, indices = nbrs.kneighbors(y)
    dist1 = np.max(distances)

    nbrs = NearestNeighbors(n_neighbors=1).fit(y)
    distances, indices = nbrs.kneighbors(x)
    dist2 = np.max(distances)

    hausdorffDistance = np.max((dist1, dist2))
    return hausdorffDistance

def F1score(S=None, G=None):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    numS = len(listS)
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))
    numG = len(listG)
    
    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    tempMat = np.zeros((numS, 3))
    tempMat[:, 0] = listS
    for iSegmentedObj in range(numS):
        intersectGTObjs = G[S == tempMat[iSegmentedObj, 0]]
        if intersectGTObjs.any():
            intersectGTObjs_flat = np.delete(intersectGTObjs.flatten(), 
                                             np.where(intersectGTObjs.flatten() == 0))
            
            if len(intersectGTObjs_flat) == 0: maxGTi = 0
            else: maxGTi = stats.mode(intersectGTObjs_flat).mode
            tempMat[iSegmentedObj, 1] = maxGTi

    for iSegmentedObj in range(numS):
        if tempMat[iSegmentedObj, 1] != 0:
            SegObj = (S == tempMat[iSegmentedObj, 0])
            GTObj = (G == tempMat[iSegmentedObj, 1])
            overlap = np.logical_and(SegObj, GTObj)
            areaOverlap = overlap.sum()
            areaGTObj = GTObj.sum()
            if areaOverlap / areaGTObj > 0.5:
                tempMat[iSegmentedObj, 2] = 1

    TP = (tempMat[:, 2] == 1).sum()
    FP = (tempMat[:, 2] == 0).sum()
    FN = numG - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    if precision + recall == 0:
        return 0
    
    score = (2 * precision * recall) / (precision + recall)
    
    return score

def ObjectDice(S, G):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    numS = len(listLabelS)
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))
    numG = len(listLabelG)

    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    temp1 = 0
    totalAreaS = (S > 0).sum()
    for iLabelS in range(len(listLabelS)):
        Si = (S == listLabelS[iLabelS])
        intersectlist = G[Si]
        if intersectlist.any():
            indexG1 = stats.mode(intersectlist).mode
            Gi = (G == indexG1)
        else:
            Gi = np.zeros(G.shape)

        omegai = Si.sum() / totalAreaS
        temp1 += omegai * Dice(Gi, Si)

    temp2 = 0
    totalAreaG = (G > 0).sum()
    for iLabelG in range(len(listLabelG)):
        tildeGi = (G == listLabelG[iLabelG])
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = (S == indextildeSi)  # np logical and?
        else:
            tildeSi = np.zeros(S.shape)

        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 += tildeOmegai * Dice(tildeGi, tildeSi)
        
    return (temp1 + temp2) / 2

def Dice(A, B):
    intersection = np.logical_and(A, B)
    return 2. * intersection.sum() / (A.sum() + B.sum())
