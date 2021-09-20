#!/home/dlee/.virtualenvs/fh/bin/python
# -*- coding: utf-8 -*-
'''
    This script does attribute variability of seasonal and monthly streamflow 
    to global and local predictors and predict streamflow.
    
    Current prediction types:
        MP1: Monthly prediction (t-1)
        MP2: Monthly prediction (t-2)
        MP3: Monthly prediction (t-3)
        MP4: Monthly prediction (t-4)
        MP5: Monthly prediction (t-5)
        MP6: Monthly prediction (t-6)
        MP7: Monthly prediction (t-7)

    Donghoon Lee (donghoon.lee@wisc.edu)
    Date modified: 05/28/2020
'''

import os, sys, getopt
import numpy as np
import pandas as pd
from sspred import SSPRED
import time

def attribute(args):
    
#     # REMOVE: Argument demonstration
#     args = '--flow=/Users/dlee/data/repo/S3Forecast/data/chtc_in/dfFlowDams0.hdf \
#             --glob=/Users/dlee/data/repo/S3Forecast/data/chtc_in/dfPredGlob.hdf \
#             --locl=/Users/dlee/data/repo/S3Forecast/data/chtc_in/dfPredDamsLocl0.hdf'.split()

    # Arguments and Parameters control
    isnote = False
    try: 
        opts, args = getopt.getopt(args, '', ['flow=', 'glob=', 'locl=', 'note='])
    except getopt.GetoptError:
        print('{} --flow=<filename> --mon=<filename> --pred=<filename>'.format(sys.argv[0]))
        sys.exit(2)
    for o, a in opts:
        if o == '--flow':
            filnFlow = a
        elif o == '--glob':
            filnGlob = a
        elif o == '--locl':
            filnLocl = a
        elif o == '--note':
            isnote = True
            note = a
        else:
            assert False, "unhandled option"

    # Load required data (Please see details in "attribute_load.py")
    # Flow Dataframe
    dfFlow = pd.read_hdf(filnFlow)
    pointList = dfFlow.columns
    print('"%s" is imported.' % filnFlow)
    # Predictor Dataframe
    dfGlob = pd.read_hdf(filnGlob)
    print('"%s" is imported.' % filnGlob)
    dfLocl = pd.read_hdf(filnLocl)
    print('"%s" is imported.' % filnLocl)

    # Set each matching item into a tuple
    lm = 9   # Maximum Lead-time
    mp1 = []    # M1: Monthly Prediction   (Glob: lm-2, Locl: 1)
    mp2 = []    # M2: Monthly Prediction   (Glob: lm-3, Locl: 2)
    mp3 = []    # M3: Monthly Prediction   (Glob: lm-4, Locl: 3)
    mp4 = []    # M4: Monthly Prediction   (Glob: lm-5, Locl: 4)
    mp5 = []    # M5: Monthly Prediction   (Glob: lm-6, Locl: 5)
    mp6 = []    # M6: Monthly Prediction   (Glob: lm-7, Locl: 6)
    mp7 = []    # M7: Monthly Prediction   (Glob: lm-8, Locl: 7)
    stime = time.time()
    for i in range(pointList.shape[0]):
        point_no = pointList[i]
        dfFlow_point = dfFlow[point_no]
        # Current predictors: ['amo', 'nao', 'oni', 'pdo', 'flow', 'swvl', 'snow']
        dfPred_point = pd.concat([dfGlob, dfLocl[point_no]], axis=1)
        # M1 Prediction
        print('%d - MP1 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,1,1,lm], [2,2,2,2,1,1,1]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp1.append(outbox)
        # M2 Prediction
        print('%d - MP2 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,2,2,lm], [3,3,3,3,2,2,2]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp2.append(outbox)
        # M3 Prediction
        print('%d - MP3 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,3,3,lm], [4,4,4,4,3,3,3]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp3.append(outbox)
        # M4 Prediction
        print('%d - MP4 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,4,4,lm], [5,5,5,5,4,4,4]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp4.append(outbox)
        # M5 Prediction
        print('%d - MP5 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,5,5,lm], [6,6,6,6,5,5,5]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp5.append(outbox)
        # M6 Prediction
        print('%d - MP6 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,6,6,lm], [7,7,7,7,6,6,6]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp6.append(outbox)
        # M7 Prediction
        print('%d - MP7 is started.' % point_no)
        leadMat = np.array([[lm,lm,lm,lm,7,7,lm], [8,8,8,8,7,7,7]])     # [[Max][Min]]
        outbox = SSPRED(dfFlow_point, dfPred_point, leadMat, point_no, targMonth=13).outbox
        mp7.append(outbox)


    # Printing total results
    etime = time.time() - stime
    print('%d points took %.2fs' % (len(pointList),etime))

    # Save prediction results
    outfiln = os.path.splitext(filnFlow)[0].split('/')[-1]
    if isnote:
        outfiln = outfiln + '_' + note
    np.savez_compressed(outfiln, mp1=mp1, mp2=mp2, mp3=mp3, mp4=mp4, mp5=mp5, mp6=mp6, mp7=mp7)
    print('{}.npz is saved.'.format(outfiln))


if __name__ == "__main__":
    attribute(sys.argv[1:])




