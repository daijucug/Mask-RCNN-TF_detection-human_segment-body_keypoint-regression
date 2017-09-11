#-------------------------------------------------------------------------------
# Name:        Chalearn LAP evaluation scripts
# Purpose:     Provide evaluation scripts for Chalearn LAP challenge tracks
#
# Author:      Xavier Baro
#              Miguel Angel Bautista
#
# Created:     21/01/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import re
import csv
import numpy
from PIL import Image


def gesture_overlap_csv(csvpathgt, csvpathpred, seqlenght):
    """ Evaluate this sample agains the ground truth file """
    maxGestures=20

    # Get the list of gestures from the ground truth and frame activation
    gtGestures = []
    binvec_gt = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathgt, 'rb') as csvfilegt:
        csvgt = csv.reader(csvfilegt)
        for row in csvgt:
            binvec_gt[int(row[1])-1, int(row[2])-1:int(row[3])-1] = 1
            gtGestures.append(int(row[1]))

    # Get the list of gestures from prediction and frame activation
    predGestures = []
    binvec_pred = numpy.zeros(maxGestures, seqlenght)
    with open(csvpathpred, 'rb') as csvfilepred:
        csvpred = csv.reader(csvfilepred)
        for row in csvpred:
            binvec_pred[int(row[1])-1, int(row[2])-1:int(row[3])-1] = 1
            predGestures.append(int(row[1]))

    # Get the list of gestures without repetitions for ground truth and predicton
    gtGestures = numpy.unique(gtGestures)
    predGestures = numpy.unique(predGestures)

    # Find false positives
    falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

    # Get overlaps for each gesture
    overlaps = []
    for idx in gtGestures:
        intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
        aux = binvec_gt[idx-1] + binvec_pred[idx-1]
        union = sum(aux > 0)
        overlaps.append(intersec/union)

    # Use real gestures and false positive gestures to calculate the final score
    return sum(overlaps)/(len(overlaps)+len(falsePos))

def action_overlap_csv(csvpathgt, csvpathpred, seqLength):
    """ Evaluate this sample against the ground truth file """
    maxGestures = 11
    maxActors = 2

    # Get the list of gestures from the ground truth and frame activation
    predGestures = []
    binvec_pred = numpy.zeros((maxActors, maxGestures, seqLength))
    gtGestures = []
    binvec_gt = numpy.zeros((maxActors, maxGestures, seqLength))

    with open(csvpathpred, 'rb') as csvfilepred:
        csvpred = csv.reader(csvfilepred)
        for row in csvpred:
            binvec_pred[int(row[0])-1, int(row[1])-1, int(row[2])-1:int(row[3])-1] = 1
            predGestures.append([int(row[0])-1, int(row[1])-1])

    # Get the list of gestures from prediction and frame activation
    with open(csvpathgt, 'rb') as csvfilegt:
        csvgt = csv.reader(csvfilegt)
        for row in csvgt:
            binvec_gt[int(row[0])-1, int(row[1])-1, int(row[2])-1:int(row[3])-1] = 1
            gtGestures.append([int(row[0])-1, int(row[1])-1])

    overlaps = []
    falsePos = []
    for actor in range(maxActors):
        # Get the list of gestures without repetitions for ground truth and predicton
        gtGesturesAux = [l[1] for l in gtGestures if l[0] == actor]
        predGesturesAux = [l[1] for l in predGestures if l[0] == actor]
        gtGesturesAux = numpy.unique(gtGesturesAux)
        predGesturesAux = numpy.unique(predGesturesAux)

        # Find false positives
        falsePos.extend(numpy.setdiff1d(gtGesturesAux, numpy.union1d(gtGesturesAux, predGesturesAux)))

        # Get overlaps for each gesture
        for idx in gtGesturesAux:
            intersec = sum(binvec_gt[actor, idx] * binvec_pred[actor, idx])
            aux = binvec_gt[actor, idx] + binvec_pred[actor, idx]
            union = sum(aux > 0)
            if union != 0:
                overlaps.append(intersec/union)
        # Use real gestures and false positive gestures to calculate the final score
    return sum(overlaps)/(len(overlaps)+len(falsePos))

def getLimbPanoramic(im, limb):

    """
    Get specific limb from panoramic image
    """
    startcol = (limb-1) * (480)
    endcol = startcol + 479
    return im[:, startcol:endcol]


def overlap_images(gtimage, predimage):

    """ this function computes the overlap between two binary images im1 and im2 """


    gtimage=(numpy.array(gtimage)>127)*1
    predimage=(numpy.array(predimage)>127)*1

    intersec = numpy.bitwise_and(gtimage, predimage)
    intersec_val = float(numpy.sum(intersec))

    union = numpy.bitwise_or(gtimage, predimage)

    union_val = float(numpy.sum(union))

    if union_val == 0:
        return 0
    else:
        if float(intersec_val / union_val)>0.5:
            return 1
        else:
            return 0



def exportGT_Gesture(dataPath, outputPath):
    """ Create Ground Truth folder. Open each file in the data path and copy labels and sample data to output path"""
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)

    # Check the output path
    if os.path.exists(outputPath) and os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
    	sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file = os.path.split(sampleFile)[1]
        sampleID = os.path.splitext(file)[0]
        samplePath = dataPath + os.path.sep + sampleID

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile = zipfile.ZipFile(sampleFile, "r")
            zipFile.extractall(samplePath)

        # Copy labels file
        sampleDataPath = samplePath + os.path.sep + sampleID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        shutil.copyfile(sampleDataPath, outputPath + sampleID + '_data.csv')

        # Copy Data file
        srcSampleDataPath = samplePath + os.path.sep + sampleID + '_data.csv'
        dstSampleDataPath = outputPath + os.path.sep + sampleID + '_data.csv'
        if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        shutil.copyfile(srcSampleDataPath, dstSampleDataPath)
        if not os.path.exists(dstSampleDataPath) or not os.path.isfile(dstSampleDataPath):
            raise Exception("Cannot copy data file: " + srcSampleDataPath + "->" + dstSampleDataPath)

        # Copy labels file
        srcSampleLabelsPath = samplePath + os.path.sep + sampleID + '_labels.csv'
        dstSampleLabelsPath = outputPath + os.path.sep + sampleID + '_labels.csv'
        if not os.path.exists(srcSampleLabelsPath) or not os.path.isfile(srcSampleLabelsPath):
            raise Exception("Invalid sample file. Sample labels is not available")
        shutil.copyfile(srcSampleLabelsPath, dstSampleLabelsPath)
        if not os.path.exists(dstSampleLabelsPath) or not os.path.isfile(dstSampleLabelsPath):
            raise Exception("Cannot copy labels file: " + srcSampleLabelsPath + "->" + dstSampleLabelsPath)

        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)


def exportGT_Action(dataPath,outputPath):
    """ Create Ground Truth folder. Open each file in the data path and copy labels and sample data to output path"""
    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)

    # Check the output path
    if os.path.exists(outputPath) or os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
        sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file=os.path.split(sampleFile)[1]
        sampleID=os.path.splitext(file)[0]
        samplePath=dataPath + os.path.sep + sampleID;

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile=zipfile.ZipFile(sampleFile,"r")
            zipFile.extractall(samplePath)


        # Copy Data file
        srcSampleDataPath=samplePath + os.path.sep + sampleID + '_data.csv'
        dstSampleDataPath=outputPath + os.path.sep + sampleID + '_data.csv'
        if not os.path.exists(srcSampleDataPath) or not os.path.isfile(srcSampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        shutil.copyfile(srcSampleDataPath,dstSampleDataPath)
        if not os.path.exists(dstSampleDataPath) or not os.path.isfile(dstSampleDataPath):
            raise Exception("Cannot copy data file: " + srcSampleDataPath + "->" + dstSampleDataPath)

        # Copy labels file
        srcSampleLabelsPath=samplePath + os.path.sep + sampleID + '_labels.csv'
        dstSampleLabelsPath=outputPath + os.path.sep + sampleID + '_labels.csv'
        if not os.path.exists(srcSampleLabelsPath) or not os.path.isfile(srcSampleLabelsPath):
            raise Exception("Invalid sample file. Sample labels is not available")
        shutil.copyfile(srcSampleLabelsPath,dstSampleLabelsPath)
        if not os.path.exists(dstSampleLabelsPath) or not os.path.isfile(dstSampleLabelsPath):
            raise Exception("Cannot copy labels file: " + srcSampleLabelsPath + "->" + dstSampleLabelsPath)

        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)


def exportGT_Pose(dataPath,outputPath):
    """ Create Ground Truth folder. Open each file in the data path and copy labels and sample data to output path"""

    nactors=2;
    nlimbs=14;

    # Check the given data path
    if not os.path.exists(dataPath) or not os.path.isdir(dataPath):
        raise Exception("Data path does not exist: " + dataPath)

    # Check the output path
    if os.path.exists(outputPath) and os.path.isdir(outputPath):
        raise Exception("Output path already exists. Remove it before start: " + outputPath)

    # Create the output path
    os.makedirs(outputPath)
    if not os.path.exists(outputPath) or not os.path.isdir(outputPath):
        raise Exception("Cannot create the output path: " + outputPath)

    # Get the list of samples
    samplesList = os.listdir(dataPath)

    # For each sample on the GT, search the given prediction
    for sample in samplesList:
        # Build paths for sample
        sampleFile = os.path.join(dataPath, sample)

        # Check that is a ZIP file
        if not os.path.isfile(sampleFile) or not sample.lower().endswith(".zip"):
            continue

        # Prepare sample information
        file=os.path.split(sampleFile)[1]
        sampleID=os.path.splitext(file)[0]
        samplePath=dataPath + os.path.sep + sampleID;

        # Unzip sample if it is necessary
        if os.path.isdir(samplePath):
            unziped = False
        else:
            unziped = True
            zipFile=zipfile.ZipFile(sampleFile,"r")
            zipFile.extractall(samplePath)

        # Copy labels images
        gtimages=os.listdir(samplePath+os.path.sep+'masks'+os.path.sep)

        for img in gtimages:

                srcSampleLabelsPath = samplePath + os.path.sep+'masks'+os.path.sep+ img
                dstSampleLabelsPath = outputPath + os.path.sep + img
                if not os.path.exists(srcSampleLabelsPath) or not os.path.isfile(srcSampleLabelsPath):
                    raise Exception("Invalid sequence file. Limb labels are not available")
                shutil.copyfile(srcSampleLabelsPath,dstSampleLabelsPath)
                if not os.path.exists(dstSampleLabelsPath) or not os.path.isfile(dstSampleLabelsPath):
                    raise Exception("Cannot copy limbs file: " + srcSampleLabelsPath + "->" + dstSampleLabelsPath)

        # Remove temporal data
        if unziped:
            shutil.rmtree(samplePath)


def evalPose(prediction_dir, truth_dir):
    """ Perform the overlap evaluation for a set of samples """

    # Get the list images from the gt
    gt_list = os.listdir(truth_dir)

    score = 0.0
    nevals = 0

    for gtlimbimage in gt_list:
        # Avoid double check, use only labels file
        if not gtlimbimage.lower().endswith(".png"):
            continue

        # Build paths for prediction and ground truth files
        aux = gtlimbimage.split('.')
        parts = aux[0].split('_')
        seqID = parts[0]
        gtlimbimagepath = os.path.join(truth_dir, gtlimbimage)
        predlimbimagepath= os.path.join(prediction_dir)+ os.path.sep + seqID+'_'+parts[1]+'_'+parts[2]+"_prediction.png"

        #check predfile exists
        if not os.path.exists(predlimbimagepath) or not os.path.isfile(predlimbimagepath):

            score += 0
            nevals += 1

        else:

            #Load images
            gtimage = Image.open(gtlimbimagepath)
            gtimage = gtimage.convert('L')
            gtimage = numpy.asarray(gtimage)

            predimage = Image.open(predlimbimagepath)
            predimage = predimage.convert('L')
            predimage = numpy.asarray(predimage)

            for limbid in range(1, 15):
                gtimagelimb = getLimbPanoramic(gtimage, limbid)
                predimagelimb = getLimbPanoramic(predimage, limbid)
                if numpy.count_nonzero(gtimagelimb) >= 1:
                    score += overlap_images(gtimagelimb, predimagelimb)
                    nevals += 1

    #release videos and return mean overlap
    return score/nevals


def evalAction(prediction_dir, truth_dir):
    """ Perform the overlap evaluation for a set of samples """
    worseVal=10000

    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;
    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
    	predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")

        # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                numFrames=int(row[0])
            del filereader

        # Get the score
        numSamples+=1
        score+=action_overlap_csv(labelsFile, predFile, numFrames)
    return score/numSamples


def evalGesture(prediction_dir,truth_dir):
    """ Perform the overlap evaluation for a set of samples """
    worseVal=10000

    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;
    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
    	predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")

        # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                numFrames=int(row[0])
            del filereader

        # Get the score
        numSamples+=1
        score+=gesture_overlap_csv(labelsFile, predFile, numFrames)

    return score/numSamples
