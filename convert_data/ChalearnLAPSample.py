#-------------------------------------------------------------------------------
# Name:        Chalearn LAP sample
# Purpose:     Provide easy access to Chalearn LAP challenge data samples
#
# Author:      Xavier Baro
#
# Created:     21/01/2014
# Copyright:   (c) Xavier Baro 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import zipfile
import shutil
import cv2
import numpy
import csv
from PIL import Image, ImageDraw


class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict();
        pos=0
        self.joins['HipCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Spine']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderCenter']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['Head']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ShoulderRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['ElbowRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['WristRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HandRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootLeft']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['HipRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['KneeRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['AnkleRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
        pos=pos+9
        self.joins['FootRight']=(map(float,data[pos:pos+3]),map(float,data[pos+3:pos+7]),map(int,data[pos+7:pos+9]))
    def getAllData(self):
        """ Return a dictionary with all the information for each skeleton node """
        return self.joins
    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][0]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][1]
        return skel
    def getPixelCoordinates(self):
        """ Get Pixel coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][2]
        return skel
    def toImage(self,width,height,bgColor):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.new('RGB', (width, height), bgColor)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            p=self.getPixelCoordinates()[link[1]]
            p.extend(self.getPixelCoordinates()[link[0]])
            draw.line(p, fill=(255,0,0), width=5)
        for node in self.getPixelCoordinates().keys():
            p=self.getPixelCoordinates()[node]
            r=5
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class GestureSample(object):
    """ Class that allows to access all the information for a certain gesture database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=GestureSample('Sample0001.zip')

        """
        # Check the given file
        if not os.path.exists(fileName) or not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath) :
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgb = cv2.VideoCapture(rgbVideoPath)
        while not self.rgb.isOpened():
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            cv2.waitKey(500)
            # Open video access for Depth information
        depthVideoPath=self.samplePath + os.path.sep + self.seqID + '_depth.mp4'
        if not os.path.exists(depthVideoPath):
            raise Exception("Invalid sample file. Depth data is not available")
        self.depth = cv2.VideoCapture(depthVideoPath)
        while not self.depth.isOpened():
            self.depth = cv2.VideoCapture(depthVideoPath)
            cv2.waitKey(500)
            # Open video access for User segmentation information
        userVideoPath=self.samplePath + os.path.sep + self.seqID + '_user.mp4'
        if not os.path.exists(userVideoPath):
            raise Exception("Invalid sample file. User segmentation data is not available")
        self.user = cv2.VideoCapture(userVideoPath)
        while not self.user.isOpened():
            self.user = cv2.VideoCapture(userVideoPath)
            cv2.waitKey(500)
            # Read skeleton data
        skeletonPath=self.samplePath + os.path.sep + self.seqID + '_skeleton.csv'
        if not os.path.exists(skeletonPath):
            raise Exception("Invalid sample file. Skeleton data is not available")
        self.skeletons=[]
        with open(skeletonPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.skeletons.append(Skeleton(row))
            del filereader
            # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
                self.data['fps']=int(row[1])
                self.data['maxDepth']=int(row[2])
            del filereader
            # Read labels data
        labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
        if not os.path.exists(labelsPath):
            warnings.warn("Labels are not available", Warning)
        self.labels=[]
        with open(labelsPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.labels.append(map(int,row))
            del filereader
    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()
    def clean(self):
        """ Clean temporal unziped data """
        shutil.rmtree(self.samplePath)
    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame
    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)
    def getDepth(self, frameNum):
        """ Get the depth image for the given frame """
        #get Depth frame
        depthData=self.getFrame(self.depth,frameNum)
        # Convert to grayscale
        depthGray=cv2.cvtColor(depthData,cv2.cv.CV_RGB2GRAY)
        # Convert to float point
        depth=depthGray.astype(numpy.float32)
        # Convert to depth values
        depth=depth/255.0*float(self.data['maxDepth'])
        depth=depth.round()
        depth=depth.astype(numpy.uint16)
        return depth
    def getUser(self, frameNum):
        """ Get user segmentation image for the given frame """
        #get user segmentation frame
        return self.getFrame(self.user,frameNum)
    def getSkeleton(self, frameNum):
        """ Get the skeleton information for a given frame. It returns a Skeleton object """
        #get user skeleton for a given frame
        # Check frame number
        # Get total number of frames
        numFrames = len(self.skeletons)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
        return self.skeletons[frameNum-1]
    def getSkeletonImage(self, frameNum):
        """ Create an image with the skeleton image for a given frame """
        return self.getSkeleton(frameNum).toImage(640,480,(255,255,255))

    def getComposedFrame(self, frameNum):
        """ Get a composition of all the modalities for a given frame """
        # get sample modalities
        rgb=self.getRGB(frameNum)
        depthValues=self.getDepth(frameNum)
        user=self.getUser(frameNum)
        skel=self.getSkeletonImage(frameNum)

        # Build depth image
        depth = depthValues.astype(numpy.float32)
        depth = depth*255.0/float(self.data['maxDepth'])
        depth = depth.round()
        depth = depth.astype(numpy.uint8)
        depth = cv2.applyColorMap(depth,cv2.COLORMAP_JET)

        # Build final image
        compSize1=(max(rgb.shape[0],depth.shape[0]),rgb.shape[1]+depth.shape[1])
        compSize2=(max(user.shape[0],skel.shape[0]),user.shape[1]+skel.shape[1])
        comp = numpy.zeros((compSize1[0]+ compSize2[0],max(compSize1[1],compSize2[1]),3), numpy.uint8)

        # Create composition
        comp[:rgb.shape[0],:rgb.shape[1],:]=rgb
        comp[:depth.shape[0],rgb.shape[1]:rgb.shape[1]+depth.shape[1],:]=depth
        comp[compSize1[0]:compSize1[0]+user.shape[0],:user.shape[1],:]=user
        comp[compSize1[0]:compSize1[0]+skel.shape[0],user.shape[1]:user.shape[1]+skel.shape[1],:]=skel

        return comp
    def getGestures(self):
        """ Get the list of gesture for this sample. Each row is a gesture, with the format (gestureID,startFrame,endFrame) """
        return self.labels
    def getGestureName(self,gestureID):
        """ Get the gesture label from a given gesture ID """
        names=('vattene','vieniqui','perfetto','furbo','cheduepalle','chevuoi','daccordo','seipazzo', \
               'combinato','freganiente','ok','cosatifarei','basta','prendere','noncenepiu','fame','tantotempo', \
               'buonissimo','messidaccordo','sonostufo')
        # Check the given file
        if gestureID<1 or gestureID>20:
            raise Exception("Invalid gesture ID <" + str(gestureID) + ">. Valid IDs are values between 1 and 20")
        return names[gestureID-1]


class ActionSample(object):
    """ Class that allows to access all the information for a certain action database sample """
    #define class to access actions data samples
    def __init__ (self,fileName):
        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=ActionSample('Seq01.zip')

        """
        # Check the given file
        if not os.path.exists(fileName) and not os.path.isfile(fileName):
            raise Exception("Sample path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file=os.path.split(fileName)[1]
        self.seqID=os.path.splitext(self.file)[0]
        self.samplePath=self.dataPath + os.path.sep + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath) :
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            zipFile.extractall(self.samplePath)

        # Open video access for RGB information
        rgbVideoPath=self.samplePath + os.path.sep + self.seqID + '_color.mp4'
        if not os.path.exists(rgbVideoPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgb = cv2.VideoCapture(rgbVideoPath)
        while not self.rgb.isOpened():
            self.rgb = cv2.VideoCapture(rgbVideoPath)
            cv2.waitKey(500)

        # Read sample data
        sampleDataPath=self.samplePath + os.path.sep + self.seqID + '_data.csv'
        if not os.path.exists(sampleDataPath):
            raise Exception("Invalid sample file. Sample data is not available")
        self.data=dict()
        with open(sampleDataPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                self.data['numFrames']=int(row[0])
            del filereader

        # Read labels data
        labelsPath=self.samplePath + os.path.sep + self.seqID + '_labels.csv'
        self.labels=[]
        if not os.path.exists(labelsPath):
            warnings.warn("Labels are not available", Warning)
        else:
            with open(labelsPath, 'rb') as csvfile:
                filereader = csv.reader(csvfile, delimiter=',')
                for row in filereader:
                    self.labels.append(map(int,row))
                del filereader

    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()

    def clean(self):
        """ Clean temporal unziped data """
        shutil.rmtree(self.samplePath)

    def getFrame(self,video, frameNum):
        """ Get a single frame from given video object """
        # Check frame number
        # Get total number of frames
        numFrames = video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # Check the given file
        if frameNum<1 or frameNum>numFrames:
            raise Exception("Invalid frame number <" + str(frameNum) + ">. Valid frames are values between 1 and " + str(int(numFrames)))
            # Set the frame index
        video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,frameNum-1)
        ret,frame=video.read()
        if ret==False:
            raise Exception("Cannot read the frame")
        return frame

    def getNumFrames(self):
        return self.data['numFrames']

    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        return self.getFrame(self.rgb,frameNum)

    def getActions(self):
        """ Get the list of gesture for this sample. Each row is an action, with the format (actionID,startFrame,endFrame) """
        return self.labels

    def getActionsName(self,actionID):
        """ Get the action label from a given action ID """
        names=('wave','point','clap','crouch','jump','walk','run','shake hands', \
               'hug','kiss','fight')
        # Check the given file
        if actionID<1 or actionID>11:
            raise Exception("Invalid action ID <" + str(actionID) + ">. Valid IDs are values between 1 and 11")
        return names[actionID-1]

    def exportPredictions(self, prediction, predPath):
        """ Export the given prediction to the correct file in the given predictions path """
        if not os.path.exists(predPath):
            os.makedirs(predPath)
        output_filename = os.path.join(predPath,  self.seqID + '_prediction.csv')
        output_file = open(output_filename, 'wb')
        for row in prediction:
            output_file.write(repr(int(row[0])) + "," + repr(int(row[1])) + "," + repr(int(row[2])) + "," + repr(int(row[3])) + "\n")
        output_file.close()

    def evaluate(self, csvpathpred):
        """ Evaluate this sample against the ground truth file """
        maxGestures = 11
        maxActors = 2
        seqLength = self.getNumFrames()

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
        for row in self.getActions():
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



class PoseSample(object):
    """ Class that allows to access all the information for a certain pose database sample """
    #define class to access gesture data samples
    def __init__ (self,fileName):

        """ Constructor. Read the sample file and unzip it if it is necessary. All the data is loaded.

            sample=PoseSample('Seq01.zip')

        """
        # Check the given file
        if not os.path.exists(fileName) and not os.path.isfile(fileName):
            raise Exception("Sequence path does not exist: " + fileName)

        # Prepare sample information
        self.fullFile = fileName
        self.dataPath = os.path.split(fileName)[0]
        self.file = os.path.split(fileName)[1]
        self.seqID = os.path.splitext(self.file)[0]
        self.samplePath = self.dataPath + self.seqID;

        # Unzip sample if it is necessary
        if os.path.isdir(self.samplePath):
            self.unzip = False
        else:
            self.unzip = True
            zipFile=zipfile.ZipFile(self.fullFile,"r")
            print self.samplePath
            zipFile.extractall(self.samplePath)

        # Set path for rgb images
        #rgbPath=self.samplePath + os.path.sep + 'imagesjpg'+ os.path.sep
	rgbPath=self.samplePath +os.path.sep+ self.samplePath+ os.path.sep+ 'imagesjpg'+ os.path.sep
	print rgbPath
        if not os.path.exists(rgbPath):
            raise Exception("Invalid sample file. RGB data is not available")
        self.rgbpath = rgbPath


        # Set path for gt images
        gtPath=self.samplePath + os.path.sep + self.samplePath + os.path.sep + 'masks' + os.path.sep
        print gtPath
        if not os.path.exists(gtPath):
            self.gtpath= "empty"
        else:
            self.gtpath = gtPath


        frames=os.listdir(self.rgbpath)
        self.numberFrames = len(frames)


    def __del__(self):
        """ Destructor. If the object unziped the sample, it remove the temporal data """
        if self.unzip:
            self.clean()

    def clean(self):
        """ Clean temporal unziped data """
        shutil.rmtree(self.samplePath)



    def getRGB(self, frameNum):
        """ Get the RGB color image for the given frame """
        #get RGB frame
        if frameNum>self.numberFrames:
            raise Exception("Number of frame has to be less than: "+ self.numberFrames)
        framepath=self.rgbpath+self.seqID[3:5]+'_'+ '%04d' %frameNum+'.jpg'
        if not os.path.isfile(framepath):
            raise Exception("RGB file does not exist: " + framepath)
        return cv2.imread(framepath)

    def getNumFrames(self):
        return self.numberFrames

    def getLimb(self, frameNum, actorID,limbID):
        """ Get the BW limb image for a certain frame and a certain limbID """

        if self.gtpath == "empty":

            raise Exception("Limb labels are not available for this sequence. This sequence belong to the validation set.")
        else:

            limbpath= self.gtpath + self.seqID[-2:]+'_'+ str(frameNum).zfill(4) +'_'+str(actorID)+'.png'
            if frameNum>self.numberFrames:
                raise Exception("Number of frame has to be less than: "+ self.numberFrames)

            if actorID<1 or actorID>2:
                raise Exception("Invalid actor ID <" + str(actorID) + ">. Valid frames are values between 1 and 2 ")

            if limbID<1 or limbID>14:
                raise Exception("Invalid limb ID <" + str(limbID) + ">. Valid frames are values between 1 and 14")

            im = cv2.imread(limbpath)
            return self.getLimbPanoramic(im, limbID)

    def getLimbsName(self,limbID):
        """ Get the limb label from a given limb ID """
        names=('head','torso','lhand','rhand','lforearm','rforearm','larm','rarm', \
               'lfoot','rfoot','lleg','rleg','lthigh','rthigh')
        # Check the given file
        if limbID<1 or limbID>14:
            raise Exception("Invalid limb ID <" + str(limbID) + ">. Valid IDs are values between 1 and 14")
        return names[limbID-1]

    def overlap_images(self, gtimage, predimage):

        """
        This function computes the hit measure of overlap between two binary images im1 and im2.

        """

        [ret, im1] = cv2.threshold(gtimage,  127, 255, cv2.THRESH_BINARY)
        [ret, im2] = cv2.threshold(predimage, 127, 255, cv2.THRESH_BINARY)
        intersec = cv2.bitwise_and(im1, im2)
        intersec_val = float(numpy.sum(intersec))
        union = cv2.bitwise_or(im1, im2)
        union_val = float(numpy.sum(union))

        if union_val == 0:
            return 0
        else:
            if float(intersec_val / union_val) > 0.5:
                return 1
            else:
                return 0

    def exportPredictions(self, prediction,frame,actor,limb,predPath):
        """
        Export the given prediction to the correct file in the given predictions path
        prediction : an indicator ndarray where pixels 1 denote the region where the limb is predicted
        """

        if not os.path.exists(predPath):
            os.makedirs(predPath)

        prediction_filename = predPath+os.path.sep+ self.seqID[3:5] +'_'+ '%04d' %frame +'_'+str(actor)+'_'+str(limb)+'_prediction.png'
        cv2.imwrite(prediction_filename,prediction)

    def convertPredictions(self,predPath, outputPath):

        """
            Convert limb predictions to panoramic images by limb-wise concatenation
        """


        if not os.path.exists(outputPath):
            print "Directory", outputPath, "does not exist, creating ..."
            os.mkdir(outputPath)
        #loop for all images in folder predPath to get the number of frames

        for frame in range(1, self.numberFrames+1):
            print "Processing frame", frame, "of", self.numberFrames
            for actor in range(1, 3):
                mosaicImage = None
                for limb in range(1, 15):
                    # load limb image
                    limbImageName = self.seqID[-2:]+"_"+str(frame).zfill(4)+"_"+str(actor)+"_"+str(limb)+".png"
                    pathImage = predPath + os.sep + limbImageName
                    print pathImage
                    if not os.path.isfile(pathImage):
                        raise Exception("Limb "+ str(limb) + " for frame " + str(frame).zfill(4) + " and actor " + str(actor) + " does not exist")
                    loadlimb = cv2.imread(pathImage, cv2.IMREAD_UNCHANGED)
                    if mosaicImage is None:
                        mosaicImage = loadlimb
                    else:
                        mosaicImage = numpy.append(mosaicImage, loadlimb, axis = 1)

                mosaicImage = Image.fromarray(mosaicImage)
                outputImageName = self.seqID[-2:] + '_' + str(frame).zfill(4) + '_' + str(actor) + "_prediction.png"
                mosaicImage.save(os.path.join(outputPath, outputImageName))
        return True
    def getLimbPanoramic(self, im, limb):

        """
        Get specific limb from panoramic image
        """
        startcol = (limb-1) * (480)
        endcol = startcol + 479
        return im[:, startcol:endcol]

    def evaluate(self, predpath):
        """ Evaluate this sample agains the ground truth file """
            # Get the list of videos from ground truth
        gt_list = os.listdir(self.gtpath)

        # For each sample on the GT, search the given prediction
        score = 0.0
        nevals = 0

        for idx, gtlimbimage in enumerate(gt_list):
            print "Processing image",idx,"of",len(gt_list),"images..."
            # Avoid double check, use only labels file
            if not gtlimbimage.lower().endswith(".png"):
                continue

            # Build paths for prediction and ground truth files
            aux = gtlimbimage.split('.')
            parts = aux[0].split('_')
            seqID = parts[0]
            gtlimbimagepath = os.path.join(self.gtpath, gtlimbimage)
            predlimbimagepath= os.path.join(predpath) + os.path.sep + seqID+'_'+parts[1]+'_'+parts[2]+"_prediction.png"

            #check predfile exists
            if not os.path.exists(predlimbimagepath) or not os.path.isfile(predlimbimagepath):
                raise Exception("Prediction file for frame "+parts[1]+" and actor "+parts[2]+" does not exist. Prediction files should end with '_prediction.png'.")

            #Load images
            gtimage = cv2.imread(gtlimbimagepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            predimage = cv2.imread(predlimbimagepath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

            for limbid in range(1, 15):
                gtimagelimb = self.getLimbPanoramic(gtimage, limbid)
                predimagelimb = self.getLimbPanoramic(predimage, limbid)
                if numpy.count_nonzero(gtimagelimb) >= 1:
                    score += self.overlap_images(gtimagelimb, predimagelimb)
                    nevals += 1
         #release videos and return mean overlap
        return score/nevals

