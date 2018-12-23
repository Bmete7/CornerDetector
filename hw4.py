# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:20:20 2018

@author: BurakBey
"""


import cv2
 
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import math
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt




class ExampleContent(QWidget):
    def __init__(self, parent,fileName1,fileName2=''):
        self.parent = parent
        
        self.labInput= QLabel()
        self.labResult= QLabel()
        
        self.qpInput = None
        self.qpResult = None
        
        QWidget.__init__(self, parent)
        self.initUI(fileName1,fileName2)
        
        
    def initUI(self,fileName1,fileName2=''):        

        groupBox1 = QGroupBox('Input Image')
        self.vBox1 = QVBoxLayout()        
        groupBox1.setLayout(self.vBox1)
        
        groupBox2 = QGroupBox('Result Image')
        self.vBox2 = QVBoxLayout()
        groupBox2.setLayout(self.vBox2)        
        
        hBox = QHBoxLayout()
        hBox.addWidget(groupBox1)
        hBox.addWidget(groupBox2)

        self.setLayout(hBox)
        self.setGeometry(0, 0, 0,0)
        
        self.InputImage(fileName1)
        if fileName2 != '':
            self.ResultImage(fileName2)    
                
    def InputImage(self,fN):
        
        self.qpInput = QPixmap(fN)
        self.labInput.setPixmap(self.qpInput)
        self.vBox1.addWidget(self.labInput)

    def ResultImage(self,fN):
        self.qpResult = QPixmap(fN)
        self.labResult.setPixmap(self.qpResult)
        self.vBox2.addWidget(self.labResult)
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = "CornerDetector-Segmentor"
        self.top = 1000
        self.left = 200
        self.width = 500
        self.height = 500

        self.inputImage = None
        self.inputFile = ''
        self.result = None 
        
        self.initWindow()
        
    def initWindow(self):
         
        exitAct = QAction(QIcon('exit.png'), '&Exit' , self)
        importAct = QAction('&Open Input' , self)        
        cornerAction = QAction('&Harris corner Detector' , self)
        segmentationAction = QAction('&Segmentate' , self)
        
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        importAct.setStatusTip('Open Input')
                
        exitAct.triggered.connect(self.closeApp)
        importAct.triggered.connect(self.importInput)
        cornerAction.triggered.connect(self.cornerAction)
        segmentationAction.triggered.connect(self.segmentationAction)
        
        self.statusBar()
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        fileMenu.addAction(importAct)
        
        self.content = ExampleContent(self, '', '')
        self.setCentralWidget(self.content)
        
        self.cornerToolbar = self.addToolBar('Harris Corner Detection')
        self.cornerToolbar2 = self.addToolBar('Segmentate')
        self.cornerToolbar.addAction(cornerAction)
        self.cornerToolbar2.addAction(segmentationAction)
        
        self.setWindowTitle(self.title)
        self.setStyleSheet('QMainWindow{background-color: darkgray;border: 1px solid black;}')
        self.setGeometry( self.top, self.left, self.width, self.height)
        self.show()
    
    def closeApp(self):
        sys.exit()
        
    def importInput(self):
        fileName = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Png Files (*.png)")
        if( fileName == ''):
            return
        self.inputFile = fileName[0]
        
        self.inputImage = cv2.imread(fileName[0], cv2.IMREAD_GRAYSCALE)
        h,w = self.inputImage[:,:].shape
        
#        self.inputImage = cv2.cvtColor(self.inputImage,cv2.COLOR_BGR2RGB)
        self.corners= np.zeros((h,w,3),dtype=np.uint8)
        
        self.corners[:,:,0]=self.inputImage
        self.corners[:,:,1]=self.inputImage
        self.corners[:,:,2]=self.inputImage
        cv2.imwrite('processed.png', self.corners)
        self.content = ExampleContent(self, 'processed.png')
        self.setCentralWidget(self.content)

    
    def cornerAction(self):
        if(self.inputFile==''):
            return
        self.GaussFilter()
        self.calculateGradients()
    
    
    def otsuThresholding(self):
        # Trying to maximize between class variance
        # By iterating over all possible threshold values
        h,w = self.inputImage.shape
        histog = np.zeros((256,1), dtype ='int32')
        for x in range(h):
            for y in range(w):
                histog[self.inputImage[x,y]]+= 1
        PdfHist = histog / (h*w)
        CdfHist = np.zeros((256,1), dtype ='float64')
        prevCdf = 0
        for i in range(256):
            CdfHist[i] = prevCdf + PdfHist[i]
            prevCdf = CdfHist[i]
        
        #plt.plot(PdfHist)
        #plt.show()
        
        otsuValue = 0
        otsuMax = 0
        
        for i in range(1,255):
            pdfMeans = np.zeros((2), dtype = 'float64')
            #pdfVariances = np.zeros((2), dtype = 'float64')
            
            for k in range(0,i):
                pdfMeans[0] += PdfHist[k]
            if(i!= 0):
                pdfMeans[0] /= i
            
            for k in range(i,256):
                pdfMeans[1]+=PdfHist[k]
            if((256-i) != 0):
                pdfMeans[1]/= (256-i)
            betweenClassVariance =  ((pdfMeans[0] - pdfMeans[1]) ** 2)* (CdfHist[i]) * (1 - CdfHist[i]) 
            if (betweenClassVariance >= otsuMax):
                otsuMax = betweenClassVariance
                otsuValue = i
        
        return otsuValue
            #for k in range(0,i):
            #    #pdfVariances[0] += ((PdfHist[k] - pdfMeans[0])**2)
            #if(i!= 0):
            #    pdfVariances[0] /= i
                
            #for k in range(i,256):
            #    pdfVariances[1] = ((PdfHist[k] - pdfMeans[1])**2)
            #if((256-i) != 0):
            #    pdfVariances[1]/= (256-i)
            
            #withinT = (CdfHist[i] * pdfVariances[0]) + ( (1 - CdfHist[i] ) * pdfVariances[1])
            #if(withinT < otsuMin):
            #    otsuMin = withinT
            #    otsuValue = i
        
        
    def thresholdImage(self,T):
        h,w = self.inputImage.shape
        thresImage = np.zeros((h,w) , dtype='uint8')
        for x in range(h):
            for y in range(w):
                if ( self.inputImage[x,y] <= T + 35 ):
                    thresImage[x,y] = 0
                else:
                    thresImage[x,y] = 255
        
        return thresImage
            
            
    def morphOperate(self,thresImage,iter):
        kernel = np.ones((5,5))
        eroded = cv2.erode(thresImage,kernel,iterations = (iter+2) )
        dilated = cv2.dilate(eroded,kernel,iterations = iter )
        return dilated
    
    def maskImage(self,morphedImage):
        h,w = self.inputImage.shape
        maskedImage = self.inputImage.copy()
        changedPoints = np.zeros((h,w),dtype='uint8')
        for x in range(h):
            for y in range(w):
                if(morphedImage[x,y]==0):
                    maskedImage[x,y]=0
                    changedPoints[x,y]=1
        return maskedImage,changedPoints
    
    def kMeans(self,maskedImage,changedPoints):
        h,w = self.inputImage.shape
        # selecting random initial points
        randx = np.random.randint(0,h)
        randy = np.random.randint(0,w)
        while(changedPoints[randx,randy] == 1):
            randx = np.random.randint(0,h)
            randy = np.random.randint(0,w)
        
        randx_2 = np.random.randint(0,h)
        randy_2 = np.random.randint(0,w)
        
        while(changedPoints[randx_2,randy_2] == 1 or (randx_2==randx and randy==randy_2)):
            randx_2 = np.random.randint(0,h)
            randy_2 = np.random.randint(0,w)
        
        print(str(randx) + ' ' + str(randy) + ' '+ str(maskedImage[randx,randy]))
        print(str(randx_2) + ' ' + str(randy_2) +' '+   str(maskedImage[randx_2,randy_2]))
        
        classMeans = np.zeros((2,1), dtype='float64') # the mean of each class
        classElements = np.zeros((2,1), dtype='int32') # how many elements belongs to that class
        classMeans[0] = maskedImage[randx,randy]
        classElements[0]+=1
    
        classMeans[1] = maskedImage[randx_2,randy_2]
        classElements[1]+=1
        
        convergence = 20
        clusters = np.zeros((h,w), dtype = 'uint8')
      
        finished = np.sum(changedPoints)
        
        leastMasked_h = h-1
        leastMasked_w = w-1
        maxMasked_h = 0
        maxMasked_w = 0
        print(leastMasked_h)
        print(leastMasked_w)
        print(maxMasked_h)
        print(maxMasked_w)
        for x in range(h):
            for y in range(w):
                if(changedPoints[x,y] == 0):
                    if(x < leastMasked_h):
                        leastMasked_h = x
                    if(x > maxMasked_h):
                        maxMasked_h = x
                    if(y < leastMasked_w):
                        leastMasked_w = y
                    if(y > maxMasked_w):
                        maxMasked_w = y       
        print(leastMasked_h)
        print(leastMasked_w)
        print(maxMasked_h)
        print(maxMasked_w)
        x=leastMasked_h
        y=leastMasked_w
        while((classElements[0] + classElements[1]) < (finished) or convergence> 0.01):
            if(x==maxMasked_h):
                x=leastMasked_h
                y+=1
            if(y==maxMasked_w):
                y=leastMasked_w
            if(changedPoints[x,y] == 0 ):
                retVal = self.calculateEuclidian(maskedImage[x,y], classMeans[0], classMeans[1])
                
                oldMean = classMeans[retVal]
                classMeans[retVal] = ((classMeans[retVal]* classElements[retVal])+maskedImage[x,y]) / (classElements[retVal] + 1) 
                convergence = classMeans[retVal] - oldMean
                classElements[retVal] += 1
                clusters[x,y] = 255 * retVal
            
#                print('Class Mean 1 ' + str(classMeans[0]))
#                print('Class Mean 2 ' + str(classMeans[1]))
#                print(classElements)
#                print('Class Elements 1 ' + str(classElements[0]))
#                print('Class Elements 2 ' + str(classElements[1]))
            x += 1 
        
        tumorCount = 0
        brainCount = 0
        for x in range(h):
            for y in range(w):
                if(changedPoints[x,y] == 0):
                    if(clusters[x,y] == 0 ):
                        tumorCount += 1
                    else:
                        brainCount += 1
        if(tumorCount > brainCount):
            return clusters
        
        if(brainCount > tumorCount):
            for x in range(h):
                for y in range(w):
                    if(changedPoints[x,y] == 0):
                        if(clusters[x,y] == 0 ):
                            clusters[x,y] = 255
                        else:
                            clusters[x,y] = 0
                        
                
        return clusters

    def calculateEuclidian(self,v1,v2,v3):
        d1 = np.abs(v1-v2)
        d2 = np.abs(v1-v3)
        if(d1 < d2):
            return 0
        else:
            return 1
    def edgeFinder(self,cluster):
        h,w = cluster.shape
        edges = cluster.copy()
        edgeImage = np.zeros((h,w,3), dtype='uint8')
        edgeImage[:,:,0] = self.inputImage
        edgeImage[:,:,1] = self.inputImage
        edgeImage[:,:,2] = self.inputImage
        for x in range(1,h-1):
            for y in range(1,w-1):
                if(cluster[x,y] == 255):
                    if(cluster[x+1,y] == 255 and cluster[x,y+1] == 255 and cluster[x-1,y] == 255 and cluster[x,y-1] == 255):
                        edges[x,y] = 0
                    else:
                        edges[x,y] = 255
                        edgeImage[x,y,0] = 20
                        edgeImage[x,y,1] = 20
                        edgeImage[x,y,2] = 222
        return edges,edgeImage

    def segmentationAction(self):
        if(self.inputFile==''):
            return
        
        thresIndex = self.otsuThresholding()
        thresImage = self.thresholdImage(thresIndex)
        cv2.imwrite('threshold.jpg', thresImage)
        morphedImage = self.morphOperate(thresImage,4)
        maskedImage,changedPoints = self.maskImage(morphedImage)
        kMean = self.kMeans(maskedImage,changedPoints)
        
        
        cv2.imshow('2-Morphed.', morphedImage)
        cv2.imshow('3-Masked.', maskedImage)
        cv2.imshow('4- Cluster', kMean)
        kernel = np.ones((3,3))
        eroded = cv2.erode(kMean,kernel,iterations = 2)
        clusters_dilated = cv2.dilate(eroded,kernel,iterations = 2)
        cv2.imshow('5. Dilated Cluster' , clusters_dilated)
        edges, edgeIm = self.edgeFinder(clusters_dilated)
        cv2.imshow('7. Edge Boundaries' , edges)
        cv2.imshow('6. Segmentation Image' , edgeIm)
        
        cv2.imwrite('segmented.png', edgeIm)
        self.content = ExampleContent(self, 'processed.png','segmented.png')
        self.setCentralWidget(self.content)
        
        
    def GaussFilter(self):
        
        kernel = np.ones( (3,3), dtype='float64')
        size = 3 
        mean = int(size/2)
        sigma = 1
        sumAll = 0
        for i in range(size):
            for j in range(size):
                kernel[i,j] = math.exp(-1* ((math.pow( (i-mean)/sigma, 2.0) + (math.pow((j-mean)/sigma, 2.0)) ) / (2* math.pow(sigma,2)) )) / (sigma * math.pow(2*math.pi, 1/2))
                sumAll += kernel[i,j]
        
        for i in range(size):
            for j in range(size):
                kernel[i,j] /= sumAll
        
        self.inputImage = self.convolution(kernel)
        cv2.imwrite('filtered.jpg', self.inputImage)
        
        self.content = ExampleContent(self, self.inputFile,'filtered.jpg')
        self.setCentralWidget(self.content)
        
        
    def convolution(self,dest):
        res = self.inputImage
        [h,w] = self.inputImage.shape
        [kh,kw] = dest.shape # kernel shape
        kr = int(kh/2) # kernel radius
        res = np.zeros(self.inputImage.shape)
    
        for i in range(0+kr,h-kr):
            for j in range(0+kr,w-kr):
                for k in range(-1 * kr, kr + 1):
                    for m in range(-1 * kr, kr + 1):
                        res[i,j] += dest[k,m]*self.inputImage[i+k, j+m]
        res[:,0] = res[:, 1]
        res[:,w-1] = res[:, w-2]
        res[0,:] = res[1,:]
        res[h-1,:] = res[h-2,:]
        return res


    def calculateGradients(self):
        [h,w]= self.inputImage.shape
        grad_x = np.zeros((h,w), dtype = 'float64')
        grad_y = np.zeros((h,w), dtype = 'float64')
        print('Gradients are being calculated..')
        for i in range(h):
            for j in range(w):
                if(j >= w-1):
                    grad_y[i,j] = (self.inputImage[i,j] - self.inputImage[i,j-1]) / 2
                elif (j== 0):
                    grad_y[i,j] = (self.inputImage[i,j+1] - self.inputImage[i,j]) / 2
                else:
                    grad_y[i,j] = (self.inputImage[i,j+1] - self.inputImage[i,j-1]) / 2
                if(i >= h-1):
                    grad_x[i,j] = (self.inputImage[i,j] - self.inputImage[i-1,j]) / 2
                elif (i== 0):
                    grad_x[i,j] = (self.inputImage[i+1,j] - self.inputImage[i,j]) / 2
                else:
                    grad_x[i,j] = (self.inputImage[i+1,j] - self.inputImage[i-1,j]) / 2
        try:
            fP = open(('xgrad.txt'), 'w')
            maxGradX = 0
            minGradX = 0
            maxTuple = [0,0]
            minTuple = [0,0]
            for i in range(h):
                for j in range(w):
                    fP.write(str(i) + ' '  +  str(j) + ' ' +  str(grad_x[i,j]) + '\n')
                    if(grad_x[i,j] > maxGradX):
                        maxGradX = grad_x[i,j]
                        maxTuple = [i,j]
                    if(grad_x[i,j] < minGradX):
                        minGradX = grad_x[i,j]
                        minTuple = [i,j]
            fP.close()
            print( str(maxGradX) + ' ' + str(maxTuple))
            print( str(minGradX) + ' ' + str(minTuple))
            self.inputImage[maxTuple[0],:] = 255
            self.inputImage[:,maxTuple[1]] = 255
            
            self.inputImage[minTuple[0],:] = 2
            self.inputImage[:,minTuple[1]] = 2
            
            
            cv2.imwrite('pointed.jpg', self.inputImage)
            
            self.content = ExampleContent(self, self.inputFile,'pointed.jpg')
            self.setCentralWidget(self.content)
        except:
            print('cannot open file')
        structTensor = self.createStructureTensor(grad_x,grad_y)
        R = self.calculateTrace(structTensor)
        res = self.calculateThres(R)
        
    def calculateThres(self,R):
        h,w = self.inputImage.shape
        
        t = 400000
        for x in range(h):
            for y in range(w):
                if R[x,y] > t:
                    self.corners[x,y] = [0,254,0]
                    print(str(x) + ' ' + str(y) + ' ' + str (R[x,y]))
       
        cv2.imwrite('resultA.jpg', self.corners)
        self.content = ExampleContent(self, self.inputFile,'resultA.jpg')
        self.setCentralWidget(self.content)
     
        
    def createStructureTensor(self,Ix,Iy):
        h,w = self.inputImage.shape
        structTensor = np.zeros((h,w,2,2),dtype='float64')
        Ixx = Ix*Ix
        Ixy = Ix*Iy
        Iyx = Iy*Ix
        Iyy = Iy*Iy
        
        for x in range(h):
            for y in range(w):
                for i in range(-1,2):
                    for j in range(-1,2):
                        if not(x+j<0 or x+j >= h or y+i <0 or y+i >= w):
                            structTensor[x,y,0,0]+=Ixx[x+j][y+i]
                            structTensor[x,y,0,1]+=Ixy[x+j][y+i]
                            structTensor[x,y,1,0]+=Iyx[x+j][y+i]
                            structTensor[x,y,1,1]+=Iyy[x+j][y+i]
        return structTensor
    def calculateTrace(self,M):
        h,w = self.inputImage.shape
        R = np.zeros((h,w), dtype = 'float64')
        k = 0.04
        for x in range(h):
            for y in range(w):
                deter = M[x,y,0,0]*M[x,y,1,1] - (M[x,y,1,0] * M[x,y,1,0])
                trace = M[x,y,0,0]+M[x,y,1,1]
                R[x,y] = (deter- (trace**2)*k)
        
        return R
    
        
if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    cv2.destroyAllWindows()
    sys.exit(App.exec())
    