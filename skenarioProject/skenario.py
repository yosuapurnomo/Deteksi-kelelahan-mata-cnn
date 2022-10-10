import time

import cv2
import numpy as np

from Cnn.convolutional import convolutional
from Cnn.neuralNetwork import nn
from PreProcessing import imgProcessing as proses


class skenarioProject:

    def __init__(self, epoch):
        self.arrLabel = None
        self.arrData = None
        self.epoch = epoch
        self.dataSkenario = None
        self.dataFrame = None
        self.kernel = None
        self.data = None
        self.time = None

    def resetRunTime(self):
        self.time = []

    def getRunTime(self):
        return self.time

    def setData(self, data):
        self.data = data

    def setDataFrame(self, dataFrame):
        self.dataFrame = dataFrame

    def setKernel(self, kernel):
        self.kernel = kernel

    def prePro(self, img):
        prePro = proses.PreProcessing(img)
        prePro.cropEye()
        prePro.grayScale()
        prePro.gaussanScale()
        prePro.cannyScale()
        prePro.convertBinary()
        return prePro.getOutputImg()

    # (Skenario 1) input data pada Convolution bernilai Binary | 2 Filter Kernel
    def trainSkenario1(self, test=0):
        self.arrData = []
        self.arrLabel = []
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.resetRunTime()
        self.setKernel(kernel)
        for data in self.data[test]:
            start_time = time.time()
            img = cv2.imread(self.dataFrame['file'].values[data])
            self.arrLabel.append(self.dataFrame['label'].values[data])
            prePro = self.prePro(img)
            for _ in range(2):
                conv = convolutional(prePro, self.kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=0)
                prePro = conv.getData()
            conv.flattenLayer()
            self.arrData.append(conv.getData())
            self.time.append(time.time() - start_time)

    def getDataAndLabel(self):
        arrData = np.reshape(self.arrData, (len(self.arrData), self.arrData[0].shape[0], 1))
        arrLabel = np.reshape(self.arrLabel, (len(self.arrLabel), 1))
        return arrData, arrLabel

    # (Skenario 2) input data pada Convolution bernilai non-Binary | 2 Filter Kernel
    def trainSkenario2(self, test=0):
        self.arrData = []
        self.arrLabel = []
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.resetRunTime()
        self.setKernel(kernel)
        for data in self.data[test]:
            start_time = time.time()
            img = cv2.imread(self.dataFrame['file'].values[data])
            self.arrLabel.append(self.dataFrame['label'].values[data])
            prePro = self.prePro(img)
            for _ in range(2):
                conv = convolutional(prePro, self.kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=1)
                prePro = conv.getData()
            conv.flattenLayer()
            self.arrData.append(conv.getData())
            self.time.append(time.time() - start_time)

    # (Skenario3) menggunakan 1 filter kernel Horizontal dan average poolingLayer
    def trainSkenario3(self, test=0):
        self.arrData = []
        self.arrLabel = []
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]]
        ])
        self.resetRunTime()
        self.setKernel(kernel)
        for data in self.data[test]:
            start_time = time.time()
            img = cv2.imread(self.dataFrame['file'].values[data])
            self.arrLabel.append(self.dataFrame['label'].values[data])
            prePro = self.prePro(img)
            for _ in range(2):
                conv = convolutional(prePro, self.kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=1)
                prePro = conv.getData()
            conv.flattenLayer()
            self.arrData.append(conv.getData())
            self.time.append(time.time() - start_time)

    # (Skenario4) menggunakan 1 filter kernel vertical dan average poolingLayer
    def trainSkenario4(self, test=0):
        self.arrData = []
        self.arrLabel = []
        kernel = np.array([
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.resetRunTime()
        self.setKernel(kernel)
        for data in self.data[test]:
            start_time = time.time()
            img = cv2.imread(self.dataFrame['file'].values[data])
            self.arrLabel.append(self.dataFrame['label'].values[data])
            prePro = self.prePro(img)
            for _ in range(2):
                conv = convolutional(prePro, self.kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=0)
                prePro = conv.getData()
            conv.flattenLayer()
            self.arrData.append(conv.getData())
            self.time.append(time.time() - start_time)

    def modelSkenarioA(self, arrData, arrLabel):
        modelNN = nn(arrData, arrLabel)
        modelNN.nnTrain(layer=4, note=[18, 10, 4, 2],
                        activation=['sigmoid', 'sigmoid', 'sigmoid', 'softmax'])
        modelNN.modelCompileTrain(self.epoch)
        return modelNN.getModel()

    def modelTesting(self, model, arrData, arrLabel):
        modelNN = nn(arrData, arrLabel)
        modelTes = modelNN.nnTest(model)
        return modelTes
    