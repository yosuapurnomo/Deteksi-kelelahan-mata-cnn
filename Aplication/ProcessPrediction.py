import cv2
from keras.models import model_from_json
from Cnn.convolutional import convolutional
from PreProcessing import imgProcessing as proses
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class prediction:

    def __init__(self, data):
        self.prediksi = None
        self.data = data

    def prePro(self):
        prePro = proses.PreProcessing(self.data)
        # prePro.resizeImg()
        prePro.cropEye()
        if prePro.concateImage is not None:
            prePro.grayScale()
            prePro.gaussanScale()
            prePro.cannyScale()
            prePro.convertBinary()
            self.data = prePro.getOutputImg()
        else:
            self.data = None

    def convSkenario1(self):
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.prePro()
        if self.data is not None:
            for _ in range(2):
                conv = convolutional(self.data, kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=0)
                self.data = conv.getData()
            conv.flattenLayer()
            self.data = conv.getData()

    def convSkenario2(self):
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.prePro()
        if self.data is not None:
            for _ in range(2):
                conv = convolutional(self.data, kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=1)
                self.data = conv.getData()
            conv.flattenLayer()
            self.data = conv.getData()

    def convSkenario3(self):
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]]
        ])
        self.prePro()
        if self.data is not None:
            for _ in range(2):
                conv = convolutional(self.data, kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=1)
                self.data = conv.getData()
            conv.flattenLayer()
            self.data = conv.getData()

    def convSkenario4(self):
        kernel = np.array([
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.prePro()
        if self.data is not None:
            for _ in range(2):
                conv = convolutional(self.data, kernel)
                conv.convolutionLayer(0, 1)
                conv.ReLU()
                conv.pollingLayer(mode=0)
                self.data = conv.getData()
            conv.flattenLayer()
            self.data = conv.getData()

    def processPredic(self):
        json = open(f'../app/model/model2.json', 'r')
        modelJson = json.read()
        json.close()
        model = model_from_json(modelJson)
        model.load_weights('../app/model/my_model_sken2_k1.h5')
        self.prediksi = model.predict(np.reshape(self.data, (1, len(self.data))))
        # print(prediksi)

    def getDataPredict(self):
        return self.prediksi
