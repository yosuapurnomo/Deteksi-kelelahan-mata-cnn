import cv2
from keras.models import model_from_json
from Cnn.convolutional import convolutional
from PreProcessing import imgProcessing as proses
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class openClose:

    def __init__(self):
        self.prediksi = None
        self.data = None

    def prePro(self, img):
        prePro = proses.PreProcessing(img)
        prePro.cropEye()
        prePro.grayScale()
        prePro.gaussanScale()
        prePro.cannyScale()
        prePro.convertBinary()
        self.data = prePro.getOutputImg()

    def convSkenario1(self, x):
        img = cv2.imread(x)
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.prePro(img)
        for _ in range(2):
            conv = convolutional(self.data, kernel)
            conv.convolutionLayer(0, 1)
            conv.ReLU()
            conv.pollingLayer(mode=0)
            self.data = conv.getData()
        conv.flattenLayer()
        self.data = conv.getData()

    def convSkenario2(self, x):
        img = cv2.imread(x)
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]],
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.prePro(img)
        for _ in range(2):
            conv = convolutional(self.data, kernel)
            conv.convolutionLayer(0, 1)
            conv.ReLU()
            conv.pollingLayer(mode=1)
            self.data = conv.getData()
        conv.flattenLayer()
        self.data = conv.getData()

    def convSkenario3(self, x):
        img = cv2.imread(x)
        kernel = np.array([
            [[1, 1, 1],
             [0, 0, 0],
             [-1, -1, -1]]
        ])
        self.prePro(img)
        for _ in range(2):
            conv = convolutional(self.data, kernel)
            conv.convolutionLayer(0, 1)
            conv.ReLU()
            conv.pollingLayer(mode=1)
            self.data = conv.getData()
        conv.flattenLayer()
        self.data = conv.getData()

    def convSkenario4(self, x):
        img = cv2.imread(x)
        kernel = np.array([
            [[-1, 0, 1],
             [-1, 0, 1],
             [-1, 0, 1]]
        ])
        self.prePro(img)
        for _ in range(2):
            conv = convolutional(self.data, kernel)
            conv.convolutionLayer(0, 1)
            conv.ReLU()
            conv.pollingLayer(mode=0)
            self.data = conv.getData()
        conv.flattenLayer()
        self.data = conv.getData()

    def setModelWeight(self):
        # weight = ['../model/my_model_sken1_k2.h5',
        #           '../model/my_model_sken2_k2.h5',
        #           '../model/my_model_sken3_k4.h5',
        #           '../model/my_model_sken4_k4.h5']
        json = open(f'../model/model2.json', 'r')
        modelJson = json.read()
        json.close()
        model = model_from_json(modelJson)
        model.load_weights('../model/my_model_sken2_k1.h5')
        self.prediksi = model.predict(np.reshape(self.data, (1, len(self.data))))
        # print(prediksi)

    def getDataPredict(self):
        return self.prediksi
