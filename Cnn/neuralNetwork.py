import numpy as np
import pandas as pd
# from PreProcessing.PreProcessing import PreProcessing
# from CNN.Convolutional import main as convolutional
from keras.models import Sequential
from keras.layers import Dense
import keras_metrics
from keras import backend as K
import cv2


class nn:

    def __init__(self, arrData, arrLabel):
        self.outputLayer = None
        self.arrData = arrData
        self.arrLabel = arrLabel
        self.weight = None
        self.model = None

    def setWeight(self, weight):
        self.weight = weight

    def getModel(self):
        return self.model

    def getOutputLayer(self):
        return self.outputLayer

    def nnTrain(self, layer, note, activation):
        self.model = Sequential()
        if layer == len(note) == len(activation):
            self.model.add(Dense(note[0], activation=activation[0], input_shape=[self.arrData[0].shape[0]]))
            for i in range(1, layer):
                self.model.add(Dense(note[i], activation=activation[i]))

    def modelCompileTrain(self, epoch):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                           metrics=['accuracy', keras_metrics.precision(),
                                    keras_metrics.recall()])
        self.model.summary()
        self.model.fit(self.arrData, self.arrLabel, epochs=epoch, batch_size=1)
        self.outputLayer = [K.function([self.model.input], [outputLayer.output])([self.arrData]) for outputLayer in
                            self.model.layers]
        # print(outputLayer[layer-1])

    # def modelCompilePredict(self):
    #     self.model.set_weights(self.weight)
    #     self.model.pre

    def nnTest(self, model):
        modelTes = model.evaluate(self.arrData, self.arrLabel, batch_size=1)
        return modelTes
