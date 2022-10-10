import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import save_model

from PreProcessing.imgProcessing import PreProcessing
from skenarioProject.skenario import skenarioProject
from crossValidation.Kfold import crossValidation
from plotProjectModel.plotModel import plotModel


class main:

    def __init__(self, dataFrame):
        self.labelSkenario = None
        self.dataSkenario = None
        self.dataframe = dataFrame
        self.summaryAccuracy = None

    def splitTrainTes(self, k):
        cv = crossValidation(self.dataframe['file'].values, self.dataframe['label'].values)
        self.dataSkenario = cv.splitData(k)

    def saveWeight(self, skenario, model, best):
        model.save_weights(f'app/model/my_model_sken{skenario}_k{best+1}.h5')

    def saveModel(self, model, k):
        # save_model(model, "app/model/")
        model_json = model.to_json()
        with open(f"app/model/model{k}.json", "w") as json_file:
            json_file.write(model_json)

    def runSkenario(self, skenario, epoch):
        reportTrain = []
        outputModelTrain = []
        outputModelTes = []
        runTimeTrain = []
        runTimeTes = []
        run = skenarioProject(epoch)
        for data in self.dataSkenario:
            run.setData(data)
            run.setDataFrame(self.dataframe)
            if skenario == 1:
                run.trainSkenario1(test=0)
            elif skenario == 2:
                run.trainSkenario2(test=0)
            elif skenario == 3:
                run.trainSkenario3(test=0)
            elif skenario == 4:
                run.trainSkenario4(test=0)
            trainDataSkenario, trainLabelSkenario = run.getDataAndLabel()
            modelTrain = run.modelSkenarioA(trainDataSkenario, trainLabelSkenario)
            outputModelTrain.append(modelTrain)
            reportTrain.append(modelTrain.history.history)
            runTimeTrain.append(np.mean(run.getRunTime()))
            print("Test")
            if skenario == 1:
                run.trainSkenario1(test=1)
            elif skenario == 2:
                run.trainSkenario2(test=1)
            elif skenario == 3:
                run.trainSkenario3(test=1)
            elif skenario == 4:
                run.trainSkenario4(test=1)
            testDataSkenario, testLabelSkenario = run.getDataAndLabel()
            modelTest = run.modelTesting(modelTrain, testDataSkenario, testLabelSkenario)
            outputModelTes.append(modelTest)
            runTimeTes.append(np.mean(run.getRunTime()))
        self.plotting(reportTrain, outputModelTes, runTimeTrain, runTimeTes, outputModelTrain, skenario)

    def plotting(self, modelTrain=None, modelTes=None, runTimeTrain=None, runTimeTes=None, outputModel=None, k=None):
        plot = plotModel()
        print("Model Train", modelTrain)
        plot.setDataModel(modelTrain)
        # PlotingTrain
        plot.plotLossTrain()
        plot.plotAccuracyTrain()
        plot.plotPrecisionTrain()
        plot.plotRecallTrain()
        plot.plotRunTimeTrain(runTimeTrain)

        # PlotingTes
        plot.setDataModel(modelTes)
        df = pd.read_excel('dataAccuracyHasil.xlsx', header=0, index_col=0)
        self.summaryAccuracy = plot.writeDataAccuracyHasil(df, k)
        self.saveExcel()
        plot.plotLossTes()
        plot.plotAccuracyTes()
        plot.plotPrecisionTes()
        plot.plotRecallTes()
        plot.plotRunTimeTes(runTimeTes)
        bestTes = plot.getBestAccuracyTes()
        self.saveModel(outputModel[bestTes], k)
        self.saveWeight(k, outputModel[bestTes], bestTes)
        plot.showPlot()

    def saveExcel(self):
        self.summaryAccuracy.to_excel('dataAccuracyHasil.xlsx')


# dataTrain = pd.read_excel('Dataset.xlsx', header=0, sheet_name='all')
# run = main(dataTrain)
# run.splitTrainTes(4)
# run.runSkenario(3, epoch=100)
# df = pd.read_excel('dataAccuracyHasil.xlsx', header=0, index_col=0)
# df['k1']['Skenario 1'] = 1
# print(df['k1'])
img = cv2.imread('Dataset/MataTerbuka/Train/Picture (9).jpg')
img1 = cv2.imread('Dataset/MataTertutup/Train/Picture (8).jpg')
# img = cv2.imread('Dataset/MataTerbuka/Testing/Picture (6).jpg')
# img1 = cv2.imread('Dataset/MataTertutup/Testing/Picture (8).jpg')
pro = PreProcessing(img)
pro1 = PreProcessing(img1)
# pro.resizeImg()
pro.cropEye()
cont = pro.concateImg(pro.concateImage)
pro.grayScale()
pro.gaussanScale()
pro.cannyScale()
cann = pro.getCannyScale()
pro.convertBinary()
# print(pro.getCannyScale())
pro1.cropEye()
cont1 = pro1.concateImg(pro1.concateImage)
pro1.grayScale()
pro1.gaussanScale()
pro1.cannyScale()
cann1 = pro1.getCannyScale()
pro1.convertBinary()
fig,axes = plt.subplots(2, 5)
axes[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0][0].axis('off')
axes[0][1].imshow(cv2.cvtColor(cont, cv2.COLOR_BGR2RGB))
axes[0][1].axis('off')
axes[0][2].imshow(pro.getGaussanScale(), 'gray')
axes[0][2].axis('off')
axes[0][3].imshow(cann, 'gray')
axes[0][3].axis('off')
axes[0][4].imshow(pro.getOutputImg(), 'gray')
axes[0][4].axis('off')
axes[1][0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[1][0].axis('off')
axes[1][1].imshow(cv2.cvtColor(cont1, cv2.COLOR_BGR2RGB))
axes[1][1].axis('off')
axes[1][2].imshow(pro1.getGaussanScale(), 'gray')
axes[1][2].axis('off')
axes[1][3].imshow(cann1, 'gray')
axes[1][3].axis('off')
axes[1][4].imshow(pro1.getOutputImg(), 'gray')
axes[1][4].axis('off')
# axes[1][1].imshow(pro1.getGaussanScale(), 'gray')
plt.show()
# cv2.imshow('Image', pro.getOutputImg())
# cv2.waitKey(0)