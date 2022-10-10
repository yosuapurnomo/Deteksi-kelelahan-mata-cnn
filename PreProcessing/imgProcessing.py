import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2


class PreProcessing:
    def __init__(self, data):
        self.canny = None
        self.gausanFilter = None
        self.data = data
        self.detector = FaceMeshDetector(maxFaces=1)
        self.concateImage = None
        # self.cropEye()
        # self.grayScale()
        # self.gaussanScale()
        # self.cannyScale()
        # self.convertBinary()

    def resizeImg(self):
        self.data = cv2.resize(self.data, (100, 100), interpolation = cv2.INTER_AREA)

    def cropEye(self):
        imgClose, facesClose = self.detector.findFaceMesh(self.data.copy())
        if len(facesClose) != 0:
            self.concateImage = np.array([
                cv2.resize(self.data[facesClose[0][223][1]:facesClose[0][230][1],
                           facesClose[0][226][0]:facesClose[0][244][0]], (25, 15)),
                cv2.resize(self.data[facesClose[0][443][1]:facesClose[0][450][1],
                           facesClose[0][464][0]:facesClose[0][342][0]], (25, 15))])
        else:self.concateImage = None

    def concateImg(self, image):
        return np.concatenate((image[0], image[1]), axis=1)

    def grayScale(self):
        self.concateImage = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in self.concateImage]

    def gaussanScale(self):
        self.gausanFilter = [cv2.GaussianBlur(x, (3, 3), 0) for x in self.concateImage]

    def cannyScale(self):
        self.canny = [cv2.Canny(x, 20, 105) for x in self.gausanFilter]

    def convertBinary(self):
        _, self.binary = cv2.threshold(self.concateImg(self.canny), 80, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     metode otsu

    def getOutputImg(self):
        return self.binary

    def getCannyScale(self):
        return self.concateImg(self.canny)

    def getGaussanScale(self):
        return self.concateImg(self.gausanFilter)


# idList = [BawahX, BawahY, AtasX, AtasY]
# idList = [[226, 230, 223, 244] Mata Tertutup,
#         [464, 450, 342, 443]] Mata Terbuka