import numpy as np


class convolutional:

    def __init__(self, img, kernel):
        self.img = img
        self.kernel = kernel

    def convolutionLayer(self, padding=0, stride=1):
        w = int((self.img.shape[0] - self.kernel.shape[1] + (2 * padding)) / stride) + 1
        h = int((self.img.shape[1] - self.kernel.shape[2] + (2 * padding)) / stride) + 1
        if self.img.shape[0]%self.kernel.shape[1] > 0:
            tes = np.pad(self.img, ((0, 1), (0, 0)))
        if self.img.shape[1]%self.kernel.shape[2] > 0:
            tes = np.pad(self.img, ((0, 0), (0, 1)))
        else:
            tes = self.img
        conv = np.zeros((w, h))
        yImg = 0
        for y in range(0, w):
            xImg = 0
            for x in range(0, h):
                conv[y][x] = np.sum([
                    np.sum([tes[yImg:yImg + self.kernel.shape[1],
                            xImg:xImg + self.kernel.shape[2]] * self.kernel[i] for i in range(self.kernel.shape[0])])])
                xImg += stride
            yImg += stride
        self.img = conv

    def ReLU(self):
        self.img = np.where(self.img < 0, 0, self.img)

    def pollingLayer(self, mode=0):
        if self.img.shape[0] % 2 == 1:
            self.img = np.pad(self.img, ((0, 1), (0, 0)))
        if self.img.shape[1] % 2 == 1:
            self.img = np.pad(self.img, ((0, 0), (0, 1)))
        poll = np.zeros((int(self.img.shape[0] / 2), int(self.img.shape[1] / 2)))
        # 0=Max, 1=Average
        yOut = xOut = 0
        for y in range(0, self.img.shape[0], 2):
            for x in range(0, self.img.shape[1], 2):
                if mode == 0:
                    poll[yOut][xOut] = np.max(self.img[y:y + 2, x:x + 2])
                else:
                    poll[yOut][xOut] = np.average(self.img[y:y + 2, x:x + 2])
                xOut += 1
            yOut += 1
            xOut = 0
        self.img = poll

    def flattenLayer(self):
        self.img = np.reshape(self.img, (self.img.shape[0] * self.img.shape[1]))

    def getData(self):
        return self.img
