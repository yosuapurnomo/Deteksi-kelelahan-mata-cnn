import time

import cv2
import numpy as np
from Aplication.ProcessPrediction import prediction
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class mengantuk:

    def prediction(self, frame):
        predic = prediction(frame)
        predic.convSkenario2()
        if predic.data is not None:
            predic.processPredic()
            return np.argmax(predic.getDataPredict()[0])
        else: return None

    def runWindow(self):
        video = cv2.VideoCapture(0, cv2.CAP_MSMF)
        predic = []
        loop = True
        text = False
        start = time.time()
        video.set(cv2.CAP_PROP_FPS, 24)
        while video.isOpened():
            akhir, frame = video.read()
            if text:
                cv2.putText(frame, "Kelelahan", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if not akhir:
                break
            if loop:
                check = self.prediction(frame)
                if check is not None:
                    predic.append(check)
                    if int(time.time()-start) >= 3:
                        if np.mean(predic) < 0.7:
                            print("Buka", np.mean(predic))
                            text = False
                            predic = []
                        else:
                            print("Mengantuk", np.mean(predic))
                            text = True
                            predic = []
                        start = time.time()
                    loop = False
            else: loop = True
            cv2.imshow("View", frame)
            cv2.waitKey(10)
        video.release()
        cv2.destroyAllWindows()


run = mengantuk()
run.runWindow()