import cvzone
import cv2
import os
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector()



cap.release()
cv2.destroyAllWindows()
