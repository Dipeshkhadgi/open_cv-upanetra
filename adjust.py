# import cv2
# import mediapipe as mp
# import time
# import math
# import os

# class poseDetector():
#     def __init__(self, mode=False, upBody=False, smooth=True,
#                  detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
#                                      model_complexity=1, 
#                                      smooth_landmarks=self.smooth,
#                                      min_detection_confidence=self.detectionCon, 
#                                      min_tracking_confidence=self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
#                                            self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return self.lmList

#     def findAngle(self, img, p1, p2, p3, draw=True):
#         x1, y1 = self.lmList[p1][1:]
#         x2, y2 = self.lmList[p2][1:]
#         x3, y3 = self.lmList[p3][1:]

#         angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                              math.atan2(y1 - y2, x1 - x2))
#         if angle < 0:
#             angle += 360

#         if draw:
#             cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#             cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#             cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#             cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#             cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#             cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#         return angle

# def overlay_tshirt(img, lmList, tshirt_img):
#     if len(lmList) >= 24:  # Ensure that the required landmarks are detected
#         # Get the coordinates of the shoulders and hips
#         left_shoulder = lmList[11][1:]
#         right_shoulder = lmList[12][1:]
#         left_hip = lmList[23][1:]
#         right_hip = lmList[24][1:]

#         # Calculate the width and height for the T-shirt based on shoulder and hip distances
#         shoulder_width = int(math.dist(left_shoulder, right_shoulder))
#         upper_body_height = int(math.dist(left_shoulder, left_hip))

#         # Calculate the width and height of the T-shirt
#         tshirt_width = int(shoulder_width * 1.5)
#         tshirt_height = int(upper_body_height * 1.2)

#         # Resize the T-shirt image
#         resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

#         # Calculate the top-left corner of the T-shirt image
#         center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
#         top_left_x = center_x - tshirt_width // 2
#         top_left_y = left_shoulder[1] - int(tshirt_height * 0.25)

#         # Overlay the T-shirt image onto the original image
#         for i in range(tshirt_height):
#             for j in range(tshirt_width):
#                 if top_left_y + i >= img.shape[0] or top_left_x + j >= img.shape[1]:
#                     continue
#                 alpha = resized_tshirt[i, j, 3] / 255.0  # Use the alpha channel for transparency
#                 img[top_left_y + i, top_left_x + j] = alpha * resized_tshirt[i, j, :3] + (1 - alpha) * img[top_left_y + i, top_left_x + j]
#     return img

# def main():
#     cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
#     pTime = 0
#     detector = poseDetector()

#     # Load T-shirt images from the specified directory
#     tshirt_dir = '/Users/pemagurung/Desktop/enterprise project/Vitrual Try on/upanetra/tshirt folder'
#     print("Current working directory:", os.getcwd())
#     print("Contents of the specified directory:", os.listdir(tshirt_dir))
    
#     tshirt_images = [os.path.join(tshirt_dir, f) for f in os.listdir(tshirt_dir) if f.endswith('.png')]
#     if not tshirt_images:
#         print("No T-shirt images found in the specified directory.")
#         return
#     tshirts = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in tshirt_images]
#     current_tshirt_index = 0

#     while True:
#         success, img = cap.read()
#         if not success:
#             break
        
#         # Flip the image horizontally
#         img = cv2.flip(img, 1)
        
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             img = overlay_tshirt(img, lmList, tshirts[current_tshirt_index])

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 0), 3)

#         cv2.imshow("Image", img)

#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break
#         elif key & 0xFF in [ord('n'), 83]:  # Next T-shirt with 'n' or right arrow key
#             if current_tshirt_index < len(tshirts) - 1:
#                 current_tshirt_index += 1
#         elif key & 0xFF in [ord('p'), 81]:  # Previous T-shirt with 'p' or left arrow key
#             if current_tshirt_index > 0:
#                 current_tshirt_index -= 1

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()













































# import cv2
# import mediapipe as mp
# import time
# import math
# import os

# class poseDetector():
#     def __init__(self, mode=False, upBody=False, smooth=True,
#                  detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
#                                      model_complexity=1, 
#                                      smooth_landmarks=self.smooth,
#                                      min_detection_confidence=self.detectionCon, 
#                                      min_tracking_confidence=self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
#                                            self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return self.lmList

#     def findAngle(self, img, p1, p2, p3, draw=True):
#         x1, y1 = self.lmList[p1][1:]
#         x2, y2 = self.lmList[p2][1:]
#         x3, y3 = self.lmList[p3][1:]

#         angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                              math.atan2(y1 - y2, x1 - x2))
#         if angle < 0:
#             angle += 360

#         if draw:
#             cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#             cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#             cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#             cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#             cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#             cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#         return angle

# def overlay_tshirt(img, lmList, tshirt_img):
#     if len(lmList) >= 24:  # Ensure that the required landmarks are detected
#         # Get the coordinates of the shoulders, hips, elbows, and wrists
#         left_shoulder = lmList[11][1:]
#         right_shoulder = lmList[12][1:]
#         left_hip = lmList[23][1:]
#         right_hip = lmList[24][1:]
#         left_elbow = lmList[13][1:]
#         right_elbow = lmList[14][1:]
#         left_wrist = lmList[15][1:]
#         right_wrist = lmList[16][1:]

#         # Calculate the width and height for the T-shirt based on shoulder and hip distances
#         shoulder_width = int(math.dist(left_shoulder, right_shoulder))
#         upper_body_height = int(math.dist(left_shoulder, left_hip))

#         # Calculate the width and height of the T-shirt
#         tshirt_width = int(shoulder_width * 1.5)
#         tshirt_height = int(upper_body_height * 1.2)

#         # Resize the T-shirt image
#         resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

#         # Calculate the top-left corner of the T-shirt image
#         center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
#         top_left_x = center_x - tshirt_width // 2
#         top_left_y = left_shoulder[1] - int(tshirt_height * 0.25)

#         # Overlay the T-shirt image onto the original image
#         for i in range(tshirt_height):
#             for j in range(tshirt_width):
#                 if top_left_y + i >= img.shape[0] or top_left_x + j >= img.shape[1]:
#                     continue
#                 alpha = resized_tshirt[i, j, 3] / 255.0  # Use the alpha channel for transparency
#                 img[top_left_y + i, top_left_x + j] = alpha * resized_tshirt[i, j, :3] + (1 - alpha) * img[top_left_y + i, top_left_x + j]

#         # Draw lines for better fitting visualization
#         cv2.line(img, left_shoulder, right_shoulder, (0, 255, 0), 2)
#         cv2.line(img, left_shoulder, left_elbow, (0, 255, 0), 2)
#         cv2.line(img, left_elbow, left_wrist, (0, 255, 0), 2)
#         cv2.line(img, right_shoulder, right_elbow, (0, 255, 0), 2)
#         cv2.line(img, right_elbow, right_wrist, (0, 255, 0), 2)
#         cv2.line(img, left_hip, right_hip, (0, 255, 0), 2)
        
#     return img

# def main():
#     cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
#     pTime = 0
#     detector = poseDetector()

#     # Load T-shirt images from the specified directory
#     tshirt_dir = '/Users/pemagurung/Desktop/enterprise project/Vitrual Try on/upanetra/tshirt folder'
#     print("Current working directory:", os.getcwd())
#     print("Contents of the specified directory:", os.listdir(tshirt_dir))
    
#     tshirt_images = [os.path.join(tshirt_dir, f) for f in os.listdir(tshirt_dir) if f.endswith('.png')]
#     if not tshirt_images:
#         print("No T-shirt images found in the specified directory.")
#         return
#     tshirts = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in tshirt_images]
#     current_tshirt_index = 0

#     while True:
#         success, img = cap.read()
#         if not success:
#             break
        
#         # Flip the image horizontally
#         img = cv2.flip(img, 1)
        
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             img = overlay_tshirt(img, lmList, tshirts[current_tshirt_index])

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 0), 3)

#         cv2.imshow("Image", img)

#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break
#         elif key & 0xFF in [ord('n'), 83]:  # Next T-shirt with 'n' or right arrow key
#             if current_tshirt_index < len(tshirts) - 1:
#                 current_tshirt_index += 1
#         elif key & 0xFF in [ord('p'), 81]:  # Previous T-shirt with 'p' or left arrow key
#             if current_tshirt_index > 0:
#                 current_tshirt_index -= 1

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()








# import cv2
# import mediapipe as mp
# import time
# import math
# import os

# class poseDetector():
#     def __init__(self, mode=False, upBody=False, smooth=True,
#                  detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
#                                      model_complexity=1, 
#                                      smooth_landmarks=self.smooth,
#                                      min_detection_confidence=self.detectionCon, 
#                                      min_tracking_confidence=self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
#                                            self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return self.lmList

# def overlay_tshirt(img, lmList, tshirt_img):
#     if len(lmList) >= 24:  # Ensure that the required landmarks are detected
#         # Get the coordinates of the shoulders, hips, elbows, wrists, knees, and ankles
#         left_shoulder = lmList[11][1:]
#         right_shoulder = lmList[12][1:]
#         left_hip = lmList[23][1:]
#         right_hip = lmList[24][1:]
#         left_elbow = lmList[13][1:]
#         right_elbow = lmList[14][1:]
#         left_wrist = lmList[15][1:]
#         right_wrist = lmList[16][1:]
#         left_knee = lmList[25][1:]
#         right_knee = lmList[26][1:]
#         left_ankle = lmList[27][1:]
#         right_ankle = lmList[28][1:]

#         # Calculate the width and height for the T-shirt based on shoulder and hip distances
#         shoulder_width = int(math.dist(left_shoulder, right_shoulder))
#         upper_body_height = int(math.dist(left_shoulder, left_hip))

#         # Calculate the width and height of the T-shirt
#         tshirt_width = int(shoulder_width * 1.8)
#         tshirt_height = int(upper_body_height * 1.4)

#         # Resize the T-shirt image
#         resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

#         # Calculate the top-left corner of the T-shirt image
#         center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
#         top_left_x = center_x - tshirt_width // 2
#         top_left_y = left_shoulder[1] - int(tshirt_height * 0.25)

#         # Overlay the T-shirt image onto the original image
#         for i in range(tshirt_height):
#             for j in range(tshirt_width):
#                 if top_left_y + i >= img.shape[0] or top_left_x + j >= img.shape[1]:
#                     continue
#                 alpha = resized_tshirt[i, j, 3] / 255.0  # Use the alpha channel for transparency
#                 img[top_left_y + i, top_left_x + j] = alpha * resized_tshirt[i, j, :3] + (1 - alpha) * img[top_left_y + i, top_left_x + j]

#         # Draw lines for better fitting visualization
#         cv2.line(img, left_shoulder, right_shoulder, (0, 255, 0), 2)
#         cv2.line(img, left_shoulder, left_elbow, (0, 255, 0), 2)
#         cv2.line(img, left_elbow, left_wrist, (0, 255, 0), 2)
#         cv2.line(img, right_shoulder, right_elbow, (0, 255, 0), 2)
#         cv2.line(img, right_elbow, right_wrist, (0, 255, 0), 2)
#         cv2.line(img, left_hip, right_hip, (0, 255, 0), 2)
#         cv2.line(img, left_hip, left_knee, (0, 255, 0), 2)
#         cv2.line(img, left_knee, left_ankle, (0, 255, 0), 2)
#         cv2.line(img, right_hip, right_knee, (0, 255, 0), 2)
#         cv2.line(img, right_knee, right_ankle, (0, 255, 0), 2)

#     return img

# def main():
#     cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
#     pTime = 0
#     detector = poseDetector()

#     # Load T-shirt images from the specified directory
#     tshirt_dir = '/Users/pemagurung/Desktop/enterprise project/Vitrual Try on/upanetra/tshirt folder'
#     print("Current working directory:", os.getcwd())
#     print("Contents of the specified directory:", os.listdir(tshirt_dir))
    
#     tshirt_images = [os.path.join(tshirt_dir, f) for f in os.listdir(tshirt_dir) if f.endswith('.png')]
#     if not tshirt_images:
#         print("No T-shirt images found in the specified directory.")
#         return
#     tshirts = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in tshirt_images]
#     current_tshirt_index = 0

#     while True:
#         success, img = cap.read()
#         if not success:
#             break
        
#         # Flip the image horizontally
#         img = cv2.flip(img, 1)
        
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             img = overlay_tshirt(img, lmList, tshirts[current_tshirt_index])

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 0), 3)

#         cv2.imshow("Image", img)

#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break
#         elif key & 0xFF in [ord('n'), 83]:  # Next T-shirt with 'n' or right arrow key
#             if current_tshirt_index < len(tshirts) - 1:
#                 current_tshirt_index += 1
#         elif key & 0xFF in [ord('p'), 81]:  # Previous T-shirt with 'p' or left arrow key
#             if current_tshirt_index > 0:
#                 current_tshirt_index -= 1

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




















# import cv2
# import mediapipe as mp
# import time
# import math
# import os

# class poseDetector():
#     def __init__(self, mode=False, upBody=False, smooth=True,
#                  detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
#                                      model_complexity=1, 
#                                      smooth_landmarks=self.smooth,
#                                      min_detection_confidence=self.detectionCon, 
#                                      min_tracking_confidence=self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
#                                            self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return self.lmList

# def overlay_tshirt(img, lmList, tshirt_img):
#     if len(lmList) >= 24:  # Ensure that the required landmarks are detected
#         # Get the coordinates of the shoulders, hips, elbows, wrists, knees, and ankles
#         left_shoulder = lmList[11][1:]
#         right_shoulder = lmList[12][1:]
#         left_hip = lmList[23][1:]
#         right_hip = lmList[24][1:]
#         left_elbow = lmList[13][1:]
#         right_elbow = lmList[14][1:]
#         left_wrist = lmList[15][1:]
#         right_wrist = lmList[16][1:]
#         left_knee = lmList[25][1:]
#         right_knee = lmList[26][1:]
#         left_ankle = lmList[27][1:]
#         right_ankle = lmList[28][1:]

#         # Calculate the width and height for the T-shirt based on shoulder and hip distances
#         shoulder_width = int(math.dist(left_shoulder, right_shoulder))
#         upper_body_height = int(math.dist(left_shoulder, left_hip))

#         # Calculate the width and height of the T-shirt
#         tshirt_width = int(shoulder_width * 1.8)
#         tshirt_height = int(upper_body_height * 1.4)

#         # Resize the T-shirt image
#         resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

#         # Calculate the top-left corner of the T-shirt image
#         center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
#         top_left_x = center_x - tshirt_width // 2
#         top_left_y = left_shoulder[1] - int(tshirt_height * 0.25)

#         # Overlay the T-shirt image onto the original image
#         for i in range(tshirt_height):
#             for j in range(tshirt_width):
#                 if top_left_y + i >= img.shape[0] or top_left_x + j >= img.shape[1]:
#                     continue
#                 alpha = resized_tshirt[i, j, 3] / 255.0  # Use the alpha channel for transparency
#                 img[top_left_y + i, top_left_x + j] = alpha * resized_tshirt[i, j, :3] + (1 - alpha) * img[top_left_y + i, top_left_x + j]

#         # Draw lines for better fitting visualization
#         cv2.line(img, left_shoulder, right_shoulder, (0, 255, 0), 2)
#         cv2.line(img, left_shoulder, left_elbow, (0, 255, 0), 2)
#         cv2.line(img, left_elbow, left_wrist, (0, 255, 0), 2)
#         cv2.line(img, right_shoulder, right_elbow, (0, 255, 0), 2)
#         cv2.line(img, right_elbow, right_wrist, (0, 255, 0), 2)
#         cv2.line(img, left_hip, right_hip, (0, 255, 0), 2)
#         cv2.line(img, left_hip, left_knee, (0, 255, 0), 2)
#         cv2.line(img, left_knee, left_ankle, (0, 255, 0), 2)
#         cv2.line(img, right_hip, right_knee, (0, 255, 0), 2)
#         cv2.line(img, right_knee, right_ankle, (0, 255, 0), 2)

#     return img

# def main():
#     cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
#     pTime = 0
#     detector = poseDetector()
    
#     # Initialize hand detection
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#     # Load T-shirt images from the specified directory
#     tshirt_dir = '/Users/pemagurung/Desktop/enterprise project/Vitrual Try on/upanetra/tshirt folder'
#     print("Current working directory:", os.getcwd())
#     print("Contents of the specified directory:", os.listdir(tshirt_dir))
    
#     tshirt_images = [os.path.join(tshirt_dir, f) for f in os.listdir(tshirt_dir) if f.endswith('.png')]
#     if not tshirt_images:
#         print("No T-shirt images found in the specified directory.")
#         return
#     tshirts = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in tshirt_images]
#     current_tshirt_index = 0

#     while True:
#         success, img = cap.read()
#         if not success:
#             break
        
#         # Flip the image horizontally
#         img = cv2.flip(img, 1)
        
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             img = overlay_tshirt(img, lmList, tshirts[current_tshirt_index])

#         # Process the frame for hand detection
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         hand_results = hands.process(imgRGB)
        
#         if hand_results.multi_hand_landmarks:
#             for hand_landmarks in hand_results.multi_hand_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 # Get coordinates of the index finger tip and the wrist
#                 index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
#                 h, w, c = img.shape
#                 cx1, cy1 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
#                 cx2, cy2 = int(wrist.x * w), int(wrist.y * h)
                
#                 # Detect wave gestures
#                 if cx1 < cx2:  # Hand wave to the left
#                     current_tshirt_index = (current_tshirt_index + 1) % len(tshirts)
#                 elif cx1 > cx2:  # Hand wave to the right
#                     current_tshirt_index = (current_tshirt_index - 1) % len(tshirts)
        
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 0), 3)

#         cv2.imshow("Image", img)

#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()



# import cv2
# import mediapipe as mp
# import time
# import math
# import os

# class poseDetector():
#     def __init__(self, mode=False, upBody=False, smooth=True,
#                  detectionCon=0.5, trackCon=0.5):
#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
#                                      model_complexity=1, 
#                                      smooth_landmarks=self.smooth,
#                                      min_detection_confidence=self.detectionCon, 
#                                      min_tracking_confidence=self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
#                                            self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return self.lmList

# def overlay_tshirt(img, lmList, tshirt_img):
#     if len(lmList) >= 24:  # Ensure that the required landmarks are detected
#         # Get the coordinates of the shoulders, hips, elbows, wrists, knees, and ankles
#         left_shoulder = lmList[11][1:]
#         right_shoulder = lmList[12][1:]
#         left_hip = lmList[23][1:]
#         right_hip = lmList[24][1:]
#         left_elbow = lmList[13][1:]
#         right_elbow = lmList[14][1:]
#         left_wrist = lmList[15][1:]
#         right_wrist = lmList[16][1:]
#         left_knee = lmList[25][1:]
#         right_knee = lmList[26][1:]
#         left_ankle = lmList[27][1:]
#         right_ankle = lmList[28][1:]

#         # Calculate the width and height for the T-shirt based on shoulder and hip distances
#         shoulder_width = int(math.dist(left_shoulder, right_shoulder))
#         upper_body_height = int(math.dist(left_shoulder, left_hip))

#         # Calculate the width and height of the T-shirt
#         tshirt_width = int(shoulder_width * 1.8)
#         tshirt_height = int(upper_body_height * 1.4)

#         # Resize the T-shirt image
#         resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

#         # Calculate the top-left corner of the T-shirt image
#         center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
#         top_left_x = center_x - tshirt_width // 2
#         top_left_y = left_shoulder[1] - int(tshirt_height * 0.25)

#         # Overlay the T-shirt image onto the original image
#         for i in range(tshirt_height):
#             for j in range(tshirt_width):
#                 if top_left_y + i >= img.shape[0] or top_left_x + j >= img.shape[1]:
#                     continue
#                 alpha = resized_tshirt[i, j, 3] / 255.0  # Use the alpha channel for transparency
#                 img[top_left_y + i, top_left_x + j] = alpha * resized_tshirt[i, j, :3] + (1 - alpha) * img[top_left_y + i, top_left_x + j]

#         # Draw lines for better fitting visualization
#         cv2.line(img, left_shoulder, right_shoulder, (0, 255, 0), 2)
#         cv2.line(img, left_shoulder, left_elbow, (0, 255, 0), 2)
#         cv2.line(img, left_elbow, left_wrist, (0, 255, 0), 2)
#         cv2.line(img, right_shoulder, right_elbow, (0, 255, 0), 2)
#         cv2.line(img, right_elbow, right_wrist, (0, 255, 0), 2)
#         cv2.line(img, left_hip, right_hip, (0, 255, 0), 2)
#         cv2.line(img, left_hip, left_knee, (0, 255, 0), 2)
#         cv2.line(img, left_knee, left_ankle, (0, 255, 0), 2)
#         cv2.line(img, right_hip, right_knee, (0, 255, 0), 2)
#         cv2.line(img, right_knee, right_ankle, (0, 255, 0), 2)

#     return img

# def main():
#     cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
#     pTime = 0
#     detector = poseDetector()
    
#     # Initialize hand detection
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#     # Load T-shirt images from the specified directory
#     tshirt_dir = '/Users/pemagurung/Desktop/enterprise project/Vitrual Try on/upanetra/tshirt folder'
#     print("Current working directory:", os.getcwd())
#     print("Contents of the specified directory:", os.listdir(tshirt_dir))
    
#     tshirt_images = [os.path.join(tshirt_dir, f) for f in os.listdir(tshirt_dir) if f.endswith('.png')]
#     if not tshirt_images:
#         print("No T-shirt images found in the specified directory.")
#         return
#     tshirts = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in tshirt_images]
#     current_tshirt_index = 0

#     while True:
#         success, img = cap.read()
#         if not success:
#             break
        
#         # Flip the image horizontally
#         img = cv2.flip(img, 1)
        
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             img = overlay_tshirt(img, lmList, tshirts[current_tshirt_index])

#         # Process the frame for hand detection
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         hand_results = hands.process(imgRGB)
        
#         if hand_results.multi_hand_landmarks:
#             for hand_landmarks in hand_results.multi_hand_landmarks:
#                 mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
#                 # Get coordinates of the index finger tip and the wrist
#                 index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
#                 h, w, c = img.shape
#                 cx1, cy1 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
#                 cx2, cy2 = int(wrist.x * w), int(wrist.y * h)
                
#                 # Detect wave gestures
#                 if cx1 < cx2:  # Hand wave to the left
#                     current_tshirt_index = (current_tshirt_index + 1) % len(tshirts)
#                 elif cx1 > cx2:  # Hand wave to the right
#                     current_tshirt_index = (current_tshirt_index - 1) % len(tshirts)
        
#         # Display text at the top of the image
#         cv2.putText(img, "You are looking great today!!!! Try our virtual dress", (50, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (70, 90), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 0), 3)

#         cv2.imshow("Image", img)

#         key = cv2.waitKey(1)
#         if key & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()