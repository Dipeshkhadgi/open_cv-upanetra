import cv2
import time
import os
from pose_detector import PoseDetector
from tshirt_overlay import overlay_tshirt
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    pTime = 0
    detector = PoseDetector()
    
    # Initialize hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Load T-shirt images from the specified directory
    tshirt_dir = '/Users/pemagurung/Desktop/enterprise project/Vitrual Try on/upanetra/tshirt folder'
    print("Current working directory:", os.getcwd())
    print("Contents of the specified directory:", os.listdir(tshirt_dir))
    
    tshirt_images = [os.path.join(tshirt_dir, f) for f in os.listdir(tshirt_dir) if f.endswith('.png')]
    if not tshirt_images:
        print("No T-shirt images found in the specified directory.")
        return
    tshirts = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in tshirt_images]
    current_tshirt_index = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Flip the image horizontally
        img = cv2.flip(img, 1)
        
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            img = overlay_tshirt(img, lmList, tshirts[current_tshirt_index])

        # Process the frame for hand detection
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(imgRGB)
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get coordinates of the index finger tip and the wrist
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                
                h, w, c = img.shape
                cx1, cy1 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                cx2, cy2 = int(wrist.x * w), int(wrist.y * h)
                
                # Detect wave gestures
                if cx1 < cx2:  # Hand wave to the left
                    current_tshirt_index = (current_tshirt_index + 1) % len(tshirts)
                elif cx1 > cx2:  # Hand wave to the right
                    current_tshirt_index = (current_tshirt_index - 1) % len(tshirts)
        
        # Display text at the top of the image
        cv2.putText(img, "You are looking great today!!!! Try our virtual dress", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 90), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
