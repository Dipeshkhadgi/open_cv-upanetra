import cvzone
import cv2
import os
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector()

shirtFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtFolderPath)
fixedRatio = 262 / 190  # width of shirt / width of point 11 to 12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0

imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

# Define the additional offset to stretch the shirt towards the right
additional_offset = 1

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    img = cv2.flip(img, 1)  # Keep the image flipped
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        # Print the landmarks to debug
        print(f"Landmarks: {lmList}")

        # Adjust coordinates for the flipped image
        imgWidth = img.shape[1]
        lm11 = [imgWidth - lmList[11][0], lmList[11][1]]
        lm12 = [imgWidth - lmList[12][0], lmList[12][1]]

        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        # Correct widthOfShirt calculation
        widthOfShirt = int(abs(lm12[0] - lm11[0]) * fixedRatio)
        print(f"Width of Shirt: {widthOfShirt}")
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = abs(lm12[0] - lm11[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)
        print(f"Offset: {offset}")

        try:
            # Apply additional offset to stretch the shirt towards the right
            img = cvzone.overlayPNG(img, imgShirt, (lm11[0] - offset[0] - additional_offset, lm11[1] - offset[1]))
        except Exception as e:
            print(f"Error overlaying PNG: {e}")

    # Overlay buttons
    img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
    img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

    # Left hand raised
    if lmList[15][1] < 300:
        counterLeft += 1
        cv2.ellipse(img, (139, 360), (66, 66), 0, 0, counterLeft * selectionSpeed, (0, 255, 0), 20)
        if counterLeft * selectionSpeed > 360:
            counterLeft = 0
            if imageNumber > 0:
                imageNumber -= 1
    # Right hand raised
    elif lmList[16][1] < 300:
        counterRight += 1
        cv2.ellipse(img, (1138, 360), (66, 66), 0, 0, counterRight * selectionSpeed, (0, 255, 0), 20)
        if counterRight * selectionSpeed > 360:
            counterRight = 0
            if imageNumber < len(listShirts) - 1:
                imageNumber += 1
    else:
        counterRight = 0
        counterLeft = 0

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
