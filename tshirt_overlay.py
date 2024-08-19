import cv2
import math

def overlay_tshirt(img, lmList, tshirt_img):
    if len(lmList) >= 24:  # Ensure that the required landmarks are detected
        # Get the coordinates of the shoulders, hips, elbows, wrists, knees, and ankles
        left_shoulder = lmList[11][1:]
        right_shoulder = lmList[12][1:]
        left_hip = lmList[23][1:]
        right_hip = lmList[24][1:]
        left_elbow = lmList[13][1:]
        right_elbow = lmList[14][1:]
        left_wrist = lmList[15][1:]
        right_wrist = lmList[16][1:]
        left_knee = lmList[25][1:]
        right_knee = lmList[26][1:]
        left_ankle = lmList[27][1:]
        right_ankle = lmList[28][1:]

        # Calculate the width and height for the T-shirt based on shoulder and hip distances
        shoulder_width = int(math.dist(left_shoulder, right_shoulder))
        upper_body_height = int(math.dist(left_shoulder, left_hip))

        # Calculate the width and height of the T-shirt
        tshirt_width = int(shoulder_width * 1.8)
        tshirt_height = int(upper_body_height * 1.4)

        # Resize the T-shirt image
        resized_tshirt = cv2.resize(tshirt_img, (tshirt_width, tshirt_height))

        # Calculate the top-left corner of the T-shirt image
        center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
        top_left_x = center_x - tshirt_width // 2
        top_left_y = left_shoulder[1] - int(tshirt_height * 0.25)

        # Overlay the T-shirt image onto the original image
        for i in range(tshirt_height):
            for j in range(tshirt_width):
                if top_left_y + i >= img.shape[0] or top_left_x + j >= img.shape[1]:
                    continue
                alpha = resized_tshirt[i, j, 3] / 255.0  # Use the alpha channel for transparency
                img[top_left_y + i, top_left_x + j] = alpha * resized_tshirt[i, j, :3] + (1 - alpha) * img[top_left_y + i, top_left_x + j]

        # Draw lines for better fitting visualization
        cv2.line(img, left_shoulder, right_shoulder, (0, 255, 0), 2)
        cv2.line(img, left_shoulder, left_elbow, (0, 255, 0), 2)
        cv2.line(img, left_elbow, left_wrist, (0, 255, 0), 2)
        cv2.line(img, right_shoulder, right_elbow, (0, 255, 0), 2)
        cv2.line(img, right_elbow, right_wrist, (0, 255, 0), 2)
        cv2.line(img, left_hip, right_hip, (0, 255, 0), 2)
        cv2.line(img, left_hip, left_knee, (0, 255, 0), 2)
        cv2.line(img, left_knee, left_ankle, (0, 255, 0), 2)
        cv2.line(img, right_hip, right_knee, (0, 255, 0), 2)
        cv2.line(img, right_knee, right_ankle, (0, 255, 0), 2)

    return img
