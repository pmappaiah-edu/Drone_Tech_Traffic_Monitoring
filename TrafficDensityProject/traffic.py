import cv2
import numpy as np

# Load video
cap = cv2.VideoCapture("drone_video.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=1000,
    varThreshold=25,
    detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    # Apply background subtraction
    frame = cv2.GaussianBlur(frame, (5,5), 0)
    fgmask = fgbg.apply(frame, learningRate=0.001)

    # Remove noise
    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        fgmask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    vehicle_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 400:  # Adjust depending on video
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            vehicle_count += 1

    # Density classification
    if vehicle_count < 10:
        density = "LOW"
        color = (0,255,0)
    elif vehicle_count < 25:
        density = "MEDIUM"
        color = (0,255,255)
    else:
        density = "HIGH"
        color = (0,0,255)

    # Display results
    cv2.putText(frame, f"Vehicles: {vehicle_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,0,0), 2)

    cv2.putText(frame, f"Traffic: {density}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("Traffic Monitoring", frame)
    cv2.imshow("Foreground Mask", fgmask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()