import cv2
from src.court_detector import CourtDetector

cap = cv2.VideoCapture("test_video.mov")

while True:
    ret, frame = cap.read()

    if not ret:
        print("the end")
        break

    detector = CourtDetector(frame)

    baseline = detector.scan_for_baseline()
    if not baseline:
        continue

    p1, p2 = baseline.limit_to_img(frame)


    cv2.line(frame, p1, p2, (0, 0, 255), 4)

    cv2.imshow("video frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()