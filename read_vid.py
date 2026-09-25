import cv2

cap = cv2.VideoCapture("/Users/polymorvic/development/tennis-court-detection/data/vids/maczka1.mp4")
i = 0
while True:
    i += 1
    ret, frame = cap.read()


    if not ret:
        print("the end")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    cv2.imshow("video frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()