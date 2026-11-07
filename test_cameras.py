import cv2

for i in range(3):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"cam {i}: {frame.shape} - showing for 3 seconds...")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print(f"cam {i}: opened but no frame")
    cap.release()

print("Done. Which camera number showed the UVC camera?")
