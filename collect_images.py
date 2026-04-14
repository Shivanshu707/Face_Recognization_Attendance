import cv2
import os

name = input("Enter name: ")
path = f"dataset/{name}"

os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        img_path = f"{path}/{count}.jpg"
        cv2.imwrite(img_path, frame)
        count += 1
        print(f"Saved {count}")

    if count == 50:
        break

cap.release()
cv2.destroyAllWindows()