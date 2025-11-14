import cv2

# img = cv2.imread('images/image.jpg')
# img = cv2.resize(img, (500, 300))
# cv2.imshow('image', img)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (600, 400))
    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()