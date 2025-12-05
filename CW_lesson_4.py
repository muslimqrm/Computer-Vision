import cv2
import numpy as np

img = cv2.imread('images/img.jpg')
scale = 2
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (9, 9), 2)

img = cv2.equalizeHist(img) # adds contrast

img_edges = cv2.Canny(img, 50, 50)

kernel = np.ones((3, 3), np.uint8)

contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 150:
        x, y, w, h = cv2.boundingRect(cnt)


        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)


        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_y = y- 10 if y - 10 > 10 else y + 10

        text = f'x: {x}, y: {y}, S'
        cv2.putText(img_copy, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow('image', img_edges)
cv2.imshow('image2', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()