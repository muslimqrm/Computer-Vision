import cv2

img = cv2.imread('images/selfie.jpg')

x = img.shape[1] // 4
y = img.shape[0] // 4

img = cv2.resize(img, (x, y))

cv2.rectangle(img, (175,170), (400,560), (0, 255, 0), 2)
cv2.putText(img, 'Muslim Seitkhalil', (170,585), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 1)

cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()