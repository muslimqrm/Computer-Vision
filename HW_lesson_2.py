import cv2

# image = cv2.imread('images/selfie.jpg')
# image = cv2.resize(image, (image.shape[1] // 4,image.shape[0] // 4))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 100, 100)
# cv2.imwrite('new_selfie.jpg', edges)
#
# cv2.imshow('Image', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image = cv2.imread('images/email.jpg')
image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(gray, 80, 160)

cv2.imwrite('new_email.jpg', edges)

cv2.imshow('Image', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
