import cv2
import numpy as np
from numpy.ma.core import filled

img = np.zeros((800, 512, 3), np.uint8)

# img[100:150, 200:200] = 189, 234, 179
# img[:] = 189, 234, 179

cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
cv2.line(img, (288, 100), (300, 150), (0,0,255), 2)
print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), ((img.shape[1]), (img.shape[0] // 2)), (255, 255, 0), 2)
cv2.circle(img, (150, 150), 50, (255, 255, 0), -1)
cv2.putText(img, 'name', (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

cv2.imshow('prymityvy', img)




cv2.waitKey(0)
cv2.destroyAllWindows()