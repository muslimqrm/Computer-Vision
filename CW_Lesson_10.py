import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_image(color, shape):
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "rectangle":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 100]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

X = []
y = []

colors = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': ( 0, 255,0),
}

shapes = ['circle', 'rectangle', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            features = [mean_color[0], mean_color[1], mean_color[2], mean_color[3]]

            X.append(features)
            y.append(f'{color_name}_{shape}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(accuracy)

test_img = generate_image((255, 13, 10), 'circle')
mean_color = cv2.mean(test_img)[:3]
features = model.predict([mean_color])
print(features)
cv2.imshow('image', test_img)
cv2.waitKey()
cv2.destroyAllWindows()