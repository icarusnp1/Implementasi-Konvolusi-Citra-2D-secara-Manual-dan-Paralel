import cv2

img = cv2.imread('DSC_0018.JPG')
img = cv2.resize(img, (5472, 3648))
img = cv2.imwrite('image-21MP.jpg', img)