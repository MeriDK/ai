import cv2
from matplotlib import pyplot as plt


img = cv2.imread('lab4/мем.jpg', 0)
cv2.imshow('image', img)
# edges = cv2.Canny(img, 100, 200)

