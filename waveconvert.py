import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./jpg/done_1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs

cv2.imwrite("小波变换之后.png",cA)
plt.imshow(img,'gray')
plt.title('result')
plt.show()
