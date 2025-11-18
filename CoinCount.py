import cv2
import numpy as np
import os

# Open images
img = cv2.imread('test_images/test1.jpg', cv2.IMREAD_COLOR_BGR)
fivecentscara = cv2.imread('coins/5centscara.jpg', cv2.IMREAD_GRAYSCALE)
fivecentscoroa = cv2.imread('coins/5centscoroa.jpg', cv2.IMREAD_GRAYSCALE)
fivecents = (fivecentscara, fivecentscoroa)

tencentscara = cv2.imread('coins/10centscara.jpg', cv2.IMREAD_GRAYSCALE)
tencentscoroa = cv2.imread('coins/10centscoroa.jpg', cv2.IMREAD_GRAYSCALE)
tencents = (tencentscara, tencentscoroa)

tfivecentscara = cv2.imread('coins/25centscara.jpg', cv2.IMREAD_GRAYSCALE)
tfivecentscoroa = cv2.imread('coins/25centscoroa.jpg', cv2.IMREAD_GRAYSCALE)
tfivecents = (tfivecentscara, tfivecentscoroa)

fftcentscara = cv2.imread('coins/50centscara.jpg', cv2.IMREAD_GRAYSCALE)
fftcentscoroa = cv2.imread('coins/50centscoroa.jpg', cv2.IMREAD_GRAYSCALE)
fftcents = (fftcentscara, fftcentscoroa)

onercara = cv2.imread('coins/1realcara.jpg', cv2.IMREAD_GRAYSCALE)
onercoroa = cv2.imread('coins/1realcoroa.jpg', cv2.IMREAD_GRAYSCALE)
oner = (onercara, onercoroa)

ref_coins = {
    "5cents": fivecents,
    "10cents": tencents,
    "25cents": tfivecents,
    "50cents": fftcents,
    "1real": oner
}


cv2.imshow('output', img)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
        break
    elif key == ord('q'):
        os._exit(0)
cv2.destroyAllWindows()