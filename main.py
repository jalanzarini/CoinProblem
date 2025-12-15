import cv2
import numpy as np
import os
from perspective import correct_perspective

RESIZE_DIM = 100

def get_ref_coins():
    ref_coins = {}
    for file in os.listdir('ref_coins/'):
        if file.endswith('.jpg'):
            coin_name = file[:-8]  # Remove 'cara.jpg' or 'coroa.jpg'
            if coin_name not in ref_coins:
                ref_coins[coin_name] = []
            img = cv2.imread(os.path.join('coins/', file), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
            ref_coins[coin_name].append(img)
    return ref_coins

""" def extract_coin_features(image):

    # Image feature extraction steps here
    # Get magnitude and direction of gradients
    gX = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 1, 0)
    gY = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 0, 1)
    mag = np.sqrt(gX**2 + gY**2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    phase = np.arctan2(gY, gX) * (180 / np.pi)
    
    return cv2.convertScaleAbs(mag), cv2.convertScaleAbs(phase)
 """
def process_coin_image(img):
    
    # Image processing steps here
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    details = img - blurred
    details = cv2.normalize(details, None, -1, 1, cv2.NORM_MINMAX)
    sharped = cv2.addWeighted(img, 0.9, details, 0.4, 0)
    img = cv2.normalize(sharped, None, 0, 1, cv2.NORM_MINMAX)

    return img

def separate_coins(img):
    coins = []
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)

    # Coin separation steps here
    blurred = cv2.GaussianBlur(img,(0, 0), sigmaX=7, sigmaY=7)

    _, thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        # Remove small contours and image border contours
        if cv2.contourArea(cnt) >= img_copy.shape[0]*img_copy.shape[1]*0.9 or cv2.contourArea(cnt) < 500:
            continue

        cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
        #cv2.imshow("Contours", img_copy)

        x, y, w, h = cv2.boundingRect(cnt)
        coin_img = img[y:y+h, x:x+w]
        coin_img = cv2.resize(coin_img, (RESIZE_DIM, RESIZE_DIM))
        coins.append(coin_img)

    return coins

def main():
    ref_coins = get_ref_coins()

    test_img = cv2.imread('test_images/perspective_4.jpg')
    
    coins, ellipses, img_draw = correct_perspective(test_img)

    for i in range(len(coins)):
        coins[i] = cv2.resize(coins[i], (100, 100))

    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    hog_ref_features = {}
    for coin_name, images in ref_coins.items():
        hog_ref_features[coin_name] = []
        for img in images:
            #img = process_coin_image(img)
            for rot in range(0, 360, 10):
                M = cv2.getRotationMatrix2D((RESIZE_DIM/2, RESIZE_DIM/2), rot, 1)
                img_rotated = cv2.warpAffine(img, M, (RESIZE_DIM, RESIZE_DIM))
                h = hog.compute((img_rotated * 255).astype(np.uint8))
                hog_ref_features[coin_name].append(h)
    
    name_value = {
        '5cents': 0.05,
        '5centsc': 0.05,
        '10cents': 0.10,
        '10centsc': 0.10,
        '25cents': 0.25,
        '25centsc': 0.25,
        '50cents': 0.50,
        '50centsc': 0.50,
        '1real': 1.00,
        '1realc': 1.00
    }
    total = 0.0
    for idx, coin in enumerate(coins):
        #coin = process_coin_image(coin)
        h = hog.compute((coin * 255).astype(np.uint8))
        best_match = (-1, None)
        for coin_name, features in hog_ref_features.items():
            for ref_h in features:
                match = np.corrcoef(h.flatten(), ref_h.flatten())[0, 1]
                if match > best_match[0]:
                    best_match = (match, coin_name)
        cv2.imshow(f"Coin {idx+1}", coin)
        print(f"Coin {idx+1}: Detected as {best_match[1]} with score {best_match[0]}")

        if best_match[1] in name_value:
            total += name_value[best_match[1]]

        ellipse = ellipses[idx]
        cv2.ellipse(img_draw, ellipse, (0, 255, 0), 2)
        center_x, center_y = int(ellipse[0][0]), int(ellipse[0][1])
        text_x = center_x - 30
        text_y = int(center_y - max(ellipse[1])/2)
        label = best_match[1]
        cv2.putText(img_draw, f"{label}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    print(f"Total value of coins: R$ {total:.2f}")
    cv2.imshow("Result", img_draw)

if __name__ == "__main__":
    main()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            break
        if key == ord("q"):
            os._exit(0)
    cv2.destroyAllWindows()
