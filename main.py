import cv2
import numpy as np
import os

def get_ref_coins():
    ref_coins = {}
    for file in os.listdir('coins/'):
        if file.endswith('.jpg'):
            coin_name = file[:-8]  # Remove 'cara.jpg' or 'coroa.jpg'
            if coin_name not in ref_coins:
                ref_coins[coin_name] = []
            img = cv2.imread(os.path.join('coins/', file), cv2.IMREAD_GRAYSCALE)
            ref_coins[coin_name].append(img)
    return ref_coins

def process_ref_coins(ref_coins):
    processed_coins = {}
    for coin_name, images in ref_coins.items():
        processed_images = []
        for img in images:
            _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
            if np.median(thresh) < 128:
                thresh = cv2.bitwise_not(thresh)
            binary_img = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
            processed_images.append(binary_img)
        processed_coins[coin_name] = processed_images
    return processed_coins

def extract_coin_features(image):
    edges = cv2.Canny(image, 100, 200)
    # Filter edges to keep only strong edges
    strong_edges = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)[1]
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(strong_edges, None)
    return keypoints, descriptors

def main():
    ref_coins = get_ref_coins()
    processed_coins = process_ref_coins(ref_coins)

    for coin_name, images in processed_coins.items():
        for idx, img in enumerate(images):
            keypoints, descriptors = extract_coin_features(img)
            img = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)
            window_name = f"{coin_name} - Image {idx+1}"
            cv2.imshow(window_name, img)

if __name__ == "__main__":
    main()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            break
        if key == ord("q"):
            os._exit(0)
    cv2.destroyAllWindows()
