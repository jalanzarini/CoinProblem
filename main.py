import cv2
import numpy as np
import os

def get_ref_coins():
    ref_coins = {}
    for file in os.listdir('ref_coins/'):
        if file.endswith('.jpg'):
            coin_name = file[:-8]  # Remove 'cara.jpg' or 'coroa.jpg'
            if coin_name not in ref_coins:
                ref_coins[coin_name] = []
            img = cv2.imread(os.path.join('coins/', file), cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0
            img = cv2.resize(img, (500, 500))
            ref_coins[coin_name].append(img)
    return ref_coins

def extract_coin_features(image):

    # Image feature extraction steps here
    # Get magnitude and direction of gradients
    grad = cv2.Sobel((image * 255).astype(np.uint8), cv2.CV_64F, 1, 1, ksize=5)
    mag = cv2.magnitude(grad, grad)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    dir = cv2.phase(grad, grad, angleInDegrees=True)
    
    # Rotate the magnitude to the maximum gradient direction
    mean_dir = np.mean(dir[mag > np.percentile(mag, 90)])
    rot_mat = cv2.getRotationMatrix2D(( image.shape[1] / 2, image.shape[0] / 2), mean_dir, 1)
    mag = cv2.warpAffine(mag, rot_mat, (image.shape[1], image.shape[0]))
    
    return cv2.convertScaleAbs(mag), cv2.convertScaleAbs(dir)

def process_coin_image(img):
    
    # Image processing steps here
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    img = img ** 0.7
    img = 0.9*img + 0.3*(img - blurred)

    return img

def separete_coins(img):
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
        coin_img = cv2.resize(coin_img, (500, 500))
        coins.append(coin_img)

    return coins

def align_coins(test_coin, ref_coin):
    # Coin alignment steps here
    test_coin_mag, test_coin_dir = test_coin
    ref_coin_mag, ref_coin_dir = ref_coin

    mean_dir_ref = np.mean(ref_coin_dir[ref_coin_mag > np.percentile(ref_coin_mag, 90)])
    mean_dir_test = np.mean(test_coin_dir[test_coin_mag > np.percentile(test_coin_mag, 90)])
    mean_dir = mean_dir_ref - mean_dir_test
    rot_mat = cv2.getRotationMatrix2D(( test_coin_mag.shape[1] / 2, test_coin_mag.shape[0] / 2), mean_dir, 1)
    aligned_mag = cv2.warpAffine(test_coin_mag, rot_mat, (test_coin_mag.shape[1], test_coin_mag.shape[0]))

    return aligned_mag, test_coin_dir

def main():
    ref_coins = get_ref_coins()

    ref_features = {coin_name: {} for coin_name in ref_coins.keys()}
    for coin_name, images in ref_coins.items():
        for idx, img in enumerate(images):
            processed_img = process_coin_image(img)
            mag, dir = extract_coin_features(processed_img)
            ref_features[coin_name][idx] = (mag.astype(np.uint8), dir.astype(np.uint8))
            #cv2.imshow(f'{coin_name} - Image {idx+1}', mag)
            #cv2.imshow(f'Ref {coin_name} Image {idx+1}', img)
            #cv2.imshow(f'Ref {coin_name} Image {idx+1} - Processed', processed_img)

    test_img = cv2.imread('test_images/blackback.jpg', cv2.IMREAD_GRAYSCALE)
    #test_img = cv2.resize(test_img, (500, 500))
    
    coins = separete_coins(test_img)

    for idx, coin in enumerate(coins):
        processed_coin = process_coin_image(coin.astype(np.float32) / 255.0)
        test_coin_features = extract_coin_features(processed_coin)

        # Compare to reference features
        maior = (0, ('', -1))  # (correlation, (coin_name, ref_idx))
        for ref_coin_name, ref_coin_features in ref_features.items():
            for ref_idx, ref_coin_feature in ref_coin_features.items():                    
                # Simple comparison
                test_coin_features = align_coins(test_coin_features, ref_coin_feature)
                correlation = np.corrcoef(test_coin_features[0].flatten(), ref_coin_feature[0].flatten())[0, 1]
                
                if correlation > maior[0]:
                    maior = (correlation, (ref_coin_name, ref_idx))

                print(f'Coin {idx+1} vs {ref_coin_name} Image {ref_idx+1}: Correlation = {correlation:.4f}')
        
        print(f'Best match for Coin {idx+1}: {maior[1][0]} Image {maior[1][1]+1} with Correlation = {maior[0]:.4f}')
        cv2.imshow(f'Coin {idx+1} vs {maior[1][0]} Image {maior[1][1]+1}', test_coin_features[0])


if __name__ == "__main__":
    main()
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            break
        if key == ord("q"):
            os._exit(0)
    cv2.destroyAllWindows()
