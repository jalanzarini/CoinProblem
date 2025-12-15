import cv2
import numpy as np
from perspective import correct_perspective

def analyze_coin_color(roi):
    """
    Returns 'silver', 'gold', 'copper', or 'bimetallic'
    """
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Calculate average saturation and hue of the center
    height, width = roi.shape[:2]
    center_region = hsv[height//4:3*height//4, width//4:3*width//4]
    avg_sat = np.mean(center_region[:, :, 1])
    avg_hue = np.mean(center_region[:, :, 0]) # OpenCV Hue is 0-179
    
    # Heuristic Thresholds (You may need to tune these based on lighting)
    SAT_THRESH = 40  # Below this is grayscale/silver
    
    if avg_sat < SAT_THRESH:
        return "silver"
    
    # If it has color, check for 1 Real (Bimetallic)
    # Check the corners of the ROI (the outer ring of the coin)
    # A simple mask approach is better, but for simplicity:
    outer_region = hsv[0:5, 0:5] # Top left corner sample
    outer_sat = np.mean(outer_region[:, :, 1])
    
    # 1 Real has Silver center (Low Sat) and Gold Ring (High Sat)
    if avg_sat < SAT_THRESH and outer_sat > SAT_THRESH:
        return "1_real"
        
    # Differentiate Copper vs Gold by Hue
    # Yellow is roughly 20-35 in OpenCV Hue
    # Red is roughly 0-10 or 170-180
    if 15 < avg_hue < 45:
        return "gold" # 25 cents
    else:
        return "copper" # 5 cents

# --- Main Pipeline ---
img = cv2.imread("test_images/perspective_4.jpg")
img = correct_perspective(img)
img = cv2.resize(img, (800, 600))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.0, 50, 
                           param1=70, param2=60, minRadius=20, maxRadius=200)

if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Find max radius to normalize sizes (assuming largest coin is present)
    max_r = np.max(circles[0, :, 2])
    
    for i in circles[0, :]:
        x, y, r = i[0], i[1], i[2]
        
        # Crop the coin
        roi = img[y-r:y+r, x-r:x+r]
        if roi.size == 0: continue
        
        color_type = analyze_coin_color(roi)
        label = "Unknown"
        
        # Classification Logic
        if color_type == "1_real":
            label = "R$ 1.00"
        elif color_type == "copper":
            label = "R$ 0.05"
        elif color_type == "gold":
            if r > max_r * 0.85: 
                label = "R$ 0.25"
            else:
                label = "R$ 0.10"
        elif color_type == "silver":
            # Distinguish 10 vs 50 by size
            # 1 real is close to the max size, 50 cents is much smaller
            if r > max_r * 0.9: 
                label = "R$ 1.00"
            else:
                label = "R$ 0.50"
        
        # Draw
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.putText(img, label, (x-20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)