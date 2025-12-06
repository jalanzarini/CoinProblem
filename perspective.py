import cv2
import numpy as np
import os
import math

FILENAME = "test_images/perspective_3.jpg"

def main():
  # READ IMAGE
  img_bgr = cv2.imread(FILENAME)
  img_bgr = cv2.resize(img_bgr, (0, 0), fx=.3, fy=.3);
  img_bgr_out = img_bgr.copy()
  img_blur_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  # Thresh
  img_blur = cv2.GaussianBlur(img_blur_gray, (5, 5), 0)
  adaptative_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
  adaptative_thresh = 255-adaptative_thresh
  cv2.imshow('ADAPTATIVE THRESH', adaptative_thresh)    

  # Gaps
  img_open = cv2.morphologyEx(adaptative_thresh, cv2.MORPH_OPEN, (3, 3))
  cv2.imshow('OPEN IMAGE', img_open)

  # Blur again
  open_img_blur = img_open.copy()
  for i in range(100):
    open_img_blur = cv2.GaussianBlur(open_img_blur, (5, 5), 0)
  
  # Normalize
  open_img_blur_norm = cv2.normalize(open_img_blur, None, 0, 255, cv2.NORM_MINMAX)
  cv2.imshow('OPEN IMAGE BLUR NORM', open_img_blur_norm)

  # Thresh
  ret, thresh_otsu = cv2.threshold(open_img_blur_norm, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  cv2.imshow('OTSU_IMG', thresh_otsu)

  contours, hierarchy = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(img_bgr_out, contours, -1, (0,0,255), 5)
  # cv2.imshow('CONTOURS', img_bgr_out)

  best = None
  best_fit = 0
  for contour in contours:
    if(len(contour) >= 5):
      ellipse = cv2.fitEllipse(contour)
      mask = 255*np.ones(img_bgr_out.shape[:2], dtype=np.uint8) 
      cv2.ellipse(mask, ellipse, (0, 255, 0), -1)
      mask = 255-mask
      img_ellipse_masked = cv2.bitwise_and(thresh_otsu, thresh_otsu, mask=mask)
      pixels_inside_ellipse = cv2.countNonZero(mask)
      one_pixels_inside_ellipse = cv2.countNonZero(img_ellipse_masked)
      fit = (one_pixels_inside_ellipse/pixels_inside_ellipse)
      if fit > best_fit:
        best_fit = fit
        best = ellipse

  print(best)
  x = best[0][0]
  y = best[0][1]
  major_axis_size = best[1][0]
  minor_axis_size = best[1][1]
  angle = best[2]

  points = np.float32([
    [x+major_axis_size*math.cos(angle)/2, y+major_axis_size*math.sin(angle)/2],
    [x-minor_axis_size*math.sin(angle)/2, y+minor_axis_size*math.cos(angle)/2],
    [x-major_axis_size*math.cos(angle)/2, y-major_axis_size*math.sin(angle)/2],
    [x+minor_axis_size*math.sin(angle)/2, y-minor_axis_size*math.cos(angle)/2],
  ])
  points_adjusted = np.float32([
    [x, y-minor_axis_size/2],
    [x+minor_axis_size/2, y],
    [x, y+minor_axis_size/2],
    [x-minor_axis_size/2, y],
  ])
  # points_adjusted = np.float32([
  #   [100, 0],
  #   [0, 100],
  #   [200, 100],
  #   [100, 200],
  # ])

  img_bgr_draw = img_bgr.copy()
  cv2.ellipse(img_bgr_draw, best, (0, 255, 0), 2)
  cv2.circle(img_bgr_draw, (int(x), int(y)), 3, (255, 0, 255), -1)
  for index, point in enumerate(points):
    cv2.circle(img_bgr_draw, np.int32(point), 3, (255, 0, 0), -1)
    cv2.imshow('Contours', img_bgr_draw)
    cv2.waitKey(0)    
  for index, point in enumerate(points_adjusted):
    cv2.circle(img_bgr_draw, np.int32(point), 3, (0, 0, 255), -1)
    cv2.imshow('Contours', img_bgr_draw)
    cv2.waitKey(0)    
  cv2.imshow('Contours', img_bgr_draw)

  transform = cv2.getPerspectiveTransform(points, points_adjusted) 
  img_transformed = cv2.warpPerspective(img_bgr_out,transform,(900, 900))
  cv2.imshow('Transformed', img_transformed)

if __name__ == "__main__":
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
