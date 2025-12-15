import cv2
import numpy as np
import os
import math

FILENAME = "test_images/perspective_4.jpg"

def correct_perspective(in_img):
  #Convert to grayscale
  out_img = in_img.copy()
  if in_img.shape[0] > 1000 or in_img.shape[1] > 1000:
    in_img = cv2.resize(in_img, (0, 0), fx=.3, fy=.3)
    out_img = cv2.resize(out_img, (0, 0), fx=.3, fy=.3)
  if len(in_img.shape) == 3:
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
  else:
    in_img = in_img.copy()

  # Find coins
  man_img = in_img.copy()
  man_img = cv2.GaussianBlur(man_img, (5, 5), 0)
  man_img = cv2.adaptiveThreshold(man_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
  man_img = 255-man_img
  cv2.imshow('ADAPTATIVE THRESH', man_img)    

  # Remove noise
  man_img = cv2.morphologyEx(man_img, cv2.MORPH_OPEN, (3, 3))
  cv2.imshow('OPEN IMAGE', man_img)

  # Fill insides of coins
  man_img = man_img.copy()
  for i in range(50):
    man_img = cv2.GaussianBlur(man_img, (5, 5), 0)
  
  # Normalize
  man_img = cv2.normalize(man_img, None, 0, 255, cv2.NORM_MINMAX)
  cv2.imshow('OPEN IMAGE BLUR NORM', man_img)

  # Thresh
  ret, man_img = cv2.threshold(man_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  cv2.imshow('OTSU_IMG', man_img)

  #Find coins ellipses
  contours, hierarchy = cv2.findContours(man_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  mask = 255*np.zeros(man_img.shape[:2], dtype=np.uint8) 
  for contour in contours:
    if(len(contour) >= 5):
      ellipse = cv2.fitEllipse(contour)
      cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
  cv2.imshow("MASK", mask)
  in_img = in_img*mask
  cv2.imshow("MASK_in", in_img)

  #Find any ellipse to transform
  ellipses = []

  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
    if(len(contour) >= 5):
      ellipse = cv2.fitEllipse(contour)
      ellipses.append(ellipse)

  major_axis_size = ellipses[0][1][0]
  minor_axis_size = ellipses[0][1][1]
  angle = ellipses[0][2]

  points_adjusted = np.float32([
    [250, 10],
    [490, 250],
    [250, 490],
    [10, 250],
  ])

  images = []
  for index, ellipse in enumerate(ellipses):
    x = ellipse[0][0]
    y = ellipse[0][1]
    major_axis_size = ellipse[1][0]
    minor_axis_size = ellipse[1][1]
    angle = ellipse[2]
    points = np.float32([
      [x+major_axis_size*math.cos(angle)/2, y+major_axis_size*math.sin(angle)/2],
      [x-minor_axis_size*math.sin(angle)/2, y+minor_axis_size*math.cos(angle)/2],
      [x-major_axis_size*math.cos(angle)/2, y-major_axis_size*math.sin(angle)/2],
      [x+minor_axis_size*math.sin(angle)/2, y-minor_axis_size*math.cos(angle)/2],
    ])
    
    # cv2.circle(in_img, (int(x), int(y)), 2, (0, 255, 0), -1)
    # for point in points:
    #   cv2.circle(in_img, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    # cv2.imshow("Points", in_img)
    # cv2.waitKey(0)

    transform = cv2.getPerspectiveTransform(points, points_adjusted) 
    img_transformed = cv2.warpPerspective(in_img,transform,(501, 501))
    cv2.imshow('Transformed'+str(index), img_transformed)
    images.append(img_transformed)
  return images, ellipses, out_img

if __name__ == "__main__":
  in_img = cv2.imread(FILENAME)

  correct_perspective(in_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
