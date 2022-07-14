from audioop import avg
from inspect import Parameter
from turtle import right
import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinate(image, line_prms):
    slope, intercept = line_prms
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2= int((y2 - intercept)/slope)
    return np.array([x1,y1, x2, y2])

def avg_slope_intercept(image, lines):
    # left&right lines fitted
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0) 
    right_fit_avg = np.average(right_fit, axis=0)
    # print(left_fit_avg,'left side print')
    # print(right_fit_avg, 'right side print')
    left_line = make_coordinate(image, left_fit_avg)
    right_line = make_coordinate(image, right_fit_avg)
    return np.array([left_line,right_line])


def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    return cv2.Canny(blur,50,150)

def triangle_shaped_image(image):
    height = image.shape[0]
    polygones = np.array([
        # [(350,height),(950,height),(550,250)]
        [(200,height),(1100,height),(550,250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygones,255)
    # v2 = cv2.fillPoly(image,polygones,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image =  np.zeros_like(image)
    if lines is not None:
        for  x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1), (x2,y2),(255,0,0),10)
    return line_image

# for image start
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = triangle_shaped_image(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

avg_lines = avg_slope_intercept(lane_image, lines)
# black display
line_image = display_lines(lane_image, avg_lines)
# on screen display
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('Ai Lane Detection', combo_image)
cv2.waitKey(0)
# for image end

# plt.imshow( canny)
# plt.show()

# cap = cv2.VideoCapture('test2.mp4')
# while(cap.isOpened()):
#     _, frame = cap.read()
#     canny_image = canny(frame)
#     cropped_image = triangle_shaped_image(canny_image)
#     lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

#     avg_lines = avg_slope_intercept(frame, lines)
#     # black display
#     line_image = display_lines(frame, avg_lines)
#     # on screen display
#     combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
#     cv2.imshow('Ai Lane Detection', combo_image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()