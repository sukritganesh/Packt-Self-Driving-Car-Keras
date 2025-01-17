import cv2
import numpy as np
import matplotlib.pyplot as plt

# generate coordinates to define line
# lines start at bottom of image, go 2/5 way up
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))      # line goes 2/5 up the image
    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# left and right lane lines should be created
# using averages of detected lines
# right line segments have positive slope, left have negative slope
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        # right lines have positive slope, left lines have negative slope
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # generate coordinates to define line
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

def canny(image):
    # Step 1: Convert Image to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Filter out noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Edge detection with Canny
    canny = cv2.Canny(blur, 50, 150)

    return canny

# add lines to image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if (lines is not None):
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)       # Blue Lines, thickness=10

    return line_image

# return enclosed region of field of view
# create polygon mask
def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(200, height), (1100, height), (500, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, gamma=1)
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, gamma=1)
    cv2.imshow('result', combo_image)
    # display image, unless we press 'q'!
    if (cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
