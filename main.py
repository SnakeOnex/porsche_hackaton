import cv2
import numpy as np
from lane_detection import detect_lane, detect_edges, region_of_interest, detect_line_segments, average_slope_intercept
from heading import *
import math


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2),
                         line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image


# edges = detect_edges(frame)
# cv2.imshow("edges", edges)
# cv2.waitKey(0)


# cropped = region_of_interest(edges)
# cv2.imshow("edges", cropped)
# cv2.waitKey(0)


# cv2.destroyAllWindows()
frame = cv2.imread('CarlaData/images/06.png')
frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

edges = detect_edges(frame)
cropped_edges = region_of_interest(edges)
line_segments = detect_line_segments(cropped_edges)
# print(line_segments)
print('here')
lane_lines = average_slope_intercept(frame, line_segments)
print(lane_lines)

cv2.imshow("heading", cropped_edges)
cv2.waitKey(0)

lane_lines = detect_lane(frame)
print(len(lane_lines))
lane_lines_image = display_lines(frame, lane_lines)
heading_image = display_heading_line(
    frame, compute_steering_angle(frame, lane_lines))
cv2.imshow("lane lines", lane_lines_image)
cv2.imshow("heading", heading_image)
cv2.waitKey(0)
