import cv2
import numpy as np
import math
from scipy import ndimage



def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result, rot_mat

def detect_angle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("reee", hsv)
    lower_blue = np.array([140, 30, 200])
    upper_blue = np.array([160, 170, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #cv2.imshow("aaamask", mask)
    cntr, hir = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(frame, cntr, -1, (0, 255, 0), 3)
    ps = []
    a = True
    for i in cntr:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            ps.append((cx, cy))


    print(np.rad2deg(np.arctan2(ps[1][1]-ps[0][1], ps[1][0]-ps[0][0])))
    return np.rad2deg(np.arctan2(ps[1][1]-ps[0][1], ps[1][0]-ps[0][0])) + 90
    #cv2.imshow("aaaaa", frame)

def detect_lane(frame):
    detect_aruco(frame)
    edges = detect_edges(frame)
    # cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(edges)
    lane_lines, _, _ = average_slope_intercept(frame, line_segments)

    return lane_lines


def detect_edges(frame, return_mask=False, show_image=False, lower_thresh=[0, 0, 230], upper_thresh=[180, 255, 255]):
    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # show_image("hsv", hsv)
    lower_blue = np.array(lower_thresh)
    upper_blue = np.array(upper_thresh)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #cv2.imshow("mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)
    if not return_mask:
        return edges
    else:
        return edges, mask


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 0.55),
        (width, height * 0.55),
        (width, 0.9 * height),
        (0, 0.9 * height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    # cv2.imshow("ROI", mask)
    cropped_edges = cv2.bitwise_and(edges, mask)

    return cropped_edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=8, maxLineGap=4)

    # print('line segments', line_segments)
    return line_segments


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        # print('No line_segment segments detected')
        return [], [], []

    height, width, _ = frame.shape
    # print(height, width)
    left_fit = []
    right_fit = []

    boundary = 0.4
    # left lane line segment should be on left 2/3 of the screen
    left_region_boundary = width * boundary
    # right lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * (1 - boundary)

    # print(left_region_boundary, right_region_boundary)

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # print('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue

            # print(x1, y1, x2, y2)
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            # if slope < 0:
            if x1 < left_region_boundary and x2 < left_region_boundary:
                left_fit.append((slope, intercept))
            elif x1 > right_region_boundary and x2 > right_region_boundary:
                right_fit.append((slope, intercept))

    # print(left_fit)
    # print(right_fit)

    # print(left_fit)
    # print("====")
    # print(right_fit)

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    # print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
    left_lines = []
    for left in left_fit:
        left_lines.append(make_points(frame, left))

    right_lines = []
    for right in right_fit:
        right_lines.append(make_points(frame, right))

    return lane_lines, left_lines, right_lines


# def average_slope_intercept(frame, line_segments):
#     """
#     This function combines line segments into one or two lane lines
#     If all line slopes are < 0: then we only have detected left lane
#     If all line slopes are > 0: then we only have detected right lane
#     """
#     lane_lines = []
#     if line_segments is None:
#         print('No line_segment segments detected')
#         return lane_lines

#     height, width, _ = frame.shape
#     print(height, width)
#     left_fit = []
#     right_fit = []

#     boundary = 2/5
#     left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
#     right_region_boundary = width * boundary # right lane line segment should be on left 2/3 of the screen

#     for line_segment in line_segments:
#         for x1, y1, x2, y2 in line_segment:
#             if x1 == x2:
#                 print('skipping vertical line segment (slope=inf): %s' % line_segment)
#                 continue
#             fit = np.polyfit((x1, x2), (y1, y2), 1)
#             slope = fit[0]
#             intercept = fit[1]
#             if slope < 0:
#                 if x1 < left_region_boundary and x2 < left_region_boundary:
#                     left_fit.append((slope, intercept))
#             else:
#                 if x1 > right_region_boundary and x2 > right_region_boundary:
#                     right_fit.append((slope, intercept))

#     left_fit_average = np.average(left_fit, axis=0)
#     if len(left_fit) > 0:
#         lane_lines.append(make_points(frame, left_fit_average))

#     right_fit_average = np.average(right_fit, axis=0)
#     if len(right_fit) > 0:
#         lane_lines.append(make_points(frame, right_fit_average))

#     print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

#     return lane_lines


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    if abs(slope - 0) < 0.001:
        x1 = 2 * width
        x2 = 2 * width
    else:
        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2),
                         line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image
