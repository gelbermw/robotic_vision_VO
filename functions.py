import numpy as np
import math
import cv2 as cv
import random
import os
import pickle


def distance_to_cam(z_pt, x_pt, y_pt):
    return np.sqrt((x_pt*x_pt) + (y_pt*y_pt) + (z_pt*z_pt))


def distance_between_pts(z1_pt, x1_pt, y1_pt, z2_pt, x2_pt, y2_pt):
    return np.sqrt((x2_pt-x1_pt)**2 + (y2_pt-y1_pt)**2 + (z2_pt-z1_pt)**2)


def camera_angle(a, b, c):    # angle alpha based on triangle
    numer = b**2 + c**2 - a**2
    denom = 2 * b * c
    alpha = math.acos(numer / denom)
    return alpha


def point1_angle(a, b, c):    # angle beta based on triangle
    numer = a**2 + c**2 - b**2
    denom = 2 * a * c
    beta = math.acos(numer / denom)
    return beta


def point2_angle(a, b, c):    # angle beta based on triangle
    numer = a**2 + b**2 - c**2
    denom = 2 * a * b
    gamma = math.acos(numer / denom)
    return gamma


def transpose_array(arr):
    return [[arr[j][i] for j in range(len(arr))] for i in range(len(arr[0]))]


def p_r_y(r):
    r = np.asarray(r)
    pitch = 0 # beta
    roll = 0 # gamma
    yaw = 0 # alpha
    print(r)
    pitch = math.asin(-1*r[2][0])
    yaw = math.acos((r[0][0])/(math.cos(pitch)))
    roll = math.acos((r[2][2])/(math.cos(pitch)))
    pitch = np.rad2deg(pitch)
    yaw = np.rad2deg(yaw)
    roll = np.rad2deg(roll)
    return pitch, roll, yaw     # return degrees


def rotationMatrixToEulerAngles(R):
    # assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def draw_delta(img, new, old, offset=(0, 0), color=(0, 1, 0), scale=5, thickness=1, double_box=False, fade_factor=0.5):
    new = np.int32(np.array(new) * scale)
    offset = np.array(offset)
    faded_color = tuple(np.array(color) * fade_factor)

    img = cv.rectangle(img,
                       tuple(new), tuple(new + scale),
                       color, thickness)

    if old is not None:
        old = np.int32((np.array(old) + offset) * scale)
        img = cv.line(img,
                      tuple(new + int(scale / 2)),
                      tuple(old + int(scale / 2)),
                      faded_color, thickness=thickness)

        if double_box:
            img = cv.rectangle(img,
                               tuple(old), tuple(old + scale),
                               color, thickness)

    return img


def render_keypoints(img, points0, points1, scale=5, offset=(0, 0), double_box=False):
    img = cv.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv.INTER_AREA)

    img = np.float32(img/255)
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # for feat in self.frames[-1].features:
    #     draw_delta(img, feat, None, offset, color=(0, 0.5, 0.5), double_box=double_box)
    if points0 is not None:
        for point, p2 in zip(points0, points1):
            draw_delta(img, np.array(point), np.array(p2), offset, color=(0, 1, 0), scale=scale, double_box=double_box)


    return img