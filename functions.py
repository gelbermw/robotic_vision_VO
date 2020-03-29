import numpy as np
import math
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
