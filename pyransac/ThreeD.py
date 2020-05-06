"""2D line module.

This module contains the model for a 2-dimensional line.
"""

# Standard library imports
from dataclasses import dataclass
import math
from typing import List
from numpy import linalg
import numpy as np

# Local application imports
from pyransac.base import Model
import transform as tr


@dataclass(order=True)
class ThreeDPoint:
    """2-dimensional point class.

    This is a simple class to contain Cartesian coordinates of 2D point.
    """
    point_1: [float, float, float]  # pylint: disable=invalid-name
    """x coordinate of point."""
    point_2: [float, float, float]  # pylint: disable=invalid-name
    """y coordinate of point"""


class ThreeD(Model):
    """Model for a 3-dimensional vector.

    """
    def __init__(self, rotate=None, translate=None):
        self._rotate = rotate
        self._translate = translate

    @property
    def rotate(self):
        """Gets the rotation matrix of the model.

        :return: rotation (None if model not made).
        """
        return self._rotate

    @property
    def translate(self):
        """Gets the translation vector of the model.

        :return: translation vector (None if model not made).
        """
        return self._translate

    def make_model(self, points: List[ThreeDPoint]) -> None:
        # print("length of points is: ", len(points))
        """Makes equation for 2D line given two data points.

        Model parameters are stored internally.

        :param points: list of data points to make model
            (length must be 2)
        :return: None
        """
        # print("List contents in make model:\n", points)      # debugging
        new_points1 = []
        new_points2 = []
        for j in points:
            new_points1.append(j[0])
            new_points2.append(j[1])
        # print("new untupled results: \n", new_points1, "\n", new_points2)     # debugging

        new_points1 = np.asmatrix(np.transpose(new_points1))
        new_points2 = np.asmatrix(np.transpose(new_points2))

        if len(points) != 4:
            raise ValueError(f'Need 4 points to make pose estimate, not {len(points)}')

        # try:
        self._rotate, self._translate = tr.rigid_transform_3D(new_points1, new_points2)
        # print("Rotation: \n", self._rotate, "\n Translation: \n", self._translate)        # debugging
        # except:
        #     pass




    def calc_error(self, point: ThreeDPoint) -> float:
        """Calculate error between data point and 2D model.

        :param point: data point to calculate error with
        :return: calculated error
        """
        # calc error using ||p2 - (R*p1+T)||^2
        # print("In calc error, this is the point: ", point[1])        # debugging
        # print(self._rotate.dot(point[0]))
        error = self._rotate.dot(point[0]) + self._translate
        # print("R*p1+T = ", error)
        final_error = linalg.norm(point[1] - error)
        # print("before squaring", final_error)
        return final_error**2
