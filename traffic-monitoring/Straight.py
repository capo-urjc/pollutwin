import cv2
import numpy as np


class Straight:
    def __init__(self, pt1=None, pt2=None, m=None, n=None):
        if pt1 is not None and pt2 is not None:
            self.__m, self.__n = self.__create_with_points(pt1, pt2)
        elif m is not None and n is not None:
            self.__m = m
            self.__n = n
        else:
            raise Exception("Parameters of constructor not correct.")

    def __create_with_points(self, pt1: list, pt2: list) -> tuple:
        pt1 = self.__to_homogeneous(pt1)
        pt2 = self.__to_homogeneous(pt2)

        r: np.array = np.cross(pt1, pt2)
        r = self.__normalize(r)

        m, n = self.__pte_ord(r)

        return m, n

    def is_above(self, punto: tuple) -> bool:
        y: float = self.__m * punto[0] + self.__n

        return punto[1] < y

    def __normalize(self, r: np.array) -> np.array:
        return r / r[2]

    def __pte_ord(self, r: np.array) -> np.array:
        return -r[0] / r[1], -r[2] / r[1]

    def __to_homogeneous(self, pt1: list) -> list:
        pt1.append(1)
        return pt1

    def paint(self, image: np.ndarray, color: tuple = (0, 255, 0), thickness: int = 2) -> None:
        x1 = 0
        y1 = int(self.__m * x1 + self.__n)
        x2 = image.shape[1]
        y2 = int(self.__m * x2 + self.__n)

        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

