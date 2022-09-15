import numpy as np
import math


def distanceManhattan(X, Y):
    result = np.subtract(X, Y)
    result = np.absolute(result)
    value = result.sum()
    return value


def distanceEukliedian(X, Y):
    result = np.subtract(X, Y)
    result = np.power(result, 2)
    value = result.sum()
    value = math.sqrt(value)
    return value


def distanceMaksimum(X, Y):
    result = np.subtract(X, Y)
    result = np.absolute(result)
    max_value = np.max(result)
    return max_value


def distanceMinkowski(X, Y, p=2):
    result = np.subtract(X, Y)
    result = np.absolute(result)
    result = np.power(result, p)
    value = result.sum()
    value = pow(value, 1/p)
    return value

def distanceChi2(X, Y):
    for elem in list(zip(X, Y)):
        to_sum = []
        if (elem[0] + elem[1]) == 0:
            to_sum.append(0)
        else:
            part = ((elem[0]-elem[1]) ** 2) / (elem[0] + elem[1])
            to_sum.append(part)
    chi = 0.5 * np.sum(to_sum)
    return chi


def calculateDistanceManhattan(filepath, inputimagedata):
    data = np.load(filepath)
    result = distanceManhattan(data, inputimagedata)
    return result


def calculateDistanceEukliedian(filepath, inputimagedata):
    data = np.load(filepath)
    result = distanceEukliedian(data, inputimagedata)
    return result


def calculateDistanceMaksimum(filepath, inputimagedata):
    data = np.load(filepath)
    result = distanceMaksimum(data, inputimagedata)
    return result


def calculateDistanceMinkowski(filepath, inputimagedata):
    data = np.load(filepath)
    result = distanceMinkowski(data, inputimagedata, 2)
    return result


def calculateDistance(filepath, inputimagedata, func, separator=None):
    data = np.load(filepath)
    if separator:
        data = data[separator[0]: separator[1]]
    result = func(data, inputimagedata)
    return result
