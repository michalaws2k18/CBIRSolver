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


def distanceMinkowski(X, Y, p):
    result = np.subtract(X, Y)
    result = np.absolute(result)
    result = np.power(result, p)
    value = result.sum()
    value = pow(value, 1/p)
    return value


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
