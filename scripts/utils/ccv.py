import numpy as np
from PIL import Image
import cv2
from pprint import pprint


def is_adjacent(x1, y1, x2, y2):
    ''' Returns true if (x1, y1) is adjacent to (x2, y2), and false otherwise '''
    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)
    return not (x_diff == 1 and y_diff == 1) and (x_diff <= 1 and y_diff <= 1)


def find_max_cliques(arr, n):
    ''' Returns a 2*n dimensional vector
    v_i, v_{i+1} describes the number of coherent and incoherent pixels respectively a given color
    '''
    tau = int(arr.shape[0] * arr.shape[1] *
              0.01)  # Classify as coherent is area is >= 1%
    ccv = [0 for i in range(n**3 * 2)]
    unique = np.unique(arr)
    for u in unique:
        x, y = np.where(arr == u)
        groups = []
        coherent = 0
        incoherent = 0

        for i in range(len(x)):
            found_group = False
            for group in groups:
                if found_group:
                    break

                for coord in group:
                    xj, yj = coord
                    if is_adjacent(x[i], y[i], xj, yj):
                        found_group = True
                        group[(x[i], y[i])] = 1
                        break
            if not found_group:
                groups.append({(x[i], y[i]): 1})

        for group in groups:
            num_pixels = len(group)
            if num_pixels >= tau:
                coherent += num_pixels
            else:
                incoherent += num_pixels

        assert(coherent + incoherent == len(x))

        index = int(u)
        ccv[index*2] = coherent
        ccv[index*2+1] = incoherent

    return ccv


def get_ccv(img, n):
    # Blur pixel slightly using avg pooling with 3x3 kernel
    print(np.shape(img))
    blur_img = cv2.blur(img, (3, 3))
    print(np.shape(blur_img))
    blur_flat = blur_img.reshape(120*80, 3)

    # Discretize colors
    hist, edges = np.histogramdd(blur_flat, bins=n)

    graph = np.zeros((img.shape[0], img.shape[1]))
    result = np.zeros(blur_img.shape)

    total = 0
    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, n):
                rgb_val = [edges[0][i+1], edges[1][j+1], edges[2][k+1]]
                previous_edge = [edges[0][i], edges[1][j], edges[2][k]]
                coords = ((blur_img <= rgb_val) & (
                    blur_img >= previous_edge)).all(axis=2)
                result[coords] = rgb_val
                graph[coords] = i + j * n + k * n**2

    result = result.astype(int)
    return find_max_cliques(graph, n)


def extract_features(image, n):
    return get_ccv(image, n)


def extract_CCV(image_path, n):
    img = np.array(Image.open(image_path))
    ccv_feature = get_ccv(img, n)
    return ccv_feature


def extractAndSaveCCV(image_path, n):
    ccv_features = extract_CCV(image_path, n)
    saveas = image_path[:image_path.rfind(".jpg")] + ".npy"
    np.save(saveas, ccv_features)
