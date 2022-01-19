def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    maxV = max(r, g, b)
    minV = min(r, g, b)
    df = maxV-minV
    if df == 0:
        h = 0
    elif maxV == r:
        h = (60 * (((g-b)/df) % 6))
    elif maxV == g:
        h = 60 * (((b-r)/df) + 2)
    elif maxV == b:
        h = 60 * (((r-g)/df) + 4)
    if maxV == 0:
        s = 0
    else:
        s = (df/maxV)*100
    v = maxV*100
    return h, s, v


# print(rgb_to_hsv(255, 255, 255))
# print(rgb_to_hsv(0, 215, 0))
