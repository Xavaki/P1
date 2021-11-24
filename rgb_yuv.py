import sys
import math
import numpy as np


def rgb_2_yuv(rgb):

    if (not all([(x >= 0 and x <= 255) for x in rgb])) or (len(rgb) != 3):
        print("Please enter a valid value")
        return
    else:
        # r = rgb[0]
        # g = rgb[1]
        # b = rgb[2]

        rgb = np.append(np.array(rgb), 1)

        mat = np.array([
            [0.257, 0.504, 0.098, 16],
            [-0.148, -0.291, 0.439, 128],
            [0.439, -0.368, -0.071, 128]
        ])

        return list(np.round(mat.dot(rgb)).astype(int))


def yuv_2_rgb(yuv):

    if (not all([(x >= 0 and x <= 255) for x in yuv])) or (len(yuv) != 3):
        print("Please enter a valid value")
        return
    else:
        yuv = np.array(yuv) - np.array([16, 128, 128])
        mat = np.array([
            [1.164, 0, 1.596],
            [1.164, -0.392, -0.813],
            [1.164, 2.017, 0],
        ])
        return list(np.round(mat.dot(yuv)).astype(int))


if __name__ == '__main__':

    function = None
    params = None

    if (len(sys.argv) == 3):
        string_vals = sys.argv[2].split(",")
        if len(string_vals) == 3:
            function = sys.argv[1]
            params = [int(p) for p in string_vals]
    elif (len(sys.argv) == 5):
        function = sys.argv[1]
        params = [int(p) for p in sys.argv[2:]]

    if function == "rgb2yuv":
        print(rgb_2_yuv(params))
        exit()
    elif function == "yuv2rgb":
        print(yuv_2_rgb(params))
        exit()
    else:
        pass

    print("Incorrect input. Valid arguments are: ")
    print("rgb2yuv r g b")
    print("rgb2yuv r,g,b")
    print("yuv2rgb y u v")
    print("yuv2yrgb y,u,v")

    exit()
