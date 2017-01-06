import numpy as np

def Fourier_log(img):
    height = len(img)
    width = len(img[0])

    Fourier_log_img = np.zeros((height,width), np.float64)

    for u in xrange(height):
        for v in xrange(width):
            if img[u, v] > 0:
                Fourier_log_img[u, v] = np.log(img[u, v])

    return Fourier_log_img

def Fourier_scaling(img, method="linear"):
    height = len(img)
    width = len(img[0])
    res_img = np.zeros((height,width),np.int32)

    max_pixel = np.amax(img)
    min_pixel = np.amin(img)
    contantC = 255 / np.log10(256)

    for u in xrange(height):
        for v in xrange(width):
            if method == "linear":
                res_img[u, v] = int((img[u, v] - min_pixel) / (max_pixel - min_pixel) * 255)
            elif method == "log":
                res_img[u, v] = int(contantC * np.log10(1 + np.abs(float(255*img[u, v]) / max_pixel)))

    return res_img

def centralize(img):
    height = len(img)
    width = len(img[0])

    res_img = np.zeros((height,width),np.int32)

    for x in xrange(height):
        for y in xrange(width):
            if (x + y) % 2 != 0:
                res_img[x, y] = -1 * img[x, y]
            else:
                res_img[x, y] = img[x,y]

    return res_img

def diolog_transform(img):
    height, width = img.shape
    diolog_transform_img = np.zeros((height,width), np.int32)
    for x in xrange(height):
        for y in xrange(width):
            diolog_transform_img[height - x - 1, width - y - 1] = img[x, y]
    return diolog_transform_img

def gama_correction(img, scale):
    table = [0 for i in range(256)]
    for i in range(256):
        val = pow(float(i)/255.0 ,scale) * 255.0
        if val>255:
                val = 255
        elif val<0:
                val = 0
        table[i]= val

    height = len(img)
    width  = len(img[0])

    new_img = np.zeros((height,width), np.uint8)

    for i in range(height):
        for j in range(width):
            if new_img[i, j] < 255:
                new_img[i, j] = int(table[img[i,j]])
            else:
                new_img[i, j] = int(table[255])

    return new_img
