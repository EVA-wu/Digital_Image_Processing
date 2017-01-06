import numpy as np
import cv2
from basic import *

def fft2d(img, flags):
    if flags == 1:
        return Ffft2d(img)
    elif flags == -1:
        res = np.conjugate(img)
        return np.conjugate(Ffft2d(img))

def Ffft2d(img):
    height = len(img)
    width = len(img[0])

    Fourier_img0 = np.zeros((height,width),np.complex256)

    for u in xrange(height):
        Fourier_img0[u] = np.array(fft(img[u]))

    Fourier_img = np.zeros((height,width),np.complex256)
    for v in xrange(width):
        Fourier_img[:,v] = fft(Fourier_img0[:,v])

    return Fourier_img

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    elif N % 2 != 0:
        return DFT_slow(x)
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        return np.concatenate([[even[k] + np.exp(-2j*np.pi*k/N)*odd[k] for k in xrange(len(even))]
             ,[even[k] - np.exp(-2j*np.pi*k/N)*odd[k] for k in xrange(len(even))]])

def DFT_slow(x):
    Fourier_img = np.zeros((len(x)), np.complex256)

    for i in xrange(len(x)):
        for j in xrange(len(x)):
            Fourier_img[i] += x[j] * (np.exp(-1j*2*np.pi*(float(i*j)/len(x))))

    return Fourier_img

def Fourier_Spectrum():
    img = cv2.imread("./74.png", 0)
    img = centralize(img)
    Fourier_img = fft2d(img, 1)
    Fourier_img = np.abs(Fourier_img)
    Fourier_img = Fourier_log(Fourier_img)
    Fourier_img = Fourier_scaling(Fourier_img, "linear")
    cv2.imwrite("FFourier.png", Fourier_img)

def Fourier_Inverse():
    img = cv2.imread("./74.png", 0)
    Fourier_img = fft2d(img, 1)
    origin_img = fft2d(Fourier_img, -1).real
    origin_img = Fourier_scaling(origin_img, "linear")
    origin_img = diolog_transform(origin_img)
    cv2.imwrite("FInverseFourier.png", origin_img)

def main():
    Fourier_Spectrum()
    Fourier_Inverse()

if __name__ == "__main__":
    main()
