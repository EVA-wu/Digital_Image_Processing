import numpy as np
import cv2

#centralize image by formula f(x, y) * (-1)^(x + y)
def centralize(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    outputImage = np.zeros((height,width),np.int32)

    for i in xrange(height):
        for j in xrange(width):
        	if (i + j) % 2 != 0:
        	    outputImage[i][j] = -1 * inputImage[i][j]
        	else:
        	    outputImage[i][j] = inputImage[i][j]

    return outputImage

def fft2d(inputImage, flag):
    if flag == "FFT":
        return FFT(inputImage)
    elif flag == "IFFT":
        # the inputImage here is the one that has already done DFT
        width = len(inputImage[0])
        height = len(inputImage)
        for x in range(height):
            for y in range(width):
                inputImage[x][y] = inputImage[x][y] / float(width * height)
        return np.conjugate(IFFT(inputImage))

# Inverse Fast Fourier Transform
def IFFT(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    tmpImage = np.zeros((height,width),np.complex256)
    outputImage = np.zeros((height,width),np.complex256)

    for i in xrange(height):
        tmpImage[i] = np.array(ifft(inputImage[i]))

    for j in xrange(width):
        outputImage[:, j] = ifft(tmpImage[:, j])

    return outputImage

# divide into two parts
def ifft(x):
    N = len(x)
    if N <= 1:
        return x
    elif N % 2 != 0:
        return slow(x)
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        return np.concatenate([[even[k] + np.exp(-2j*np.pi*k/N)*odd[k] for k in xrange(len(even))]
             ,[even[k] - np.exp(-2j*np.pi*k/N)*odd[k] for k in xrange(len(even))]])

# calculate the sum of all rows
def islow(x):
    outputImage = np.zeros((len(x)), np.complex256)

    for i in xrange(len(x)):
        for j in xrange(len(x)):
            outputImage[i] += x[j] * (np.exp(1j*2*np.pi*(float(i*j)/len(x))))

    return outputImage

# Fast Fourier Transform
def FFT(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    tmpImage = np.zeros((height,width),np.complex256)
    outputImage = np.zeros((height,width),np.complex256)

    for i in xrange(height):
        tmpImage[i] = np.array(fft(inputImage[i]))

    for j in xrange(width):
        outputImage[:, j] = fft(tmpImage[:, j])

    return outputImage

# divide into two parts
def fft(x):
    N = len(x)
    if N <= 1:
        return x
    elif N % 2 != 0:
        return slow(x)
    else:
        even = fft(x[::2])
        odd = fft(x[1::2])
        return np.concatenate([[even[k] + np.exp(-2j*np.pi*k/N)*odd[k] for k in xrange(len(even))]
             ,[even[k] - np.exp(-2j*np.pi*k/N)*odd[k] for k in xrange(len(even))]])

# calculate
def slow(x):
    outputImage = np.zeros((len(x)), np.complex256)

    for i in xrange(len(x)):
        for j in xrange(len(x)):
            outputImage[i] += x[j] * (np.exp(-1j*2*np.pi*(float(i*j)/len(x))))

    return outputImage

# Log Transform
def logTransform(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    outputImage = np.zeros((height,width), np.float64)

    for i in range(height):
        for j in range(width):
        	if inputImage[i][j] > 0:
        	    outputImage[i][j] = np.log(inputImage[i][j])

    return outputImage

#rotate
def rotateTransform(inputImage):
    height, width = inputImage.shape
    outputImage = np.zeros((height,width), np.int32)
    for x in xrange(height):
        for y in xrange(width):
            outputImage[height - x - 1, width - y - 1] = inputImage[x, y]
    return outputImage

# scaling image
def scaling(inputImage):
    width = len(inputImage[0])
    height = len(inputImage)

    # the interval of the whole image
    mMax = np.amax(inputImage)
    mMin = np.amin(inputImage)

    scaling = np.zeros((height, width), np.int32)

    for i in range(height):
        for j in range(width):
                scaling[i, j] = int((inputImage[i, j] - mMin) * 255 / (mMax - mMin))

    return scaling

if __name__=="__main__":
    inputImage = cv2.imread("./74.png", 0)

    fftImage = centralize(inputImage)
    fftImage = fft2d(fftImage, "FFT")

    #Inverse Fourier Transform
    ifftImage = fft2d(fftImage, "IFFT")
    ifftImage = ifftImage.real
    ifftImage = centralize(ifftImage)
    ifftImage = scaling(ifftImage)
    ifftImage = rotateTransform(ifftImage)
    cv2.imwrite("InverseFastFourierTransform.png", ifftImage)

    # Fourier Transform
    fftImage = np.abs(fftImage)
    fftImage = logTransform(fftImage)
    fftImage = scaling(fftImage)
    cv2.imwrite("FastFourierTransform.png", fftImage)