import Image

import math

#intervals Sampling(both down scale and up scale)
# input : an Image object and a size tuple
# output : an Image object
# function : scale the source image to a new size uisng interval sampling
def scale1(img, size):
    # note the color plane
    newIm = Image.new("P", (size[0], size[1]))

    #Scaling Rate for x and y
    m = float(size[0]) / img.size[0]
    n = float(size[1]) / img.size[1]

    #algorithmn to get pixels from the source image
    for i in range(size[0]):
        for j in range(size[1]):
            x = int(i/m)
            y = int(j/n)
            newIm.putpixel((i,j), img.getpixel((x,y)))

    return newIm

#Local mean sampling
# input : an Image object and a size tuple
# output : an Image object
# function : scale the source image to a new size uisng Local mean sampling
#            it will get the average of serveral points in the image
def scale2(img, size):
    # note the color plane
    newIm = Image.new("P", (size[0], size[1]))

    #Scaling Rate for x and y
    m = float(size[0]) / img.size[0]
    n = float(size[1]) / img.size[1]

    #algorithmn to get pixels from the source image
    for i in range(size[0] - 1):
        for j in range(size[1] -1):
            x1 = int(i/m)
            x2 = int((i+1)/m)
            y1 = int(j/n)
            y2 = int((j+1)/n)
            theSum = 0

            for k in (x1, x2):
                for l in (y1, y2):
                    theSum += img.getpixel((k,l))
            # get the number of pixels
            num = (x2 - x1)*(y2 - y1)
            newIm.putpixel((i,j), int(float(theSum) / num))

    return newIm

def quantize(img, level):
    newIm = Image.new("P", (img.size[0], img.size[1]))
    segement = 256 / level
    level = 255.0 / float(level - 1)
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            temp = int(img.getpixel((i,j)) / segement)
            newIm.putpixel((i,j), temp * level)
    return newIm


if __name__ == "__main__":
    im = Image.open("./74.png")

    #process with algothmn 1
    #question 1 scale
    result1 = scale1(im, (192,128))
    result2 = scale1(im, (96,64))
    result3 = scale1(im, (48,32))
    result4 = scale1(im, (24,16))
    result5 = scale1(im, (12,8))

    result6 = scale1(im, (300,200))
    result7 = scale1(im, (450,300))
    result8 = scale1(im, (500,200))

    # # save result
    result1.save("scale_192*168.png", "png")
    result2.save("scale_96*64.png", "png")
    result3.save("scale_48*32.png", "png")
    result4.save("scale_24*16.png", "png")
    result5.save("scale_12*8.png", "png")
    result6.save("scale_300*200.png", "png")
    result7.save("scale_450*300.png", "png")
    result8.save("scale_200*200.png", "png")


    #question 2 quntinization
    result1 = quantize(im, 128)
    result2 = quantize(im, 32)
    result3 = quantize(im, 8)
    result4 = quantize(im, 4)
    result5 = quantize(im, 2)

    #save result
    result1.save("quantization128.png", "png")
    result2.save("quantization32.png", "png")
    result3.save("quantization8.png", "png")
    result4.save("quantization4.png", "png")
    result5.save("quantization2.png", "png")
