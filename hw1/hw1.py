import math
import Image

def scale(input_img, size):
    # result image
    result = Image.new("P", (size[0], size[1]))

    # scale factors
    scaleX = float(size[0])/input_img.size[0]
    scaleY = float(size[1])/input_img.size[1]

    # expand the initial image to ensure the points like (0, 0) has N4
    tmp = Image.new("P", (input_img.size[0] + 2, input_img.size[1] + 2))
    print tmp.size[0]
    print tmp.size[1]
    for i in range(tmp.size[0]):
        for j in range(tmp.size[1]):
            # print i, j # 267, 339
            if j != 0 and j != tmp.size[1] - 1 and i != 0 and i != tmp.size[0] - 1:
                # print i - 1, j - 1 265 337
                tmp.putpixel( (i,j) , input_img.getpixel( (i-1, j-1) ) )
            else:
            	    tmp.putpixel( (i,j) , 1)

    # find the position of (ri, rj) in tmp and use find its pixel in result by bi-linear interpolation
    for ri in range(result.size[0]):
        for rj in range(result.size[1]):
            # if ri or rj is equal to 0, the position will be out of range
            if  ri != 0 and rj != 0:
            	   # according to the equation
                ii = float(ri - 1) / scaleX
                jj = float(rj - 1) / scaleY

                i = int(ii)
                j = int(jj)

                u = ii - i
                v = jj - j

                i = i + 1
                j = j + 1
                # bi-linear for interpolation
                result.putpixel( (ri, rj) , (1 - u) * (1 - v) * tmp.getpixel((i,j)) + (1 - u) * v * tmp.getpixel((i,j + 1)) + u * (1 - v) * tmp.getpixel((i + 1,j)) + u * v * tmp.getpixel((i + 1,j + 1)))

    return result

def quantize(input_img, level):
    output_img = Image.new("P", (input_img.size[0], input_img.size[1]))

    # cut 256 into the level number intervals
    intervalLength = 256 / level
    # the unit gray of each interval
    unitGray = float(255) / (level - 1)

    # find the level this pixel in and then multiply the unitGray
    for i in range(input_img.size[0]):
        for j in range(input_img.size[1]):
            output_img.putpixel( (i, j), int( input_img.getpixel((i, j)) /intervalLength * unitGray ) )
    return output_img
    
if __name__ == "__main__":

    inputImage = Image.open("./beauty_with_freckle.bmp")

    result192_128 = scale(inputImage, (192, 128))
    result192_128.save("scale192_128.bmp", "bmp")
    print "Scale the image to 192 X 128 and title it 'scale192_128.png'"

    # Scale Image
    # result192_128 = scale(inputImage, (192, 128))
    # result192_128.save("scale192_128.png", "png")
    # print "Scale the image to 192 X 128 and title it 'scale192_128.png'"

    # result96_64 = scale(inputImage, (96, 64))
    # result96_64.save("scale96_64.png", "png")
    # print "Scale the image to 96 X 64 and title it 'scale96_64.png'"

    # result48_32 = scale(inputImage, (48, 32))
    # result48_32.save("scale48_32.png", "png")
    # print "Scale the image to 48 X 32 and title it 'scale48_32.png'"

    # result24_16 = scale(inputImage, (24, 16))
    # result24_16.save("scale24_16.png", "png")
    # print "Scale the image to 24 X 16 and title it 'scale24_16.png'"

    # result12_8 = scale(inputImage, (12, 8))
    # result12_8.save("scale12_8.png", "png")
    # print "Scale the image to 12 X 8 and title it 'scale12_8.png'"

    # result300_200 = scale(inputImage, (300, 200))
    # result300_200.save("scale300_200.png", "png")
    # print "Scale the image to 300 X 200 and title it 'scale300_200.png'"

    # result450_300 = scale(inputImage, (450, 300))
    # result450_300.save("scale450_300.png", "png")
    # print "Scale the image to 450 X 300 and title it 'scale450_300.png'"

    # result500_200 = scale(inputImage, (500, 200))
    # result500_200.save("scale500_200.png", "png")
    # print "Scale the image to 500 X 200 and title it 'scale500_200.png'"

    # # Quantization
    # level2 = 2
    # quantization2 = quantize(inputImage, level2)
    # quantization2.save("quantization2.png", "png")
    # print "Quantize the image into 2 level and title it 'quantization2.png'"

    # level4 = 4
    # quantization4 = quantize(inputImage, level4)
    # quantization4.save("quantization4.png", "png")
    # print "Quantize the image into 4 level and title it 'quantization4.png'"

    # level8 = 8
    # quantization8 = quantize(inputImage, level8)
    # quantization8.save("quantization8.png", "png")
    # print "Quantize the image into 8 level and title it 'quantization8.png'"

    # level32 = 32
    # quantization32 = quantize(inputImage, level32)
    # quantization32.save("quantization32.png", "png")
    # print "Quantize the image into 32 level and title it 'quantization32.png'"

    # level128 = 128
    # quantization128 = quantize(inputImage, level128)
    # quantization128.save("quantization128.png", "png")
    # print "Quantize the image into 128 level and title it 'quantization128.png'"
