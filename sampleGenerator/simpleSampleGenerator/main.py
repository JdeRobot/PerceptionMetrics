import os
import cv2
import numpy as np
import sys
import json
from sampleGenerator.sample import sample



input_folder = '/mnt/large/pentalo/sample/data/images/'
output_folder = './out/'


def getIfPropertyToTest(sampleNumber, testRatio):
        return  sampleNumber % 10 < testRatio*10




def mainDepth():
    imagesList = []
    fileLists = os.listdir(input_folder + '/camera2/')
    imagesList += [each for each in fileLists if each.endswith('.png')]
    sorted_files = sorted(imagesList, key=lambda x: int(x.split('.')[0]))

    imageIn=cv2.imread(input_folder + '/camera2/' +  sorted_files[1])
    bkg, _,_ = cv2.split(imageIn)
    fundo = cv2.blur(bkg, (5, 5))
    cv2.imshow("test", fundo)
    cv2.waitKey(0)
    for image in sorted_files[2:-1]:
        imagePath = input_folder + '/camera2/' + image
        imageIn = cv2.imread(imagePath)
        imagem,_,_ = cv2.split(imageIn)

        mascara = imagem.copy()
        cinza = imagem.copy()
        # cv2.imshow("Webcam", imagem)
        imagem = cv2.blur(imagem, (5, 5))
        cv2.absdiff(imagem, fundo, mascara)
        gray = mascara
        ret, thresh1 = cv2.threshold(gray, 50 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cinza = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        cinza = cv2.blur(cinza, (9, 9))
        cv2.imshow("a", cinza)
        cv2.waitKey(0)
        contorno, heir = cv2.findContours(cinza, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maiorArea=0
        for cnt in contorno:
            if (cv2.contourArea(cnt) > maiorArea):
                maiorArea = cv2.contourArea(cnt)
                cntInterest=cnt

        if len(contorno) != 0 :
            vertices_do_retangulo = cv2.boundingRect(cntInterest)
            retangulo_de_interesse = vertices_do_retangulo

            ponto1 = (retangulo_de_interesse[0], retangulo_de_interesse[1])
            ponto2 = (
            retangulo_de_interesse[0] + retangulo_de_interesse[2], retangulo_de_interesse[1] + retangulo_de_interesse[3])
            cv2.rectangle(imagem, ponto1, ponto2, (100), 2)
            cv2.rectangle(cinza, ponto1, ponto2, (255), 1)
            largura = ponto2[0] - ponto1[0]
            altura = ponto2[1] - ponto1[1]
            cv2.line(cinza, (ponto1[0] + largura / 2, ponto1[1]), (ponto1[0] + largura / 2, ponto2[1]), (255), 1)
            cv2.line(cinza, (ponto1[0], ponto1[1] + altura / 2), (ponto2[0], ponto1[1] + altura / 2), (255), 1)

            cv2.imshow("Mascara", mascara)
            cv2.imshow("Cinza", cinza)

            cv2.imshow("Webcam", imagem)
            # cv2.imshow("Thresholded", thresh1)
            # cv2.imshow("Fundo", fundo)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(1)




def maincolor():
    imagesList = []
    fileLists = os.listdir(input_folder + '/camera1/')
    imagesList += [each for each in fileLists if each.endswith('.png')]
    sorted_files = sorted(imagesList, key=lambda x: int(x.split('.')[0]))

    bkg=cv2.imread(input_folder + '/camera1/' +  sorted_files[1])
    fundo = cv2.blur(bkg, (5, 5))
    cv2.imshow("test", fundo)
    cv2.waitKey(0)
    for image in sorted_files[2:-1]:
        imagePath = input_folder + '/camera1/' + image
        imagem = cv2.imread(imagePath)


        mascara = imagem.copy()
        cinza = imagem.copy()
        # cv2.imshow("Webcam", imagem)
        imagem = cv2.blur(imagem, (7, 7 ))
        cv2.absdiff(imagem, fundo, mascara)
        gray = cv2.cvtColor(mascara, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cinza = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        cinza = cv2.blur(cinza, (9, 9))
        cv2.imshow("a", cinza)
        cv2.waitKey(0)
        contorno, heir = cv2.findContours(cinza, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        maiorArea=0
        for cnt in contorno:
            if (cv2.contourArea(cnt) > maiorArea):
                maiorArea = cv2.contourArea(cnt)
                cntInterest=cnt

        if len(contorno) != 0 :
            vertices_do_retangulo = cv2.boundingRect(cntInterest)
            retangulo_de_interesse = vertices_do_retangulo

            ponto1 = (retangulo_de_interesse[0], retangulo_de_interesse[1])
            ponto2 = (
            retangulo_de_interesse[0] + retangulo_de_interesse[2], retangulo_de_interesse[1] + retangulo_de_interesse[3])
            cv2.rectangle(imagem, ponto1, ponto2, (0, 0, 0), 2)
            cv2.rectangle(cinza, ponto1, ponto2, (255, 255, 255), 1)
            largura = ponto2[0] - ponto1[0]
            altura = ponto2[1] - ponto1[1]
            cv2.line(cinza, (ponto1[0] + largura / 2, ponto1[1]), (ponto1[0] + largura / 2, ponto2[1]), (255, 255, 255), 1)
            cv2.line(cinza, (ponto1[0], ponto1[1] + altura / 2), (ponto2[0], ponto1[1] + altura / 2), (255, 255, 255), 1)

            cv2.imshow("Mascara", mascara)
            cv2.imshow("Cinza", cinza)

            cv2.imshow("Webcam", imagem)
            # cv2.imshow("Thresholded", thresh1)
            # cv2.imshow("Fundo", fundo)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(1)

def main():
    minPoints=400
    index=0
    imagesList=[]
    fileLists= os.listdir(input_folder+'/camera1/')
    imagesList += [each for each in fileLists if each.endswith('.png')]
    sorted_files = sorted(imagesList, key=lambda x: int(x.split('.')[0]))

    print('source ', input_folder +  '/camera1/' + sorted_files[1])

    sourceImage=cv2.imread(input_folder + '/camera1/' +  sorted_files[1])
    sourceImage=cv2.cvtColor(sourceImage,cv2.COLOR_BGR2RGB)
    cv2.imshow("test",sourceImage)

    sourceGray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
    sourceGray = cv2.GaussianBlur(sourceGray, (21, 21), 0)

    (H,W,c) = np.shape(sourceImage)
    cv2.waitKey(0)

    #warps=[0,90,180,270]
    #scales=[1,1.5,2]
    warps=[0]
    scales=[1]
    testRatio=0.3
    imagesCounter=0
    th=60
    for image in sorted_files[2:-1]:
        testImage=False
        for warp in warps:
            print("w: ", warp)
            for scale in scales:
                imagePath = input_folder +  '/camera1/' + image
                currImage=cv2.imread(imagePath)
                currImage=cv2.cvtColor(currImage,cv2.COLOR_BGR2RGB)


                gray = cv2.cvtColor(currImage, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                frameDelta = cv2.absdiff(sourceGray, gray)
                thresh = cv2.threshold(frameDelta, th, 255, cv2.THRESH_BINARY)[1]

                # dilate the thresholded image to fill in holes, then find contours
                # on thresholded image
                # thresh = cv2.dilate(thresh, None, iterations=2)
                kernel = np.ones((5,5),np.uint8)
                thresh=cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                cv2.imshow("th",thresh)

                nPoints = cv2.countNonZero(thresh)
                if nPoints < minPoints * scale:
                    nPoints=0
                if nPoints:
                    filename, file_extension = os.path.splitext(image)
                    if (warp == warps[0] and scale == scales[0]):
                        imagesCounter+=1
                    testImage=getIfPropertyToTest(imagesCounter,testRatio)
                    print(testImage)
                    if testImage:
                        cv2.imwrite("test/" + image,currImage)

                    points = cv2.findNonZero(thresh)
                    roi = cv2.boundingRect(points)
                    s = sample(currImage,roi)
                    s.applyScale(scale)
                    s.applyWarp(warp)
                    s.printResult()
                    maskedImage = s.getImageWithData()
                    cv2.imshow("current",maskedImage)
                    if testImage:
                        cv2.imwrite("test/" + filename + '-mask.png',maskedImage)
                    else:
                        cv2.imwrite(output_folder + '/angle' + str(warp) + '_scale'+ str(scale) +'_' + filename + "-mask.png",maskedImage)
                        cv2.imwrite(output_folder + '/angle' + str(warp) + '_scale'+ str(scale) + '_' + filename + '.png',s.getImage())

                        f = open(output_folder + '/angle' + str(warp) + '_scale'+ str(scale) + '_' + filename + '.txt', 'w')
                        f.write(s.getResultStr())
                        f.close()

                    cv2.waitKey(1)
                if testImage:
                    break
            if testImage:
                break


    print(imagesCounter)



def main2():
    min_area=50
    imagesList = []
    fileLists = os.listdir(input_folder + '/camera1/')
    imagesList += [each for each in fileLists if each.endswith('.png')]
    sorted_files = sorted(imagesList, key=lambda x: int(x.split('.')[0]))

    frame=cv2.imread(input_folder + '/camera1/' +  sorted_files[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    cv2.imshow("test", gray)
    cv2.waitKey(0)
    for image in sorted_files[2:-1]:
        imagePath = input_folder + '/camera1/' + image
        imagem = cv2.imread(imagePath)

        rameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        majorArea=0
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < min_area:
                continue
            else:
                if cv2.contourArea(c) > majorArea:
                    majorArea=cv2.contourArea(c)
                    bestCnt=c

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(bestCnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
