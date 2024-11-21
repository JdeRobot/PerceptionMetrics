__author__ = 'frivas'

import os
import json
import cv2
import numpy as np
import copy


def checkRange(data_in):
	d=data_in
	if d <= 0.1:
		d=0.1
	return d

class sample:
    def __init__(self,image,roi):
        self.image = image
        self.x = roi[0]
        self.y = roi[1]
        self.w=roi[2]
        self.h=roi[3]
        self.scale=1
        self.angle=0
        self.loaded=False
        self.flipped=False



    def getImage(self):
        tempImage=copy.copy(self.image)
        return tempImage

    def getImageWidth(self):
        return np.size(self.image,1)

    def getImageHeight(self):
        return np.size(self.image,0)

    def getManualSegments(self):
        return self.manualSegments

    def getAutoSegments(self):
        return self.autoSegments

    def getAllSegments(self):
        return self.manualSegments + self.autoSegments

    def getInvalidRegions(self):
        return self.invalidPolygons

    def getImageWithData(self):
        imageProjected=copy.copy(self.image)
        cv2.rectangle(imageProjected, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 0), 2)
        return imageProjected



    def applyScale(self,scale):
        self.scale=scale
        res = cv2.resize(self.image,None,fx=scale, fy=scale)
        self.image=res

        self.x=int(self.x*scale)
        self.y=int(self.y*scale)
        self.w=int(self.w*scale)
        self.h=int(self.h*scale)

    def applyWarp(self,angle):
        self.angle=angle
        if angle != 0:
            rows = self.getImageHeight()
            cols = self.getImageWidth()
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            dst = cv2.warpAffine(self.image,M,(cols,rows))
            self.image=dst


            vertices=list()
            vertices.append((self.x,self.y))
            vertices.append((self.x,self.y+ self.h))
            vertices.append((self.x + self.w,self.y))
            vertices.append((self.x + self.w,self.y + self.h))



            warpedVertices=list()

            for vertex in vertices:
                array=[[[vertex[0],vertex[1]]]]
                newVertex=cv2.transform(np.array(array),M)
                [[vertex[0],vertex[1]]]
                warpedVertices.append([[int(newVertex[0][0][0]), int(newVertex[0][0][1])]])
            (self.x, self.y, self.w, self.h) = cv2.boundingRect(np.array(warpedVertices))


    def applyFlip(self):
        self.flipped=True
        self.load()
        res = cv2.flip(self.image,0)
        self.image=res
        newPolygons=list()

        self.y = self.getImageHeight() - self.y


    def getFlip(self):
        return self.flipped

    def getScale(self):
        return self.scale

    def getAngle(self):
        return self.angle

    def printResult(self):
        print(self.getResultStr())



    def getResultStr(self):
        tempX=checkRange(float(self.x)/float(self.getImageWidth()))
        tempY=checkRange(float(self.y)/float(self.getImageHeight()))
        if self.w + self.x >= self.getImageWidth():
            self.w= self.getImageWidth() - 1 - self.x
        if self.h + self.y >= self.getImageHeight():
            self.h= self.getImageHeight() - 1 - self.y
        return str(0) + ' ' +  str(tempX) + ' ' + str(tempY) + ' ' +  str(float(self.w)/float(self.getImageWidth())) +  ' ' +  str(float(self.h)/float(self.getImageHeight()))
