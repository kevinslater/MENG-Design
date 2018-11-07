import numpy as np
import cv2

def write(img,location):
    cv2.imwrite(location,img)

class Image(object):

    def __init__(self, location): #location is str to image's directory
        self.location = location
        self.readImage()
        self.process()
        self.getShapes()
        self.drawShapes()

    def readImage(self):
        #read image as 3d array
        self.image = cv2.imread(self.location)

    def process(self):
        #smooth the image with a bilateral filter
        blur = cv2.bilateralFilter(self.image,9,150,150)
        #turn everything below (limit) to black
        ret,proc = cv2.threshold(blur,100,255,cv2.THRESH_TOZERO)
        #turn everything above (limit) to white
        ret,proc = cv2.threshold(proc,180,255,cv2.THRESH_TRUNC)
        #highlight the edges with Canny edge detection
        self.processed_image = cv2.Canny(proc,100,200)
        
    def getShapes(self):
        #find shapes in processed (binary) image
        _, contours, _ = cv2.findContours(self.processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.shapes = [Shape(contour) for contour in contours]
        #distinguish between "big" shapes
        big_shapes = []
        for shape in self.shapes:
            if shape.area>100:
                big_shapes.append(shape)
        self.big_shapes = big_shapes

    def drawShapes(self):
        #draw big shapes over original image
        big_contours = [shape.contour for shape in self.big_shapes]
        shapes_image = np.copy(self.image)
        self.shapes_image = cv2.drawContours(shapes_image, big_contours,  -1, (0,0,255), 1 )

    def splitShapes(self,shapedir):
        #crop each shape
        shapes_split = [Shape.crop(self.shapes_image) for Shape in self.big_shapes]        
        idx = 0
        for shape in shapes_split:
            idx += 1
            name = self.location[-5] + "_" + str(len(self.big_shapes))+ "_" + str(idx)
            write(shape,shapedir+name+".jpg")
            #why do we output two copies of every image??

    def writeLabels(self,target):
        idx = 0
        for shape in self.big_shapes:
            idx+=1
            name = self.location[-5] + "_" + str(len(self.big_shapes))+ "_" + str(idx)
            loc = target+str(shape.label[0])+"/"+name+".jpg"
            write(shape.cropped, loc)

class Shape(object):

    def __init__(self, contour):
        self.contour = contour
        self.label = False
        self.getArea()
        self.getApprox()
        self.getBoundary()
    
    def getArea(self):
        self.area = cv2.contourArea(self.contour)

    def getApprox(self):
        self.approx = cv2.approxPolyDP(self.contour,0.01*cv2.arcLength(self.contour,True),True)

    def getBoundary(self): 
        x,y,w,h = cv2.boundingRect(self.contour)
        self.h = h #height
        self.w = w #width
        self.boundary = [y,y+h,x,x+w]

    def crop(self,parent):
        self.cropped = parent[self.boundary[0]:self.boundary[1],self.boundary[2]:self.boundary[3]]
        return(self.cropped)

    def pad(self,maxh,maxw):
        #pad to make all cropped images the same size before clustering
        self.padded = np.pad(self.cropped,((0, maxh - self.h), (0, maxw - self.w),(0,0)), 'constant', constant_values=0)

    def flatten(self):
        #when clustering, each shape is represented as a single row of values
        #only flatten AFTER padding
        self.flat = np.ndarray.flatten(self.padded)
 
    def setLabel(self,label):
        self.label = label
