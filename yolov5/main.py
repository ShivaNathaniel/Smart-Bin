import numpy
import torch
import cv2

#load model
model = torch.hub.load('ultralytics/yolov5', 'custom',path='best.pt') #'ultralytics/yolov5', 'yolov5s'
#model = torch


#load picture
#frame = cv2.imread('datasets/coco128/images/train2017/000000000009.jpg') # load anh , phim , camera - thu vien khac nhau
set = cv2.imread('datasets/coco128/images/train2017/l-180.jpg')
giay = cv2.imread('giay.jpg')
but = cv2.imread('but.jpg')
nilon = cv2.imread('nilon.jpg')
volon = cv2.imread('volon.jpg')
#detect
#detections = model(frame
#detections = model(frame)
detections =model(set)
#print results
results = detections.pandas().xyxy[0].to_dict(orient="records")
x= numpy.array(results)
print(x)        
for result in results:
    confidence = result['confidence']
    name = result['name']
    clas = result['class']
    if clas == 0:
        cv2.imshow('giay',giay)
    elif clas == 1:
        cv2.imshow('but',but)
    elif clas == 2:
        cv2.imshow('nilon',nilon)
    elif clas == 3:
        cv2.imshow('volon',volon)
        #x1 = int(result['xmin'])
        #y1 = int(result['ymin'])
        #x2 = int(result['xmax'])
        #y2 = int(result['ymax'])
        #print(x1,y1,x2,y2)

        #cv2.rectangle(frame,(x1,y1),(x2, y2),(255,0,0),2)

        #cv2.putText(frame, name, (x1+3,y1-10),cv2.FONT_HERSHEY_DUPLEX,1,(60,255,255),1)

#cv2.imshow('img',frame)
cv2.waitKey(0)

