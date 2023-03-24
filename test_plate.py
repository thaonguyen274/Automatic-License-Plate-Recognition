


from ultralytics import YOLO
import cv2
import imutils
import shutil
import os,glob
from ocrmodule import *

ocrpile = OCR_PILE()

model_detect = YOLO("best.pt")
label = ["plate"]
for name in glob.glob("C:/Users/thao.nguyenthu1/Desktop/images/*.jpg"):
    print(name)
    frame = cv2.imread(name)
    results = model_detect.predict(source=frame, conf=0.5)
    for rs in results:
        boxes = rs.boxes
        if len(boxes) == 0: 
            continue
        for box in boxes:
            bbox = box.xyxy.tolist()[0]
            class_name = label[int(box.cls.tolist()[0])]+ "_" + str(box.conf.tolist()[0])[:4]
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1,y1,x2,y2 = xmin, ymin, xmax, ymax
            img_cutted = frame[y1:y2,x1:x2,:].copy()
            # print(img_cutted)
            if (x2 - x1)/(y2-y1)<3:
                img0 = img_cutted[:(y2-y1)//2,:,:]
                img1 = img_cutted[(y2-y1)//2:,:,:]
                license = ocrpile.ocr(img0)[0] + ocrpile.ocr(img1)[0]
            else:
                license = ocrpile.ocr(img_cutted)[0]
            cv2.putText(frame, license,(xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0),2)
    
    cv2.imshow('webcam',  imutils.resize(frame, width=480))
    # cv2.imshow('img_cutted',  imutils.resize(img_cutted, width=200))
    cv2.waitKey(0)