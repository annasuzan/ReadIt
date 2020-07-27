import cv2
import numpy as np

img = cv2.imread('7.jpg',0)

cv2.imshow('Image',img)
cv2.waitKey(0)

_,thresh = cv2.threshold(img,116,255,cv2.THRESH_BINARY_INV)
contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
i = 0
for cnt in contours:
    i = i+1
    if cv2.contourArea(cnt) > 500:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cropped = img[y:y+h,x:x+w]
        cv2.imwrite(str(i)+'s.jpg',cropped)
        cv2.imshow('Image',img)
        cv2.waitKey(0)

cv2.destroyAllWindows()