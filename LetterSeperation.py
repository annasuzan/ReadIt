import cv2
import numpy as np
import tensorflow as tf 
import keras

def convert_to_letter(num):
    if num>=0 and num<=9 :
        ref = 48
        ascii_val = num + ref
    elif num>=10 and num<=35 :
        ref = 55
        ascii_val = num + ref
    elif num>=36 and num<=61 :
        ref = 61
        ascii_val = num + ref
    return chr(ascii_val)


img = cv2.imread('omega.jpg',0)

cv2.imshow('Image',img)
cv2.waitKey(0)

_,thresh = cv2.threshold(img,116,255,cv2.THRESH_BINARY_INV)
contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
i = 0
recognizer = tf.keras.models.load_model('emnist_classifier_mark2.h5')

predictions = {}
index = []
words = ""

for cnt in contours:
    i = i+1
    if cv2.contourArea(cnt) > 500:
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cropped = thresh[y:y+h,x:x+w]
        cropped = cv2.resize(cropped,(20,20))
        cropped = cv2.copyMakeBorder(cropped,4,4,4,4,cv2.BORDER_CONSTANT,value = (0,0,0))
        
        cropped = np.asarray(cropped)
        cropped = cropped.reshape(-1,28,28,1)
        cropped = cropped/255
        pred = recognizer.predict_classes(cropped)
        
        index.append(x)
        predictions[str(x)] = convert_to_letter(pred[0])
        
        cv2.imshow('Image',img)
        cv2.waitKey(0)

index.sort()
for val in index:
    words = words + predictions[str(val)]

print(words)   

cv2.destroyAllWindows()