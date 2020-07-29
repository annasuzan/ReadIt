#iMPORTING MODULES
import cv2
import numpy as np
import tensorflow as tf
import keras

recognizer = tf.keras.models.load_model('emnist_classifier_mark2.h5')    #importing the model that took 1.5 hours#

#FUNCTIONS
def nothing(x):                                                     #Prints value of slider in trackbar
    print(x)

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


#MAIN 

img = cv2.imread('Attempt4.jpg',0)                                      #Input Image
row,col = img.shape

cv2.imshow('Image',img)
cv2.waitKey(0)                                                          #waits till a key is pressed
# print(str(row)+','+str(col))

#Trackbar Code
cv2.namedWindow('Track')
cv2.createTrackbar('Left','Track',0,col,nothing)
cv2.createTrackbar('Right','Track',col,col,nothing)
cv2.createTrackbar('Top','Track',0,row,nothing)
cv2.createTrackbar('Bottom','Track',row,row,nothing)
cv2.createTrackbar('Filter','Track',0,255,nothing)
cv2.createTrackbar('Select','Track',0,1,nothing)


while True:
    l = cv2.getTrackbarPos('Left','Track')
    r = cv2.getTrackbarPos('Right','Track')
    t = cv2.getTrackbarPos('Top','Track')
    b = cv2.getTrackbarPos('Bottom','Track')
    
    roi = img[t:b,l:r]                                              #image is cropped to get the region of interest(roi)
    
    filter = cv2.getTrackbarPos('Filter','Track')
    _,thresh = cv2.threshold(roi,filter,255,cv2.THRESH_BINARY_INV)

    cv2.imshow('Image',thresh)
    cv2.waitKey(1)

    check = cv2.getTrackbarPos('Select','Track')                    #when the 'select' slidebar is 1,exits the image editing
    if check == 1:
        break

#STUFF JOEL KNOWS (DILATION)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))   
dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)

#Generates an array of contours
contours,_ = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

paragraph = ""






for cnt in contours:

    if cv2.contourArea(cnt) >950:
        (x,y,w,h) = cv2.boundingRect(cnt)               #function returns measurements for the bounding rectangle         
        cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)  #generates the rectangle arounf the contour of a word

        cropped = roi[y:y+h,x:x+w]                      #cropping the roi to get a word

        
        _,thresh_word = cv2.threshold(cropped,116,255,cv2.THRESH_BINARY_INV)
        contours_word,_ = cv2.findContours(thresh_word,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        predictions = {}
        index = []
        word = ""

        for cnt_w in contours_word:
            if cv2.contourArea(cnt_w) > 500:
                (x,y,w,h) = cv2.boundingRect(cnt_w)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                cropped_w = thresh_word[y:y+h,x:x+w]
                cropped_w = cv2.resize(cropped_w,(20,20))
                cropped_w = cv2.copyMakeBorder(cropped_w,4,4,4,4,cv2.BORDER_CONSTANT,value = (0,0,0))
        
                cropped_w = np.asarray(cropped_w)
                cropped_w = cropped_w.reshape(-1,28,28,1)
                cropped_w = cropped_w/255
                pred = recognizer.predict_classes(cropped_w)
        
                index.append(x)
                predictions[str(x)] = convert_to_letter(pred[0])
        
                cv2.imshow('Image',cropped)
                cv2.waitKey(0)

        index.sort()

        for val in index:
            word.append(predictions[str(val)])

        paragraph = paragraph + " " + word


file = open("Text.txt","a")
file.write(paragraph)
file.write("\n")
file.close

cv2.imshow('Result',roi)
cv2.waitKey(0)

cv2.destroyAllWindows()




