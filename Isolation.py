import cv2
import numpy as np
import pytesseract

def nothing(x):
    print(x)

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Joel\\Desktop\\School\\Joel\\CODING\\Note-Project\\Tesseract-OCR\\tesseract.exe'


img = cv2.imread('attempt3.jpg',0)



col,row = img.shape
cv2.imshow('Image',img)
cv2.waitKey(0)
print(str(row)+','+str(col))

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
    roi = img[t:b,l:r]


    filter = cv2.getTrackbarPos('Filter','Track')
    _,thresh = cv2.threshold(roi,filter,255,cv2.THRESH_BINARY_INV)

    cv2.imshow('Image',thresh)
    cv2.waitKey(1)
    check = cv2.getTrackbarPos('Select','Track')

    if check == 1:
        break


rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
dilation = cv2.dilate(thresh, rect_kernel, iterations = 1)


contours,_ = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
i = 0
for cnt in contours:
    i = i+1
    (x,y,w,h) = cv2.boundingRect(cnt)
    cv2.rectangle(roi,(x,y),(x+w,y+h),(255,0,0),2)

    cropped = roi[y:y+h,x:x+w]

    cv2.imshow(str(i),cropped)
    cv2.waitKey(1)

    file = open("Text.txt","a")

    text = pytesseract.image_to_string(cropped)

    print('The answer is:'+text)

    file.write(text)
    file.write("\n")
    file.close

cv2.imshow('Result',roi)
cv2.waitKey(0)


    

cv2.destroyAllWindows()




