import numpy as np
import cv2


image = cv2.imread("img1.jfif", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)
parameter=[1,3,5,7]
print(image.shape)
for i in parameter:
    blurred=cv2.blur(image, (i, i)) #average filter
    cv2.namedWindow( "a_blurred_"+str(i), cv2.WINDOW_NORMAL )
    cv2.imshow("a_blurred_"+str(i), blurred)
    cv2.imwrite("hw1_a\\a_img1_blurred_"+str(i)+".jpg", blurred)
    print(blurred.shape)

image = cv2.imread("img1.jfif", cv2.IMREAD_GRAYSCALE)
for i in parameter:
    blurred_2 =cv2.medianBlur(image, i)
    cv2.namedWindow( "m_blurred_"+str(i), cv2.WINDOW_NORMAL )
    cv2.imshow("m_blurred_"+str(i), blurred_2)
    cv2.imwrite("hw1_m\\m_img1_blurred_"+str(i)+".jpg", blurred_2)
    print(blurred_2.shape)
cv2.waitKey(0)
