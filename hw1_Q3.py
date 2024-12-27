import numpy as np
import cv2

image = cv2.imread("img1.jfif")
cv2.imshow("Original", image)
parameter=[0.1,0.5, 1, 5,10]

def creat_gaussian_kernel(size,sigma):
    k=size//2
    x,y=np.mgrid[-k:k+1,-k:k+1]  #np.mgrid 用來生成座標網格，若 size = 5，則範圍為 [-2, -1, 0, 1, 2]
    gauss=np.exp(-(x**2+y**2)/(2*sigma**2))  #計算高斯值
    kernel=gauss/gauss.sum()  #normalization
    return kernel
    
for i in parameter:
    kernel=creat_gaussian_kernel(3,i)
    filteered_image=cv2.filter2D(image,-1,kernel)

    # 顯示與保存濾波後的圖像
    cv2.namedWindow("LowPassFilter"+str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("LowPassFilter"+str(i), filteered_image)
    cv2.imwrite("hw1_g\\g_img1_LowPassFilter_"+str(i)+".jpg", filteered_image)
    
cv2.waitKey(0)
