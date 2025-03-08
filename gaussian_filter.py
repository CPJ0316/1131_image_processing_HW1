import numpy as np
import cv2

image = cv2.imread("img1.jfif")
cv2.imshow("Original", image)
parameter=[0.1,0.5,1,1.5,3]

def create_gaussian_kernel(size, sigma):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    gauss = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = gauss / gauss.sum()
    return kernel


for i in parameter:
    #blurred=cv2.blur(image, (i, i))
    #filtered_image = cv2.GaussianBlur(image, (3, 3), sigmaX=i) 
    #adfiltered_image = cv2.addWeighted(image, 1.0, filtered_image, -0.5, 0)
    kernel = create_gaussian_kernel(size=3, sigma=i) #生成高斯濾波器
    filtered_image = cv2.filter2D(image, -1, kernel) #應用濾波器
    
    cv2.namedWindow( "GaussianBlur_"+str(i), cv2.WINDOW_NORMAL )
    cv2.imshow("GaussianBlur_"+str(i), filtered_image)
    cv2.imwrite("hw1_g\\g_img1_GaussianBlur_"+str(i)+".jpg", filtered_image)

cv2.waitKey(0)
    
    #在 OpenCV 的 cv2.GaussianBlur 函數中，sigmaX 是高斯濾波器在 X 軸方向 的標準差 (𝜎σ)。它控制高斯分布的寬度，直接影響濾波器的平滑效果。
    #如果未指定 sigmaX 或將其設為 0:OpenCV 會根據濾波器的大小自動計算一個適合的標準差：
