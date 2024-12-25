import numpy as np
import cv2

image = cv2.imread("img1.jfif")
cv2.imshow("Original", image)
parameter=[1.1,2,5,10]
for i in parameter:
    low_pass_kernel = np.array([[1, i, 1],
                            [i, i*i, i],
                            [1, i, 1]], dtype=np.float32) / (4+4*i+i*i)
    filtered_image = cv2.filter2D(image, -1, low_pass_kernel)

    # 顯示與保存濾波後的圖像
    cv2.namedWindow("LowPassFilter"+str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("LowPassFilter"+str(i), filtered_image)
    cv2.imwrite("hw1_g_ver2\\g_img1_LowPassFilter_"+str(i)+".jpg", filtered_image)
cv2.waitKey(0)
'''
# 定義固定的低通濾波器
low_pass_kernel = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 16.0

# 應用濾波器
filtered_image = cv2.filter2D(image, -1, low_pass_kernel)

# 顯示與保存濾波後的圖像
cv2.namedWindow("LowPassFilter", cv2.WINDOW_NORMAL)
cv2.imshow("LowPassFilter", filtered_image)
cv2.imwrite("hw1_g_ver2\\g_img1_LowPassFilter_1.jpg", filtered_image)


low_pass_kernel_2 = np.array([[1, 3, 1],
                            [3, 9, 3],
                            [1, 3, 1]], dtype=np.float32) / 25

# 應用濾波器
filtered_image = cv2.filter2D(image, -1, low_pass_kernel_2)

# 顯示與保存濾波後的圖像
cv2.namedWindow("LowPassFilter_2", cv2.WINDOW_NORMAL)
cv2.imshow("LowPassFilter_2", filtered_image)
cv2.imwrite("hw1_g_ver2\\g_img1_LowPassFilter_2.jpg", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
def create_gaussian_kernel(size, sigma):
    k = size // 2
    x, y = np.mgrid[-k:k+1, -k:k+1]
    gauss = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = gauss / gauss.sum()
    return kernel


for i in parameter:
    kernel = create_gaussian_kernel(size=3, sigma=i) #生成高斯濾波器
    filtered_image = cv2.filter2D(image, -1, kernel) #應用濾波器
    
    cv2.namedWindow( "GaussianBlur_"+str(i), cv2.WINDOW_NORMAL )
    cv2.imshow("GaussianBlur_"+str(i), filtered_image)
    cv2.imwrite("hw1_g\\g_img1_GaussianBlur_"+str(i)+".jpg", filtered_image)

cv2.waitKey(0)
'''
    
    #在 OpenCV 的 cv2.GaussianBlur 函數中，sigmaX 是高斯濾波器在 X 軸方向 的標準差 (𝜎σ)。它控制高斯分布的寬度，直接影響濾波器的平滑效果。
    #如果未指定 sigmaX 或將其設為 0:OpenCV 會根據濾波器的大小自動計算一個適合的標準差：
