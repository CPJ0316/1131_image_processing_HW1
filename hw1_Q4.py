import numpy as np
import cv2

# 读取图像
image = cv2.imread("img1.jfif", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)

# 图像尺寸
rows, cols = image.shape

# 定义参数
parameter = [0.1,0.5, 1, 5,10]

def creat_gaussian_kernel(size,sigma):
    k=size//2
    x,y=np.mgrid[-k:k+1,-k:k+1] #np.mgrid 用來生成座標網格，若 size = 5，則範圍為 [-2, -1, 0, 1, 2]
    gauss=np.exp(-(x**2+y**2)/(2*sigma**2)) #計算高斯值
    kernel=gauss/gauss.sum() #normalization
    return kernel

for i in parameter:
    # 產生kernel
    low_pass_kernel=creat_gaussian_kernel(3,i)

    # 将空间域滤波器扩展到与图像相同大小
    padded_kernel = np.zeros((rows, cols), dtype=np.float32)
    padded_kernel[:3, :3] = low_pass_kernel

    # 進行傅立葉轉換+移到中心
    dft_image = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft_image)
    dft_kernel = cv2.dft(np.float32(padded_kernel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_kernel_middle = np.fft.fftshift(dft_kernel)

    # 应用频域滤波器
    dft_filtered = np.zeros_like(dft_image)
    dft_filtered[:, :, 0] = dft_shift[:, :, 0] * dft_kernel_middle[:, :, 0] - dft_shift[:, :, 1] * dft_kernel_middle[:, :, 1]
    dft_filtered[:, :, 1] = dft_shift[:, :, 0] * dft_kernel_middle[:, :, 1] + dft_shift[:, :, 1] * dft_kernel_middle[:, :, 0]

    # 逆傅里叶变换还原图像
    f_ishift = np.fft.ifftshift(dft_filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化到可显示范围
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    # 显示与保存结果
    cv2.imshow(f"FourierFilter_{i}", img_back)
    cv2.imwrite(f"hw1_lp_F\\Fourier_LowPassFilter_{i}.jpg", img_back)

cv2.waitKey(0)
cv2.destroyAllWindows()
