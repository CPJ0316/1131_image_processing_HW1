import cv2
import numpy as np
import matplotlib.pyplot as plt

# 設計頻域高斯低通濾波器
def create_fourier_low_pass(shape, sigma_f):
    """
    創建一個頻域中的高斯低通濾波器掩模
    :param shape: 圖像的大小 (rows, cols)
    :param sigma_f: 頻域濾波器標準差
    :return: 高斯低通濾波掩模
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            # 計算到中心的距離
            distance = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            mask[i, j] = np.exp(-(distance**2) / (2 * (sigma_f**2)))
    
    return mask

# 載入圖像
image = cv2.imread('img1.jfif', cv2.IMREAD_GRAYSCALE)

# 空間域高斯濾波器標準差（第3題的設定值）
sigma_s = 0.01  # 修改此值以改變空間域的標準差

# 計算頻域高斯濾波器的標準差
sigma_f = 1 / (2 * np.pi * sigma_s)

# Fourier Transform
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)  # 將低頻移到中心

# 創建高斯低通濾波器掩模
rows, cols = image.shape
low_pass_mask = create_fourier_low_pass((rows, cols), sigma_f )

# 應用濾波器
filtered_dft = dft_shift * low_pass_mask

# 逆傅立葉變換
inverse_dft_shift = np.fft.ifftshift(filtered_dft)
smoothed_image = np.abs(np.fft.ifft2(inverse_dft_shift))

# 可視化結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title("Original Image"), plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2), plt.title("Low-Pass Mask (Frequency)"), plt.imshow(low_pass_mask, cmap='gray')
plt.subplot(1, 3, 3), plt.title("Smoothed Image"), plt.imshow(smoothed_image, cmap='gray')
plt.show()
