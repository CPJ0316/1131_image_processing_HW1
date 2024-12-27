import numpy as np
import cv2

image = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", image)
parameter=[1,3,5,7]

image_2 = cv2.imread("img2.jpg", cv2.IMREAD_GRAYSCALE)
for i in parameter:
    shaped_2=cv2.Sobel(image_2, -1, 1, 1, i)
    adinfor_sharped_2 = cv2.addWeighted(image_2, 1.0, shaped_2, -0.5, 0)
    cv2.namedWindow( "s_shaped_"+str(i), cv2.WINDOW_NORMAL )
    cv2.imshow("s_shaped_"+str(i), adinfor_sharped_2)
    cv2.imwrite("hw1_s\\s_img1_shaped__"+str(i)+".jpg", adinfor_sharped_2)
    cv2.imwrite("hw1_s_edge\\s_img1_shaped__"+str(i)+".jpg", shaped_2)

'''
cv2.Sobel(src, ddepth, dx, dy, ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
# img 來源影像
# ddepth 影像深度，設定 -1 表示使用圖片原本影像深度
# dx 針對 x 軸抓取邊緣
# dy 針對 y 軸抓取邊緣
# ksize 運算區域大小，預設 1 ( 必須是正奇數 )
# scale 縮放比例常數，預設 1 ( 必須是正奇數 )

cv2.Laplacian(src, ddepth, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

#delta=添加到結果中的偏移量
#borderType （可選，預設值為 cv2.BORDER_DEFAULT）

圖像邊界的處理方式，常見選項：
cv2.BORDER_CONSTANT：填充邊界為常數值。
cv2.BORDER_REPLICATE：複製最外層的像素。
cv2.BORDER_REFLECT：對稱填充。

cv2.addWeighted(src1, alpha, src2, beta, gamma)
alpha, beta:權重係數
gamma:偏移量（亮度調整）
'''

parameter_f=[1,3,5,7,11,21,51]
for i in parameter_f:
    # 傅利葉
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    #Create Filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Create mask
    mask = np.ones((rows, cols, 2), np.uint8)
    r=(i-1)//2
    mask[crow - r:crow + r + 1, ccol - r:ccol + r + 1] = 0
    
    #Apply the Filter
    fshift = dft_shift * mask

    #逆傅立葉
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])#取兩個參數的平方和取根

    # normalize
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)

    adinfor_sharped = cv2.addWeighted(image, 1, img_back, -0.5, 0)

    # Display the results
    cv2.namedWindow( "f_shaped_"+str(i), cv2.WINDOW_NORMAL )
    cv2.imshow("f_shaped_"+str(i), adinfor_sharped)
    cv2.imwrite("hw1_f\\f_img1_shaped_"+str(i)+".jpg", adinfor_sharped)
    cv2.imwrite("hw1_f_edge\\f_img1_shaped_"+str(i)+".jpg", img_back)

cv2.waitKey(0)
