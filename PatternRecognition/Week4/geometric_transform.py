import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

plt.figure(figsize=(17, 5))

img = cv2.imread('hand.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height = img.shape[0]
width = img.shape[1]
plt.subplot(1, 3, 1)
plt.imshow(img)
plt.axis('off')
plt.title('original')


# OpenCV를 이용한 변환 행렬 도출
center = (width / 2, height / 2)
cv_M = cv2.getRotationMatrix2D(center, 90, 1.0)  # 회전 방향이 반시계방향(CCW; Counter Clock-Wise)
cv_result = cv2.warpAffine(img, cv_M, (width, height))
print('>> OpenCV Rotation matrix')
print(cv_M, end='\n\n')

plt.subplot(1, 3, 2)
plt.imshow(cv_result)
plt.axis('off')
plt.title('cv_result')

# 직접 도출한 행렬을 이용한 회전 변환
'''
M1 = np.array([[1, 2, 3],
               [2, 3, 4],
               [5, 6, 7]])
M2 = np.array([[1, 1, 0],
               [2, 1, 1],
               [1, 1, 0]])
M3 = np.matmul(M1, M2)
'''
fwd_tran_M = np.array([[1, 0, -width/2],
                       [0, 1, -height/2],
                       [0, 0, 1]])

rad = math.pi / 2
M1 = np.array([[math.cos(rad), -(math.sin(rad)), 0],
               [math.sin(rad), math.cos(rad), 0],
               [0, 0, 1]])
               
og_tran_M = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

new_M = np.matmul(fwd_tran_M, M1)
my_M = np.matmul(new_M, og_tran_M)

print('>> My matrix')
print(my_M)
my_result = cv2.warpAffine(img, my_M[:2, ], (width, height))

plt.subplot(1, 3, 3)
plt.imshow(my_result)
plt.axis('off')
plt.title('my_result')

# figure 출력
plt.tight_layout()
plt.show()
