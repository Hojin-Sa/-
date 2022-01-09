import cv2
import numpy as np

# 모델 영상의 2차원 H-S 히스토 계산
img_m = cv2.imread('model.jpg')
hsv_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2HSV)
hist_m = cv2.calHist([hsv_m], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 입력 영상의 2차원 H-S 히스토 계산
img_i = cv2.imread('hand.jpg')
hsv_i = cv2.cvtColor(img_i, cv2.COLOR_BGR2HSV)
hist_i = cv2.calHist([hsv_i], [0, 1], None, [180, 256], [0, 180, 0, 256])

# 히스토그램 정규화
#cv2.normalize(hist_i, None, 0, 255, cv2.norm_MINMAX)
#cv2.normalize(hist_m, None, 0, 255, cv2.norm_MINMAX)
hist_m = hist_m / (img_m.shape[0] * img_m.shape[1]) # / 높이*너비
hist_i = hist_i / img_i.size
print("maximum of hist_m: % f" % hist_m.max())  # 값 범위 체크: 1.0이하
print("maximum of hist_i: % f" % hist_i.max())  # 값 범위 체크: 1.0이하

# 비율 히스토그램 계산
#hist_r = min((hist_m/hist_i), 1.0)
hist_r = hist_m / (hist_i + 1e-7)
hist_r = np.minimum(hist_r, 1.0)
print("range of hist_r: [%.1f, %.1f" % (hist_r.min(), hist_r.max()))

# 히스토그램 역투영 수행
#result = cv2.calcBackProject([img_i], [0, 1], hist_r,  [0, 180, 0, 256], 1)
height, width = img_i.shape[0], img_i.shape[1]   # 영상의 높이와 너비정보
result = np.zeros_like(img_i, dtypes='float32')  # 입력영상과 동일한 크기의 0으로 초기화된 배열 생성
h, s, v = cv2.split(hsv_i)                       # 채널 분리

for i in range(height):                          # 모든 픽셀을 순회하며 처리
    for j in range(width):
        h_value = h[i, j]                        # (i, j)번째 픽셀의 hue 값
        s_value = s[i, j]                        # (i, j)번째 픽셀의 saturation 값
        confidence = hist_r[h_value, s_value]    # (i, j)번째 픽셀의 신뢰도 함수
        result[i, j] = confidence                # 신뢰도 점수를 결과 이미지 (i, j)번째 픽셀에 저장



# 이진화 수행(화소값이 임계값 0.02보다 크면 255, 그렇지 않으면 0)
ret, thresholded = cv2.threshold(result, 0.02, 255, cv2.THRESH_BINARY)
cv2.imwrite('result.jpg', thresholded)

# 모폴로지 연산 적용
kernel = np.ones((13,13))
improved = cv2.morphologhEx(thresholded, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('morphology.jpg', improved)
