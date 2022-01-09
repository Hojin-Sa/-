import cv2
import time

start1 = time.time()

# 이미지 불러오기
img_query = cv2.imread('box.png')           # query image
img_train = cv2.imread('box_in_scene.png')  # train image
gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
gray_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)

# SIFT
sift = cv2.xfeatures2d.SIFT_create()  # SIFT 검출기 생성

# keypoint 검출 및 descriptor 생성
kp_query = sift.detect(image=gray_query, mask=None)
kp_query, desc_query = sift.compute(image=gray_query, keypoints=kp_query)
# kp_query, desc_query = sift.detectAndCompute(image=gray_query, mask=None)
kp_train, desc_train = sift.detectAndCompute(gray_train, None)

# feature matching
bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=False)  # BFMatcher 객체 생성
matches = bf.match(queryDescriptors=desc_query, trainDescriptors=desc_train)  # descriptor 간 매칭 수행

matches = sorted(matches, key=lambda x: x.distance)  # distance를 기준으로 오름차순 정렬

# distance가 가장 작은 10개의 매칭쌍만 그리기 (Minkowski distance)
## https://docs.opencv.org/2.4/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html#drawmatches
res = cv2.drawMatches(img1=img_query,
                      keypoints1=kp_query,
                      img2=img_train,
                      keypoints2=kp_train,
                      matches1to2=matches[:10],
                      outImg=None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('result', res)
cv2.imwrite('result.png', res)
cv2.waitKey(0)

cv2.destroyAllWindows()

end1 = time.time()
print(f"{end1 - start1:.6f} sec")



start2 = time.time()

# 이미지 불러오기
img_query = cv2.imread('box.png')           # query image
img_train = cv2.imread('box_in_scene.png')  # train image
gray_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
gray_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)

# SURF
surf = cv2.xfeatures2d.SURF_create()  # SURF 검출기 생성

# keypoint 검출 및 descriptor 생성
kp_query = surf.detect(image=gray_query, mask=None)
kp_query, desc_query = surf.compute(image=gray_query, keypoints=kp_query)
# kp_query, desc_query = sift.detectAndCompute(image=gray_query, mask=None)
kp_train, desc_train = surf.detectAndCompute(gray_train, None)

# feature matching
bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=False)  # BFMatcher 객체 생성
matches = bf.match(queryDescriptors=desc_query, trainDescriptors=desc_train)  # descriptor 간 매칭 수행

matches = sorted(matches, key=lambda x: x.distance)  # distance를 기준으로 오름차순 정렬

# distance가 가장 작은 10개의 매칭쌍만 그리기 (Minkowski distance)
## https://docs.opencv.org/2.4/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html#drawmatches
res = cv2.drawMatches(img1=img_query,
                      keypoints1=kp_query,
                      img2=img_train,
                      keypoints2=kp_train,
                      matches1to2=matches[:10],
                      outImg=None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('result', res)
cv2.imwrite('result.png', res)
cv2.waitKey(0)

cv2.destroyAllWindows()

end2 = time.time()

print(img_query.shape)
print(f"SIFT: {end1 - start1:.6f} sec")
print(f"SURF: {end2 - start2:.6f} sec")





#조교님꺼
import cv2
import time

sift = cv2.xfeatures2d.SIFT_create()  # SIFT 검출기 생성
surf = cv2.xfeatures2d.SURF_create()  # SURF 검출기 생성

for scale_factor in [0.5, 1.0, 2.0, 10]:
    #이미지 불러와서 리사이징
    img = cv2.imread('butterfly.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
    print('>>', gray.shape)

    # SIFT 특징 검출 속도 계산
    t1 = time.time()
    keypoints = sift.detect(image=gray, mask=None)
    t2 = time.time()
    print('SIFT: %f sec' % (t2 - t1))

    # SURF 특징 검출 속도 계산
    t1 = time.time()
    keypoints = surf.detect(image=gray, mask=None)
    t2 = time.time()
    print('SURF: %f sec' % (t2 - t1))