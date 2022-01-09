import cv2

def pipeline(img):
#img = cv2.imread('./test_images/solidWhiteRight.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (15,15), 0.0)

    edge_img = cv2.Canny(blurred_img, 70, 140)

    return edge_img




cap = cv2.VideoCapture('./test_videos/solidWhiteRIght.mp4')

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    edge_img = pipeline(frame)

    cv2.imshow('frame', frame)
    key = cv2.waitkey(30) 
    if key == ord('x'):
        break

cap.release()




