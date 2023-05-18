import cv2

wajah = cv2.CascadeClassifier('.xml')

img = cv2.imread('Face 2.jpeg')
img_gray = cv2.cvcolor(img, cv2.COLOR_GRAY)

deteksi_wajah = wajah.detectorMultiscale(
    img_gray,
    1.3,
    5
)

for(x,y,w,h) in deteksi_wajah:
    cv2.rectangle(
        img,
        (x,y),
        (x+w, y+h),
        (0, 255,0),
        2
    )

    cv2.imshow('Facedetector', img)

    cv2.waitkey(0)
    cv2.destroyAllWindows()
