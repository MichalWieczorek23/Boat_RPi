import random
import time
import cv2

image = cv2.imread('Screenshot_20221214-125535(1).jpg')
image = cv2.resize(image, (600, 900))
# Ininicjalizacja modulu 'selective search' as 'ss'
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()        # Dostepna jest wersja fast lub quality

start = time.time()
rects = ss.process()
end = time.time()
print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))


# Przejscie przez kawalki segmentacji w celu wizualizacji. Przegladamy tylko okienka o rozmiarze wiekszym niz 20% baseImage.
roi = image.copy()
for (x, y, w, h) in rects:

    # Sprawdzeni czy wysokosc i szerokosc wycinka stanowia przynajmniej 20% obrazu.
    if (w / float(image.shape[1]) < 0.2 or h / float(image.shape[0]) < 0.2):
        continue

    # Wizualizacja ROIs
    cv2.rectangle(roi, (x, y), (x + w, y + h),
                  (0, 200, 0), 2)

roi = cv2.resize(roi, (640, 640))
final = cv2.hconcat([cv2.resize(image, (640, 640)), roi])       # Konkatenacja okien i obrazu bazowego
cv2.imshow('ROI', final)
cv2.waitKey(0)
