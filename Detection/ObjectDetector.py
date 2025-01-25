import numpy as np
from tensorflow.keras.applications.xception import preprocess_input
from imutils.object_detection import non_max_suppression
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from keras import models
import time
import cv2

def ObjectDetector(image_path):
    # Stworzenie dwoch oddzielnych list, z czego jedna zawiera obraz w formacie rgb a druga zawiera wspolrzedne naroznikow
    rois = []
    boxes = []
    start = time.time()
    image = cv2.imread(image_path)
    image = cv2.resize(image, (600, 400))
    #cv2.imshow("Regional proposal object detection", image)
    #cv2.waitKey(0)

    # Ininicjalizacja modulu 'selective search' as 'ss'
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()      # Dostepna jest wersja fast lub quality


    rects = ss.process()
    (H, W) = image.shape[:2]

    for (x, y, w, h) in rects:
        # przynajmniej 20% rozmiaru obrazu i nie więcej niż 80%
        if w / float(W) < 0.2 or w / float(W) > 0.7 or h / float(H) < 0.2 or h / float(H) > 0.7:
            continue
        else:
            # Ekstrakcja ROI z obrazu zrodlowego
            roi = image[y:y + h, x:x + w]

            # Konwersja do RGB
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Resize
            roi = cv2.resize(roi, (300, 300))

            # Dalszy preprocessing
            roi = img_to_array(roi)
            roi = preprocess_input(roi)

            # Dodanie kolejnych roi do listy
            rois.append(roi)

            # Zapamietanie wspolrzednych kolejnych zaznaczonych prostokatow
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append((x1, y1, x2, y2))

    # Model
    model = models.load_model('Inz_3_notcomplete_but_improved_15_epoch_v2.h5')

    # Konwersja ROIS do numpy array w celu pozniejszej predykcji
    input_array = np.array(rois)
    print('Input array shape is;', input_array.shape)

    # Predykcje
    preds = model.predict(input_array)
    # Inicjalizacja slownika
    objects = {}
    labels = ['Butelka plastikowa','Paczka czipsow','Kubeczek plastikowy', 'Nakretka plastikowa', 'Butelka szklana', 'Puszka metalowa', 'Widelec', 'Woreczek']
    colors = [(160, 218, 215),(135, 221, 124),(135, 80, 124),(225, 194, 0),(225, 199, 255),(252, 167, 174),(252, 119, 174)]

    trshd = 0.99
    for (i, pred) in enumerate(preds):
        max_value = max(pred)
        if (max_value) >= trshd:
            index_max = max(range(len(pred)), key=pred.__getitem__)

            box = boxes[i]
            label = labels[index_max]
            value = objects.get(label, [])

            value.append((box, max_value))

            objects[label] = value

    img_copy = image.copy()
    '''
    # Przejscie po wszystkich obiektach ktorych prawdopodobienstwo jest wieksze niz 90
    for label in objects.keys():

        boxes = np.array([pred[0] for pred in objects[label]])
        proba = np.array([pred[1] for pred in objects[label]])
        boxes = non_max_suppression(boxes, proba)

        # Wypakowanie wspolrzednych ograniczajacego "boxa"
        (startX, startY, endX, endY) = boxes[0]

        # Zaznaczenie ograniczajacego "boxa"
        cv2.rectangle(img_copy, (startX, startY),
                      (endX, endY), colors[labels.index(label)], 2)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        # Dodanie nazwy etykiety na wykrytym obiekcie
        cv2.putText(img_copy, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[labels.index(label)], 2)

        # Wyswietlenie obrazu
    '''
    total_x = 0
    total_y = 0
    total_x_end = 0
    total_y_end = 0
    i = 0
    for label in objects.keys():

        boxes = np.array([pred[0] for pred in objects[label]])
        proba = np.array([pred[1] for pred in objects[label]])
        boxes = non_max_suppression(boxes, proba)

        # Wypakowanie wspolrzednych ograniczajacego "boxa"
        (startX, startY, endX, endY) = boxes[0]
        total_x += startX
        total_y += startY
        total_x_end += endX
        total_y_end += endY
        i += 1

    avg_x = total_x / i
    avg_y = total_y / i
    avg_x_end = total_x_end / i
    avg_y_end = total_y_end / i

    # Zaznaczenie ograniczajacego "boxa"
    cv2.rectangle(img_copy, (int(avg_x), int(avg_y)),
                  (int(avg_x_end), int(avg_y_end)), colors[0], 2)


    end = time.time()
    cv2.imshow("Regional proposal object detection", img_copy)
    cv2.waitKey(0)


    print("[INFO] selective search took {:.4f} seconds".format(end - start))
    print("[INFO] {} total region proposals".format(len(rects)))

ObjectDetector("Screenshot_20221214-125535(1).jpg")