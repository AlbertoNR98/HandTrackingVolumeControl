"""
HandTrackingModule - Métodos de hand tracking basados en mediapipe (https://mediapipe.dev/)
Dependencias:
    - Python 3.7 (Conda)
    - opencv-python 4.5.1.48
    - mediapipe 0.8.3.1
    - pycaw 20181226
"""
import cv2
import mediapipe as mp
import time
import math


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.results = None
        self.lmList = None
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands  # Importa el modelo ML de manos de mediapipe
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # ID de los puntos relacionados con las puntas de los dedos

    def findHands(self, img, draw=True):  # Busca manos en el fotograma y dibuja los puntos clave y sus conexiones
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:  # Puntos clave = landmarks
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):  # Devuelve las coordenadas de los puntos (en px) y del marco que
        # rodea la mano
        xList = []
        yList = []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):    # Detecta que dedos estan estirados
        fingers = []
        # Pulgar
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Resto de dedos
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):     # Distancia entre dos puntos. Se da su ID
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    wCam, hCam = 640, 480
    cap = cv2.VideoCapture(0)  # Si webcam integrada
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)    # Si webcam externa
    if not cap.isOpened():
        raise IOError("Webcam no encontrada")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)  # propId = 3
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)  # propId = 4

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmL, _ = detector.findPosition(img)
        if len(lmL) != 0:
            print(lmL[4])  # Indice = numero de landmark

        cTime = time.time()
        fps = 1 / (cTime - pTime)  # Ojo con division entre 0
        pTime = cTime
        # fps = cap.get(cv2.CAP_PROP_FPS)   # Frame rate fijo
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

        cv2.imshow("Captura", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
