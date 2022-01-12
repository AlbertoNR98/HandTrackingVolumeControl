"""
VolumeHandControl - Control de volumen de Windows mediante gestos
Dependencias:
    - Python 3.7 (Conda)
    - opencv-python 4.5.1.48
    - mediapipe 0.8.3.1 - Modelos de machine learning
    - pycaw 20181226 - Control de volumen de Windows
    - HandTrackingModule.py
"""
import cv2
import time
import numpy as np
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inicializacion de la camara y medida de FPS
cap = cv2.VideoCapture(0)   # Si webcam integrada
# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)    # Si webcam externa
if not cap.isOpened():
    raise IOError("Webcam no encontrada")
wCam, hCam = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)  # propId = 3
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)    # propId = 4
pTime = 0

# Objeto de HandTrackingModule
detector = htm.handDetector(detectionCon=0.7, maxHands=1)

# Inicializacion del modulo de volumen (Windows)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Parametros de volumen
volRange = volume.GetVolumeRange()  # Rango de volumen del SO
minVol = volRange[0]
maxVol = volRange[1]
vol = 0  # Valor de volumen
volBar = 400  # Tamaño de la barra en la que se puede ver el volumen que se va a establecer
volPer = 0  # Porcentaje a establecer
colorVol = (255, 0, 0)  # Cuando se establece un nuevo valor, cambia de color momentaneamente

area = 0    # Bounding box

while True:
    success, img = cap.read()

    # Busca una mano en el fotograma
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        # Filtra segun el tamaño de la mano (depende de la cercania a la camara)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
        if 250 < area < 1000:
            # Distancia entre las puntas del pulgar y el indice
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convierte la distancia anterior a valores de volumen
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])
            # Reduce la resolucion del porcentaje de volumen que se puede obtener para suavizarlo
            smoothness = 10
            volPer = smoothness * round(volPer/smoothness)

            # Comprueba cuantos dedos de la mano estan estirados
            fingers = detector.fingersUp()
            # Si el pulgar no lo esta, establece el volumen
            if not fingers[4]:
                volSet = volPer / 100
                if volSet > 0:
                    volume.SetMute(False, None)
                    volume.SetMasterVolumeLevelScalar(volSet, None)
                else:
                    volume.SetMasterVolumeLevelScalar(volSet, None)
                    volume.SetMute(True, None)

                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                colorVol = (0, 255, 0)
            else:
                colorVol = (255, 0, 0)

    # Graficos
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'Vol set: {int(cVol)} %', (390, 50), cv2.FONT_HERSHEY_DUPLEX, 1, colorVol, 2)

    # Frame rate
    cTime = time.time()
    try:
        fps = 1 / (cTime - pTime)
    except ZeroDivisionError:
        fps = 0
    pTime = cTime

    cv2.putText(img, f'{str(int(fps))} FPS', (40, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Captura", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

