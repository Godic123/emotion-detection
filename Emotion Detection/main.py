import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from deepface import DeepFace
from collections import Counter
from collections import deque


def main():
    # img = cv2.imread('sadman.jpg')
    # predictions = DeepFace.analyze(img)
    # print(predictions['dominant_emotion'])
    # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(gray,1.1,4)

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.putText(img, predictions['dominant_emotion'], (0, 50), font , 1, (0, 0, 255), 2, cv2.LINE_4)
    # plt.show()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    queue = deque(["neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral", "neutral"])
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret,frame = cap.read()
        result = DeepFace.analyze(frame, actions = ['emotion'], enforce_detection=False)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = faceCascade.detectMultiScale(gray,1.1,4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame, result['dominant_emotion'], (50, 50), font , 3, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow('Original video', frame)

        queue.popleft()
        queue.append(result['dominant_emotion'])
        a = dict(Counter(queue))
        print(a)
        
        if a.get('sad') == None or a.get('sad') >= 7:
            print("True")



        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




main()
