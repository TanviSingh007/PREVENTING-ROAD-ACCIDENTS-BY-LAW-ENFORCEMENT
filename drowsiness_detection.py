# Import the necessary packages
import datetime as dt
from EAR_calculator import *
from imutils import face_utils
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib import style
import imutils
import dlib
import time
import argparse
import cv2
from playsound import playsound
from scipy.spatial import distance as dist
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime

style.use('fivethirtyeight')

def Slope(a, b, c, d):
    if (c-a!=0):
        return (d - b) / (c - a)
# Creating the dataset
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

########## KNN CODE ############
def distance(v1, v2):
    # Eucledian
    return np.sqrt(((v1 - v2) ** 2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])
    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)
    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


################################



# all eye  and mouth aspect ratio with time
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []
# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1)
args = vars(ap.parse_args())

# Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink
EAR_THRESHOLD = 0.2
# Declare another costant to hold the consecutive number of frames to consider for a blink
CONSECUTIVE_FRAMES = 20
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 16

# Initialize two counters
BLINK_COUNT = 0
FRAME_COUNT = 0

# Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO]Loading the predictor.....")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Grab the indexes of the facial landamarks for the left and right eye respectively
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream and allow the camera to warm-up
print("[INFO]Loading Camera.....")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2)

assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0
count_seatbelt=0
count_unregistered_driver=1
# Now, loop over all the frames and detect the faces
first=0;
#cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}
face_recognition_frames = 0

# Dataset prepration
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

dir = os.listdir('./face_dataset/')


if (len(dir) != 1):
    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    print(face_labels.shape)
    print(face_dataset.shape)
    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    print(trainset.shape)
    font = cv2.FONT_HERSHEY_SIMPLEX
else:
    naamjochapega="UNREGISTERED DRIVER (NOT IN RECORDS)"


check=False
while True:
    beltframe = vs.read()
    frame = vs.read()
    beltframe = imutils.resize(beltframe, height=450)
    beltgray = cv2.cvtColor(beltframe, cv2.COLOR_BGR2GRAY)
    belt = False
    blur = cv2.blur(beltgray, (1, 1))
    edges = cv2.Canny(blur, 50, 400)
    recframe = vs.read()

        # Convert frame to grayscale
    recgray = cv2.cvtColor(recframe, cv2.COLOR_BGR2GRAY)

    # Detect multi faces in the image
    faces = face_cascade.detectMultiScale(recgray, 1.3, 5)
    if (len(dir) != 1):
        for face in faces:
            x, y, w, h = face

            # Get the face ROI
            offset = 5
            face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
            face_section = cv2.resize(face_section, (100, 100))

            outtt = knn(trainset, face_section.flatten())
            if(names[int(outtt)]!=""):
                cv2.putText(frame, names[int(outtt)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            else:
                cv2.putText(frame, "UNRECOGNIZED DRIVER ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            naamjochapega=names[int(outtt)]
            check=False
    else :
        naamjochapega="UNRECOGNIZED UNREGISTERED DRIVER"
        if check==False :
          cv2.imwrite("UNREGISTERED DRIVERS/unregistered_driver%d.jpg" % count_unregistered_driver, recframe)
          check=True
          count_unregistered_driver +=1


        # Draw rectangle in the original image




#
    # # Previous Line Slope
    ps = 0
    #
    # # Previous Line Co-ordinates
    px1, py1, px2, py2 = 0, 0, 0, 0
    #
    # # Extracting Lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 270, 30, maxLineGap=20, minLineLength=170)

    if lines is not None:

        # Loop line by line
        for line in lines:

            # Co-ordinates Of Current Line
            x1, y1, x2, y2 = line[0]

            # Slope Of Current Line
            if (x2 - x1 != 0):
                s = abs(Slope(x1, y1, x2, y2))

            # If Current Line's Slope Is Greater Than 0.7 And Less Than 2

            if ((s > 0.7) and (s < 2)):

                # And Previous Line's Slope Is Within 0.7 To 2
                if ((abs(ps) > 0.7) and (abs(ps) < 2)):

                    # And Both The Lines Are Not Too Far From Each Other
                    if (((abs(x1 - px1) > 5) and (abs(x2 - px2) > 5)) or ((abs(y1 - py1) > 5) and (abs(y2 - py2) > 5))):
                        # Plot The Lines On "beltframe"
                        cv2.line(beltframe, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.line(beltframe, (px1, py1), (px2, py2), (0, 0, 255), 3)

                        # Belt Is Detected
                        print("Belt Detected")
                        cv2.putText(frame, "SEATBELT DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 0, 255), 3)
                        count_seatbelt = count_seatbelt + 1
                        cv2.imwrite("dataset/seatbelt_DETECTED%d.jpg" % count_seatbelt, beltframe)
                        belt = True

            # Otherwise Current Slope Becomes Previous Slope (ps) And Current Line Becomes Previous Line (px1, py1, px2, py2)
            ps = s
            px1, py1, px2, py2 = line[0]

    if belt == False:
        print("No Seatbelt detected")
        cv2.putText(frame, "No Seatbelt detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 3)
        if(first == 0):
            #playsound('sound files/alarm.mp3')
            #cv2.imshow("Output", frame)
            first+=1
    # cv2.imshow("Output", beltframe)
    # key = cv2.waitKey(1) & 0xFF
    #
    # if key == ord('q'):
    #     break
    # Extract a frame

    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    # Resize the frame
    frame = imutils.resize(frame, width=500)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    rects = detector(frame, 1)

    # Now loop over all the face detections and apply the predictor
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array
        shape = face_utils.shape_to_np(shape)

        # Draw a rectangle over the detected face
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put a number
        cv2.putText(frame, naamjochapega, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        mouth = shape[mstart:mend]
        # Compute the EAR for both the eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Take the average of both the EAR
        EAR = (leftEAR + rightEAR) / 2.0
        # live datawrite in csv
        ear_list.append(EAR)
        # print(ear_list)

        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # Draw the contours
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR / 10)
        # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place
        # Thus, count the number of frames for which the eye remains closed
        if EAR < EAR_THRESHOLD:
            FRAME_COUNT += 1

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                count_sleep += 1
                # Add the frame to the dataset ar a proof of drowsy driving
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                playsound('sound files/alarm.mp3')
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                playsound('sound files/warning.mp3')
            FRAME_COUNT = 0
        # cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if the person is yawning
        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add the frame to the dataset ar a proof of drowsy driving
            cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
            playsound('sound files/alarm.mp3')
            playsound('sound files/warning_yawn.mp3')
    # total data collection for plotting
    for i in ear_list:
        total_ear.append(i)
    for i in mar_list:
        total_mar.append(i)
    for i in ts:
        total_ts.append(i)
    # display the frame

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

a = total_ear
b = total_mar
c = total_ts

df = pd.DataFrame({"EAR": a, "MAR": b, "TIME": c})
df.to_csv("op_webcam.csv", index=False)
df = pd.read_csv("op_webcam.csv")
check=False
df.plot(x='TIME', y=['EAR', 'MAR'])
# plt.xticks(rotation=45, ha='right')

plt.subplots_adjust(bottom=0.30)
plt.title('EAR & MAR calculation over time of webcam')
plt.ylabel('EAR & MAR')
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
cv2.destroyAllWindows()
vs.stop()