import numpy as np
import cv2
import os


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

cap = cv2.VideoCapture(0)
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

if (len(dir) != 0):
    face_dataset = np.concatenate(face_data, axis=0)
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    print(face_labels.shape)
    print(face_dataset.shape)

    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    print(trainset.shape)

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, recframe = cap.read()
        if ret == False:
            continue
        # Convert frame to grayscale
        gray = cv2.cvtColor(recframe, cv2.COLOR_BGR2GRAY)

        # Detect multi faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for face in faces:
            x, y, w, h = face

            # Get the face ROI
            offset = 5
            face_section = recframe[y - offset:y + h + offset, x - offset:x + w + offset]
            face_section = cv2.resize(face_section, (100, 100))

            out = knn(trainset, face_section.flatten())

            # Draw rectangle in the original image
            cv2.putText(recframe, names[int(out)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(recframe, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Faces", recframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    NUM = 0
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    while True:
        ret, recframe = cap.read()
        NUM = NUM + 1

        gray_frame = cv2.cvtColor(recframe, cv2.COLOR_BGR2GRAY)

        if ret == False:
            continue

        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) == 0:
            continue

        for face in faces[:1]:
            x, y, w, h = face

            offset = 10
            face_offset = recframe[y - offset:y + h + offset, x - offset:x + w + offset]
            face_selection = cv2.resize(face_offset, (100, 100))
            cv2.putText(recframe, "UNREGISTERED DRIVER", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face", face_selection)
            cv2.rectangle(recframe, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("faces", recframe)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


