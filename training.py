import cv2
import os
import numpy as np
from PIL import Image
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def getImagesAndLabels():
    # get the  of all the files in the folder
    images = [os..join(, f) for f in os.listdir()]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image s and loading the Ids and the images
    for image in images:
        # loading the image and converting it to gray scale
        pilImage = Image.open(image).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os..split(image)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids


faces, Ids = getImagesAndLabels('TrainingImage')
recognizer.train(faces, np.array(Ids))
recognizer.save('TrainingImageLabel/trainner.yml')
