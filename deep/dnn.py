from model import create_model
from align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import numpy as np
import os.path
import cv2
import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import warnings
# Suppress LabelEncoder warning
warnings.filterwarnings('ignore')

class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file
        # recognize target
        self.target = None
    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_image(path):
    img = cv2.imread(path, 1)
    return img[...,::-1]


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/png' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.png':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

class DeepFace:
    def __init__(self):
        self.pre_model = create_model()
        self.pre_model.load_weights('./deep/weights/nn4.small2.v1.h5')
        self.alignment = AlignDlib('./deep/models/landmarks.dat')
        self.metadata = None
        self.embedded = None
        self.classifier = None

    def pre_train(self, path):
        self.metadata = load_metadata(path)
        self.embedded = np.zeros((self.metadata.shape[0], 128))

        for i, m in enumerate(self.metadata):
            img = load_image(m.image_path())
            img = self.align_image(img)
            if img is None:
                continue
            # scale RGB values to interval [0,1]
            img = (img / 255.).astype(np.float32)
            # obtain embedding vector for image
            self.embedded[i] = self.pre_model.predict(np.expand_dims(img, axis=0))[0]

        targets = np.array([m.name for m in self.metadata])

        #   LabelEncoder可以将标签分配一个0—n_classes - 1 之间的编码
        self.encoder = LabelEncoder()
        self.encoder.fit(targets)
        # 将各种标签分配一个可数的连续编号
        self.y = self.encoder.transform(targets)
        print("inception net pre train finish ...")

    def align_image(self, img):
        return self.alignment.align(96, img, self.alignment.getLargestFaceBoundingBox(img),
                               landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

    def train(self, classifier=LinearSVC):
        X_train = self.embedded
        y_train = self.y
        print("use Method" + str(classifier) + "to train model.")

        self.classifier = globals()[classifier]()
        self.classifier.fit(X_train, y_train)

    def predict(self, img):
        img = self.align_image(img)
        if img is None:
            return str("UnKnown")
        img = (img / 255.).astype(np.float32)
        one_embedded = self.pre_model.predict(np.expand_dims(img, axis=0))[0]
        example_prediction = self.classifier.predict([one_embedded])
        print(example_prediction)
        example_identity = self.encoder.inverse_transform(example_prediction)[0]

        print("the result of recognition: " + str(example_identity))
        return str(example_identity)

def main(argv=None):
    work = DeepFace()
    img = load_image("people.jpg")
    ans = work.align_image(img)
    cv2.imshow("img", img)
    cv2.imshow("ans", ans)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
