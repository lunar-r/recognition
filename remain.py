import os
import cv2
import sys
import face_recognition
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from deep.dnn import DeepFace
from second import Ui_MainWindow


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.time_camera = QTimer()
        self.time_video = QTimer()
        self.time_flash = QTimer()
        self.cap = cv2.VideoCapture()
        self.worker = DeepFace()
        self.model = "Inception"
        self.class_method = "KNeighborsClassifier"
        self.known_names = None
        self.known_encodings = None
        self.frame = None
        self.thd = None
        self.pool = QThreadPool.globalInstance()
        self.img_state = False
        self.infos = ["KNeighborsClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "LinearSVC"]
        self.img_url = "source/images/"
        self.harr_filepath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(self.harr_filepath)  # 加载人脸特征分类器

        self.load_known(self.img_url)
        self.worker.pre_train(self.img_url)

        self.set_ui(self)
        self.setWindowTitle("FaceSystem")
        self.setWindowIcon(QIcon("./source/logos/apple.png"))
        self.show()

    def set_slot(self):
        self.time_camera.timeout.connect(self.show_camera)
        self.time_video.timeout.connect(self.show_video)
        self.time_flash.timeout.connect(self.start_face)

        self.btn_face_dect.clicked.connect(self.face_detect)
        self.btn_face_reco.clicked.connect(self.face_reco)
        self.btn_face_anal.clicked.connect(self.face_analysis)

        self.btn_data_anal.clicked.connect(self.data_anal)
        self.btn_data_list.clicked.connect(self.data_list)

        self.btn_start_set.clicked.connect(self.start_set)
        self.btn_data_set.clicked.connect(self.data_set)

        self.btn_open_camera.clicked.connect(self.open_camera)
        self.btn_open_video.clicked.connect(self.open_video)
        self.btn_save_img.clicked.connect(self.save_img)

        self.btn_select_algo.clicked.connect(self.select_algo)
        self.btn_select_class.clicked.connect(self.select_classifier)

    def load_known(self, root_dir):
        if root_dir is None:
            return
        names = []
        encodings = []
        print(root_dir)
        for name in os.listdir(root_dir):
            names.append(name)
            for file in os.listdir(os.path.join(root_dir, name)):
                img = face_recognition.load_image_file(root_dir + "/" + name + "/" + file)
                temp = face_recognition.face_encodings(img)
                if len(temp) == 0:
                    print("face encoding is not exist, img error, exit.")
                    sys.exit(0)
                encoding = temp[0]
                encodings.append(encoding)
                break
        self.known_names = names
        self.known_encodings = encodings
        print("Known data are loaded (for resNet)...")

    def set_names(self, data):
        names = list(filter(lambda x: x != '', data.split(",")))
        print(names)
        for i, name in enumerate(names):
            if i < 4:
                # 展示识别到的信息即可
                if str.lower(name) != "unknown":
                    print(type(name))
                    print(name)
                    self.set_name(i, name)
                    self.set_label(i, name)

    # to fix
    def set_name(self, idx, data):
        self.name_infos[idx].setText(data)

    # to fix
    def set_label(self, idx, name):
        img = None
        print("in set label func, name: " + name)
        path = self.img_url + name
        for file in os.listdir(path):
            file = path + "/" + file
            print(file)
            img = QPixmap(file)
            break
        print(img.height())
        self.label_img[idx].setPixmap(img)

    def select_classifier(self, classifier):
        self.class_method = classifier
        print("Select classifier : " + classifier)

    def select_model(self, model):
        self.model = model
        print("Change model to : " + model)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MyWindow()
    sys.exit(app.exec_())


class Assistant(QObject):
    signal = pyqtSignal(str)

    def run(self, res):
        self.signal.emit(str(res))


class ResThread(QRunnable):
    def __init__(self):
        super(ResThread, self).__init__()
        self.img = None
        self.helper = Assistant()
        self.known_face_names = None
        self.known_face_encodings = None

    def set_img(self, img):
        self.img = img

    def set_dict(self, known_names, known_encodings):
        self.known_face_names = known_names
        self.known_face_encodings = known_encodings

    def run(self):
        face_locations = face_recognition.face_locations(self.img)
        face_encodings = face_recognition.face_encodings(self.img, face_locations)
        res = ""
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            res = res + name + ","
        self.helper.run(res)


class FaceThread(QRunnable):
    def __init__(self):
        super(FaceThread, self).__init__()
        self.img = None
        self.model = None
        self.worker = None
        self.classifier = None
        self.class_method = None
        self.helper = Assistant()

    def set_img(self, img):
        self.img = img

    def set_worker(self, worker):
        self.worker = worker

    def set_classifier(self, classifier):
        self.classifier = classifier

    def run(self):
        self.worker.train(self.classifier)
        res = self.worker.predict(self.img)
        self.helper.run(res)