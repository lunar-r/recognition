import os
import cv2
import sys
import face_recognition
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from deep.dnn import DeepFace


class ResThread(QThread):
    signal = pyqtSignal(str)

    def __init__(self):
        super(ResThread, self).__init__()
        self.img = None
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
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            res = res + name + ","
        self.signal.emit(str(res))


class Assistant(QObject):
    signal = pyqtSignal(str)

    def run(self, res):
        self.signal.emit(str(res))


class FaceThread(QRunnable):
    def __init__(self):
        super(FaceThread, self).__init__()
        self.img = None
        self.model = None
        self.worker = None
        self.classifier = None
        self.class_method = None
        self.helper = Assistant()

    def __del__(self):
        print("Thread is deleted...")

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


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.thread_num = 0
        self.thread_fin = 0
        self.time_camera = QTimer()
        self.time_video = QTimer()
        self.time_flash = QTimer()
        self.cap = cv2.VideoCapture()
        self.worker = DeepFace()
        self.model = "Inception"
        self.class_method = "KNeighborsClassifier"
        self.init_data()
        self.init_ui()
        self.init_slot()

    def init_data(self):
        self.frame = None
        self.thd = None
        self.pool = QThreadPool.globalInstance()
        self.img_state = False
        self.infos = ["KNeighborsClassifier", "DecisionTreeClassifier",
                 "RandomForestClassifier", "LinearSVC"]

        self.img_url = "source/images/"
        harr_filepath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.classifier = cv2.CascadeClassifier(harr_filepath)  # 加载人脸特征分类器
        self.load_known(self.img_url)
        self.worker.pre_train(self.img_url)

    def load_known(self, root_dir):
        if root_dir is None:
            return
        names = []
        encodings = []
        for name in os.listdir(root_dir):
            names.append(name)
            for file in os.listdir(os.path.join(root_dir, name)):
                img = face_recognition.load_image_file(root_dir + "/"+ name + "/"+ file)
                encoding = face_recognition.face_encodings(img)[0]
                encodings.append(encoding)
                break
        self.known_names = names
        self.known_encodings = encodings
        print("Known data are loaded...")

    def init_slot(self):
        self.time_camera.timeout.connect(self.show_camera)
        self.time_video.timeout.connect(self.show_video)
        self.time_flash.timeout.connect(self.start_face)

        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_face_reco.clicked.connect(self.open_face)

        self.btn_save_img.clicked.connect(self.save_img)
        self.btn_quit.clicked.connect(self.close)

        self.combox_classifier.activated[str].connect(self.select_classifier)
        self.combox_model.activated[str].connect(self.select_model)

    def init_ui(self):

        left_box = QWidget()
        right_box = QWidget()
        left_in_box = QWidget()

        # left
        self.label_img_1 = QLabel()
        self.label_img_2 = QLabel()
        self.label_img_3 = QLabel()
        self.label_img_4 = QLabel()
        self.name_img_1 = QLabel("Unknown", self)
        self.name_img_2 = QLabel("Unknown", self)
        self.name_img_3 = QLabel("Unknown", self)
        self.name_img_4 = QLabel("Unknown", self)

        self.label_img = [self.label_img_1, self.label_img_2, self.label_img_3, self.label_img_4]
        self.name_infos = [self.name_img_1, self.name_img_2, self.name_img_3, self.name_img_4]
        self.label_video = QLabel()
        self.label_video.setFixedSize(641, 481)
        self.label_video.setAutoFillBackground(False)

        #right
        self.btn_img = QPushButton("打开图片")
        self.btn_camera = QPushButton("打开相机")
        self.btn_video = QPushButton("打开视频")
        self.btn_save_img = QPushButton("保存照片")
        self.btn_model = QPushButton("选择识别模型")
        self.btn_face_reco = QPushButton("人脸识别")
        self.btn_quit = QPushButton("退出程序")
        self.combox_classifier = QComboBox(self)
        self.combox_classifier.addItems(self.infos)
        self.combox_model = QComboBox(self)
        self.combox_model.addItems(["Inception", "ResNet"])

        combox_classifier_layer = QHBoxLayout()
        combox_classifier_layer.addWidget(QLabel("Classifier Selection:", self))
        combox_classifier_layer.addWidget(self.combox_classifier)
        combox_classifier_unit = QWidget()
        combox_classifier_unit.setLayout(combox_classifier_layer)

        combox_model_layer = QHBoxLayout()
        combox_model_layer.addWidget(QLabel("Model Selection:", self))
        combox_model_layer.addWidget(self.combox_model)
        combox_model_unit = QWidget()
        combox_model_unit.setLayout(combox_model_layer)

        face_detect_show_layer = QHBoxLayout()
        face_detect_show_layer.addWidget(self.label_img_1)
        face_detect_show_layer.addWidget(self.label_img_2)
        face_detect_show_layer.addWidget(self.label_img_3)
        face_detect_show_layer.addWidget(self.label_img_4)
        face_detect_show = QWidget()
        face_detect_show.setLayout(face_detect_show_layer)

        face_info_show_layer = QHBoxLayout()
        face_info_show_layer.addWidget(self.name_img_1)
        face_info_show_layer.addWidget(self.name_img_2)
        face_info_show_layer.addWidget(self.name_img_3)
        face_info_show_layer.addWidget(self.name_img_4)
        face_info_show = QWidget()
        face_info_show.setLayout(face_info_show_layer)

        left_in_layer = QVBoxLayout()
        left_in_layer.addWidget(face_detect_show)
        left_in_layer.addWidget(face_info_show)
        left_in_box.setLayout(left_in_layer)

        left_layer = QVBoxLayout()
        left_layer.addWidget(left_in_box)
        left_layer.addWidget(self.label_video)
        left_box.setLayout(left_layer)

        right_layer = QVBoxLayout()
        right_layer.addWidget(self.btn_img)
        right_layer.addWidget(self.btn_camera)
        right_layer.addWidget(self.btn_video)
        right_layer.addWidget(self.btn_save_img)
        right_layer.addWidget(combox_classifier_unit)
        right_layer.addWidget(combox_model_unit)
        right_layer.addWidget(self.btn_face_reco)
        right_layer.addWidget(self.btn_quit)
        right_layer.addStretch(1)
        right_box.setLayout(right_layer)

        global_layer = QHBoxLayout()
        global_layer.addWidget(left_box)
        global_layer.addWidget(right_box)

        self.setLayout(global_layer)
        self.resize(1024, 768)
        self.center()
        self.setWindowTitle("FaceSystem")
        self.setWindowIcon(QIcon("./source/img/apple.png"))
        self.show()

    def set_names(self, data):
        names = list(filter(lambda x: x != '', data.split(",")))
        print(names)
        for i, name in enumerate(names):
            if i < 4:
                # 展示识别到的信息即可
                self.set_name(i, name)
                self.set_label(i, name)

    def set_name(self, idx, data):
        self.name_infos[idx].setText(data)

    def set_label(self, idx, name):
        img = None
        path = self.img_url + "/" + name
        for file in os.listdir(path):
            img = QPixmap(file)
            break
        self.label_img[idx].setPixmap(img)

    def start_face(self):
        if self.frame is not None:
            if self.model == "Inception":
                self.thd = FaceThread()
                self.thread_num = self.thread_num + 1
                print("create new thread : " + str(self.thread_num))
                self.thd.set_img(self.frame)
                self.thd.set_worker(self.worker)
                self.thd.set_classifier(self.class_method)
                self.thd.helper.signal.connect(self.set_names)
            else:
                self.thd = ResThread()
                self.thd.set_img(self.frame)
                self.thd.set_dict(self.known_names, self.known_encodings)
                self.thd.signal.connect(self.set_names)
            self.pool.start(self.thd)

    def open_face(self):
        if self.time_flash.isActive() is False:
                self.time_flash.start(2000)
                self.btn_face_reco.setText(u"关闭人脸识别")
        else:
            self.time_flash.stop()
            self.btn_face_reco.setText(u"打开人脸识别")

    def open_camera(self):
        if self.time_camera.isActive() is False:
            flag = self.cap.open(0)
            if flag is False:
                QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
            else:
                self.time_camera.start(30)
                self.btn_camera.setText("关闭相机")
        else:
            self.time_camera.stop()
            self.cap.release()
            self.label_video.clear()
            self.btn_camera.setText(u'打开相机')

    def open_video(self):
        if self.time_video.isActive() is False:
            fname, type = QFileDialog.getOpenFileName(self, 'Open File', "D:/UserInfo/workDir/face/face_system", "Video Files (*.mp4 *.avi)")
            flag = self.cap.open(fname)
            if flag is False:
                # 使用u， 使得中文以 unicode格式存储，保证不出现乱码
                QMessageBox.warning(self, u"Warning", u"请检测视频是否损坏", buttons=QMessageBox.Ok,  defaultButton=QMessageBox.Ok)
            else:
                self.time_video.start(30)
                self.btn_video.setText("关闭视频")
        else:
            self.time_video.stop()
            self.cap.release()
            self.label_video.clear()
            self.btn_video.setText(u'打开视频')

    def show_camera(self):
        ret, self.frame = self.cap.read()
        show = cv2.resize(self.frame, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
        # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(self.showImage))

    def show_video(self):
        ret, self.frame = self.cap.read()
        show = cv2.resize(self.frame, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray_image, 1.3, 5)  # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
        for (x, y, w, h) in faces:
            cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出人脸
        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(self.showImage))

    def save_img(self):
        self.time_camera.stop()
        self.time_video.stop()
        name, state = QInputDialog.getText(self, "人名", "请输入保存者名称", QLineEdit.Normal, "Liming")
        if name is "LiMing" or state is False:
            return
        img_path = os.path.join(self.img_url + name)
        folder = os.path.exists(img_path)
        if not folder:
            os.mkdir(img_path)
        img_path = img_path + "/"
        cur = 10001
        while os.path.isfile(img_path + name + "_" + str(cur) + ".jpg"):
            cur = cur + 1
        path = img_path + name + "_" + str(cur) + ".jpg"
        self.showImage.save(path)
        self.time_video.start(30)
        self.time_camera.start(30)

    def select_classifier(self, classifier):
        self.class_method = classifier
        print("Select classifier : " + classifier)

    def select_model(self, model):
        self.model = model
        print("Change model to : " + model)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width()-size.width()) / 2, (screen.height()-size.height()) / 2)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, QCloseEvent):
        reply = QMessageBox.question(self, u"确认退出", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.cap.release()
            self.time_camera.stop()
            self.time_video.stop()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    block = MainUI()
    sys.exit(app.exec_())


    # def open_img(self):
    #     if self.img_state:
    #         self.btn_img.setText(u"打开图片")
    #     else:
    #         self.fname, type = QFileDialog.getOpenFileName(self, 'Open File', "D:/UserInfo/GitPro/TensorFlow-Examples/face_system/images/", "Image Files (*.jpg *.png)")
    #         img = QPixmap(self.fname)
    #         self.btn_img.setText(u"关闭图片")
    #     self.img_state = not self.img_state