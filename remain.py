import os
import cv2
import sys
import time
import copy
import json
import logging
import datetime
import face_recognition
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from deep.dnn import DeepFace
from second_ui import Ui_MainWindow


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
        self.logger = None

    def set_img(self, img):
        self.img = img

    def set_dict(self, known_names, known_encodings):
        self.known_face_names = known_names
        self.known_face_encodings = known_encodings

    def set_logger(self, logger):
        self.logger = logger

    def run(self):
        begin = datetime.datetime.now()
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
        end = datetime.datetime.now()
        self.logger.info("face recognition in resnet is %s", res)
        self.logger.debug("face recognition in resnet use  %s  seconds ...", str(end - begin))


class FaceThread(QRunnable):
    def __init__(self):
        super(FaceThread, self).__init__()
        self.img = None
        self.model = None
        self.worker = None
        self.classifier = None
        self.class_method = None
        self.logger = None
        self.helper = Assistant()

    def set_img(self, img):
        self.img = img

    def set_worker(self, worker):
        self.worker = worker

    def set_logger(self, logger):
        self.logger = logger

    def set_classifier(self, classifier):
        self.classifier = classifier

    def run(self):
        self.worker.train(self.classifier)
        start = datetime.datetime.now()
        res = self.worker.predict(self.img)
        end = datetime.datetime.now()
        self.helper.run(res)
        self.logger.info("face recognition in inception is %s", res)
        self.logger.debug("face recognition in inception net use  %s  seconds ...", str(end - start))


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        first = datetime.datetime.now()
        super(MyWindow, self).__init__()
        self.logger = logging.getLogger("global_logger")
        self.print_handler = None
        self.file_handler = None
        self.time_camera = QTimer()
        self.time_video = QTimer()
        self.time_flash = QTimer()
        self.cap = cv2.VideoCapture()
        self.worker = DeepFace()
        self.model = "Inception"
        self.class_method = "KNeighborsClassifier"
        self.known_names = None
        self.known_encodings = None
        self.thd = None
        self.show_img = None
        self.frame = None
        self.showImage = None
        self.pool = QThreadPool.globalInstance()
        self.img_state = False
        self.face_detect_state = False
        self.img_url = "source/images/"
        self.infos = ["KNeighborsClassifier", "DecisionTreeClassifier", "RandomForestClassifier", "LinearSVC"]
        self.harr_filepath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        # 加载人脸特征分类器
        self.classifier = cv2.CascadeClassifier(self.harr_filepath)

        self.pre_train_flag = True
        self.pre_known_flag = True

        with open("status.json", "r") as f:
            flags = json.load(f)
            self.pre_train_flag = flags["pre_train_flag"]
            self.pre_known_flag = flags["pre_known_flag"]

        self.set_ui(self)
        self.set_log()

        if self.pre_known_flag is True:
            self.load_known(self.img_url)
        else:
            self.label_model_info.setText("known data for Resnet not be loaded.")

        if self.pre_train_flag is True:
            f = datetime.datetime.now()
            self.worker.pre_train(self.img_url)
            s = datetime.datetime.now()
            self.logger.debug("pre train of inception use %s seconds", str(s - f))
        else:
            self.label_accuracy_info.setText("pre train pof inception not be executed.")

        menu = QMenu()
        menu.addAction("Inception", self.select_model_ince)
        menu.addAction("ResNet", self.select_model_res)
        self.btn_select_algo.setMenu(menu)

        menu = QMenu()
        menu.addAction("KNN", self.select_class_knn)
        menu.addAction("DTree", self.select_class_dtree)
        menu.addAction("RandomForest", self.select_class_rf)
        menu.addAction("LinearRegression", self.select_class_lr)
        self.btn_select_class.setMenu(menu)

        self.offset = 0
        self.idx = 5
        self.set_data()
        self.set_slot()
        self.setWindowTitle("FaceSystem")
        self.setWindowIcon(QIcon("./source/logos/apple.png"))
        self.qss = self.read_qss('./source/stylesheet/main.qss')
        self.setStyleSheet(self.qss)

        second = datetime.datetime.now()
        self.logger.info("start program use %s seconds ...", str(second - first))
        self.show()

    def set_log(self):
        self.logger.setLevel(logging.DEBUG)
        self.print_handler = logging.StreamHandler(sys.stderr)
        self.print_handler.setLevel(logging.DEBUG)
        self.print_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.file_handler = logging.FileHandler("face_info.log")
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger.addHandler(self.print_handler)
        self.logger.addHandler(self.file_handler)

    def select_model_ince(self):
        self.model = "Inception"
        self.label_accuracy_info.setText("95.70%")
        self.label_model_info.setText("Inception Net")
        self.logger.debug("select model: %s", self.model)

    def select_model_res(self):
        self.model = "ResNet"
        self.label_accuracy_info.setText("99.38%")
        self.label_model_info.setText("ResNet")
        self.logger.debug("select model: %s", self.model)

    def select_class_knn(self):
        self.class_method = self.infos[0]
        self.label_class_info.setText(self.infos[0])
        self.logger.debug("select classifier: %s", self.class_method)

    def select_class_dtree(self):
        self.class_method = self.infos[1]
        self.label_class_info.setText(self.infos[1])
        self.logger.debug("select classifier: %s", self.class_method)

    def select_class_rf(self):
        self.class_method = self.infos[2]
        self.label_class_info.setText(self.infos[2])
        self.logger.debug("select classifier: %s", self.class_method)

    def select_class_lr(self):
        self.class_method = self.infos[3]
        self.label_class_info.setText(self.infos[3])
        self.logger.debug("select classifier: %s", self.class_method)

    @staticmethod
    def read_qss(style):
        with open(style, 'r') as f:
            return f.read()

    def show_data(self, url, offset, idx):
        if url is None:
            return
        count = -1
        for name in os.listdir(url):
            count = count + 1
            if count < offset:
                continue
            if count >= offset + idx:
                break
            row = count - offset + 1

            item = QTableWidgetItem()
            item.setText(name)
            self.database_show.setItem(row, 0, item)
            sum = 0
            img_path = "./source/otherImg/people.jpg"
            for file in os.listdir(os.path.join(url, name)):
                img_path = url + "/" + name + "/" + file
                sum = sum + 1
            item = QTableWidgetItem()
            item.setText(str(sum))
            self.database_show.setItem(row, 1, item)

            item = QTableWidgetItem()
            time_info = time.ctime(os.stat(img_path).st_ctime)
            item.setText(time_info)
            self.database_show.setItem(row, 2, item)
            item = QLabel("")
            item.setAlignment(Qt.AlignCenter)
            item.setPixmap(QPixmap(img_path).scaled(60, 60))
            self.database_show.setCellWidget(row, 3, item)

    def set_data(self):
        self.img_infos = [self.label_img_info_1, self.label_img_info_2, self.label_img_info_3]
        self.img_shows = [self.label_img_show_1, self.label_img_show_2, self.label_img_show_3]
        for item in self.img_shows:
            item.setFixedSize(192, 128)
            item.setScaledContents(True)

        self.logger.debug("Finish set labels ..")
        self.show_data(self.img_url, self.offset, self.idx)
        self.logger.debug("Finish show data ...")

    def set_slot(self):
        self.time_camera.timeout.connect(self.show_camera)
        self.time_video.timeout.connect(self.show_video)
        self.time_flash.timeout.connect(self.show_face)

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

        self.btn_left_page.clicked.connect(self.turn_left_page)
        self.btn_right_page.clicked.connect(self.turn_right_page)

        self.btn_load_data.clicked.connect(self.warp_load_known)
        self.btn_pre_train.clicked.connect(self.warp_pre_train)

        self.checkBox_load_database.stateChanged.connect(self.update_load_state)
        self.checkBox_pretrain_model.stateChanged.connect(self.update_model_state)

    def update_load_state(self):
        cur = self.checkBox_load_database.isChecked()
        if cur is not self.pre_known_flag:
            self.pre_known_flag = not self.pre_known_flag
        self.update_json(self.pre_known_flag, self.pre_train_flag)

    def update_model_state(self):
        cur = self.checkBox_pretrain_model.isChecked()
        if cur is not self.pre_train_flag:
            self.pre_train_flag = not self.pre_train_flag
        self.update_json(self.pre_known_flag, self.pre_train_flag)

    def update_json(self, known, train):
        dict = {
            "pre_train_flag": train,
            "pre_known_flag": known
        }

        with open("status.json", "w") as f:
            json.dump(dict, f)
            self.logger.info("update init state of variables, pre_train: %s, known_data: %s", train, known)

    def warp_load_known(self):
        self.load_known(self.img_url)

    def warp_pre_train(self):
        first = datetime.datetime.now()
        self.worker.pre_train(self.img_url)
        second = datetime.datetime.now()
        self.logger.info("pre train for inception use %s seconds ..", str(second - first))

    def turn_left_page(self):
        self.logger.debug("turn to the left page ...")
        if self.offset < self.idx:
            self.logger.debug("at the first page ...")
        else:
            self.offset = self.offset - self.idx
            self.show_data(self.img_url, self.offset, self.idx)

    def turn_right_page(self):
        self.logger.debug("turn to the right page ...")
        self.offset = self.offset + self.idx
        self.show_data(self.img_url, self.offset, self.idx)

    def show_camera(self):
        ret, self.frame = self.cap.read()
        self.show_img = cv2.resize(self.frame, (512, 384))
        show = cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB)
        if self.face_detect_state:
            # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
            gray_image = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            faces = self.classifier.detectMultiScale(gray_image, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出人脸
        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(self.showImage))

    def show_video(self):
        ret, self.frame = self.cap.read()
        if ret is False:
            self.close_video()
        self.show_img = cv2.resize(self.frame, (512, 384))
        show = cv2.cvtColor(self.show_img, cv2.COLOR_BGR2RGB)
        if self.face_detect_state:
            # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
            gray_image = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
            faces = self.classifier.detectMultiScale(gray_image, 1.2, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出人脸
        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(self.showImage))

    def show_face(self):
        if self.frame is not None:
            if self.model == "Inception":
                self.thd = FaceThread()
                self.thd.set_img(self.frame)
                self.thd.set_logger(self.logger)
                self.thd.set_worker(self.worker)
                self.thd.set_classifier(self.class_method)
                self.thd.helper.signal.connect(self.set_names)
                self.logger.debug("start inception net thread(single)")
            else:
                self.thd = ResThread()
                temp = copy.copy(self.frame)
                temp = cv2.resize(temp, (0, 0), fx=0.3, fy=0.3)
                self.thd.set_img(temp)
                self.thd.set_logger(self.logger)
                self.thd.set_dict(self.known_names, self.known_encodings)
                self.thd.helper.signal.connect(self.set_names)
                self.logger.debug("start resnet thread(multi)")
            self.pool.start(self.thd)

    def face_detect(self):
        self.stackedWidget.setCurrentWidget(self.page_index)
        self.face_detect_state = not self.face_detect_state

    def face_reco(self):
        self.stackedWidget.setCurrentWidget(self.page_index)
        if self.time_flash.isActive() is False:
            self.time_flash.start(4000)
            self.btn_face_reco.setText(u"关闭人脸识别")
        else:
            self.time_flash.stop()
            self.btn_face_reco.setText(u"人脸识别")

    def face_analysis(self):
        self.stackedWidget.setCurrentWidget(self.page_face_anal)
        pass

    def data_list(self):
        self.stackedWidget.setCurrentWidget(self.page_data_show)

    def data_anal(self):
        self.stackedWidget.setCurrentWidget(self.page_logging)
        self.show_log("face_info.log")

    def start_set(self):
        self.stackedWidget.setCurrentWidget(self.page_setup)

    def data_set(self):
        self.stackedWidget.setCurrentWidget(self.page_setup)

    def show_log(self, path):
        with open(path, 'r') as f:
            context = f.read()
            print(context)
        self.text_logging.setText(context)

    def open_camera(self):
        if self.time_camera.isActive() is False:
            flag = self.cap.open(0)
            if flag is False:
                QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)
            else:
                self.time_camera.start(30)
                self.btn_open_camera.setText("关闭相机")
        else:
            self.close_source("camera")

    def open_video(self):
        if self.time_video.isActive() is False:
            fname, type = QFileDialog.getOpenFileName(self, 'Open File', "D:/UserInfo/workDir/face/face_system", "Video Files (*.mp4 *.avi)")
            flag = self.cap.open(fname)
            if flag is False:
                # 使用u， 使得中文以 unicode格式存储，保证不出现乱码
                QMessageBox.warning(self, u"Warning", u"请检测视频是否损坏", buttons=QMessageBox.Ok,  defaultButton=QMessageBox.Ok)
            else:
                self.time_video.start(30)
                self.btn_open_video.setText("关闭视频")
        else:
            self.close_source("video")

    def save_img(self):
        if self.show_img is None:
            return
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
        self.show_img.save(path)
        self.logger.info("save the person: %s in the path: %s", name, path)
        self.time_video.start(30)
        self.time_camera.start(30)

    def load_known(self, root_dir):
        if root_dir is None:
            return
        first = datetime.datetime.now()
        names = []
        encodings = []
        for name in os.listdir(root_dir):
            names.append(name)
            for file in os.listdir(os.path.join(root_dir, name)):
                img = face_recognition.load_image_file(root_dir + "/" + name + "/" + file)
                temp = face_recognition.face_encodings(img)
                if len(temp) == 0:
                    self.logger.debug("face encoding error in %s / %s", name, file)
                    continue

                encoding = temp[0]
                encodings.append(encoding)
                break

        self.known_names = names
        self.known_encodings = encodings
        second = datetime.datetime.now()
        self.logger.debug("upload known data for resnet use %s seconds ..", str(second - first))

    def set_names(self, data):
        names = list(filter(lambda x: x != '', data.split(",")))
        for i in range(3):
            self.img_shows[i].clear()
            self.img_infos[i].clear()
        for i, name in enumerate(names):
            if i < 3:
                # 展示识别到的信息即可
                if str.lower(name) != "unknown":
                    self.set_name(i, name)
                    self.set_label(i, name)

    def set_name(self, idx, data):
        self.img_infos[idx].setText(data)

    def set_label(self, idx, name):
        img = None
        path = self.img_url + name
        for file in os.listdir(path):
            file = path + "/" + file
            img = QPixmap(file)
            break
        self.img_shows[idx].setPixmap(img)

    def close_source(self, data):
        self.cap.release()
        self.label_video.clear()
        for i in range(3):
            self.img_infos[i].clear()
            self.img_shows[i].clear()

        self.time_flash.stop()
        self.btn_face_reco.setText(u"人脸识别")
        if data == "video":
            self.time_video.stop()
            self.btn_open_video.setText(u'打开视频')
        else:
            self.time_camera.stop()
            self.btn_open_camera.setText(u'打开摄像头')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MyWindow()
    sys.exit(app.exec_())





'''
    # 加载数据与模型用其他线程， error， 其他线程加载错误, 可以载入之后再进行加载。。。（done）
    # 3人检测，（done）
    # 数据库展示,Widget（Done）
    # 界面切换，使用stack进行顶部切换即可，， UI继续优化  （done）
    # CSS 界面， （todo）
'''