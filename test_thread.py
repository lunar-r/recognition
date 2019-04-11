from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import os
import cv2
# from deep.dnn import DeepFace


class Assistant(QObject):
    signal = pyqtSignal(str)

    def run(self, res):
        self.signal.emit(str(res))


class Work(QRunnable):
    def __init__(self):
        super(Work, self).__init__()
        self.helper = Assistant()

    def run(self):
        res = "Going"
        self.helper.run(res)
        print("emit signal with helper")


class Slot(QWidget):
    def __init__(self):
        super().__init__()
        self.pool = QThreadPool.globalInstance()
        print(self.pool.maxThreadCount())
        self.img_url = "source/images/"
        self.classifier = "KNeighborsClassifier"
        # self.worker = DeepFace()
        self.img = cv2.imread("Rowing.jpg")
        self.init_ui()
        # qss 而非 css  ...
        self.qss = self.read_qss("source/stylesheet/main.qss")
        self.setStyleSheet(self.qss)
        self.show()

    @staticmethod
    def read_qss(style):
        with open(style, 'r') as f:
            return f.read()

    def btn_event(self):
        print("into btn event")
        work = Work()
        work.helper.signal.connect(self.set_text)
        self.pool.start(work)
        print("start work")

    def set_text(self, text="123"):
        self.label.setText(text)
        print("Set Text: " + text)

    def set_label(self):
        img = None
        path = self.img_url + "Colin_Powell"
        for file in os.listdir(path):
            file = path + "/" + file
            img = QPixmap(file)
            break
        print(img.height())
        self.label_img.setPixmap(img)
        self.label_img.setScaledContents(True)
        print("Set Img success")

    def init_ui(self):
        grid = QGridLayout()
        self.label = QLabel("001", self)
        self.label.setObjectName("label1")
        self.label_img = QLabel("Img", self)
        self.label_img.setObjectName("label2")
        self.btn = QPushButton("Scar")
        self.btn.setObjectName("btn")
        # self.btn.setStyleSheet("color: #e74c3c")
        self.btn_img = QPushButton("Img")
        self.btn_img.setObjectName("btn")
        # self.btn_img.setStyleSheet("color: #2ecc71")
        self.btn.clicked.connect(self.btn_event)
        self.btn_img.clicked.connect(self.set_label)
        grid.addWidget(self.label)
        grid.addWidget(self.label_img)
        grid.addWidget(self.btn)
        grid.addWidget(self.btn_img)
        self.setLayout(grid)
        self.setGeometry(300, 300, 1024, 768)
        self.setWindowTitle("Event Detection")

    def start_work(self):
        # 此处图片无法立刻显示的原因在于， 进行识别的耗时操作在主线程中处理，阻塞了界面刷新，等待计算结束，自然展示出来
        # 此外，调试有些时候本事会导致 程序失去响应。。
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
        img = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        print(img)
        print(img.height())
        self.label_img.setPixmap(QPixmap.fromImage(img))
        self.worker.pre_train(self.img_url)
        self.worker.train(self.classifier)
        res = self.worker.predict(self.img)
        print(res)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    slot = Slot()

    sys.exit(app.exec_())

