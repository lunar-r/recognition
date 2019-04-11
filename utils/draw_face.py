from mtcnn.mtcnn import MTCNN
# self.detector = MTCNN()
# res = self.detector.detect_faces(show_1)
# for face in res:
#     box = face['box']
#     cv2.rectangle(show_1, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 155, 255), 2)
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.timer_camera = QTimer()  # 需要定时器刷新摄像头界面
        self.video = cv2.VideoCapture()
        harr_filepath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # 系统安装的是opencv-contrib-python
        self.classifier = cv2.CascadeClassifier(harr_filepath)  # 加载人脸特征分类器
        self.set_ui()  # 初始化UI界面
        self.slot_init()  # 初始化信号槽

    def set_ui(self):
        # 布局设置
        self.layout_main = QHBoxLayout()  # 整体框架是水平布局
        self.layout_button = QVBoxLayout()  # 按键布局是垂直布局

        # 按钮设置
        self.btn_video = QPushButton('打开视频')
        self.btn_quit = QPushButton('退出')

        # 显示视频
        self.label_video = QLabel()
        self.label_video.setFixedSize(800, 600)
        self.label_video.setAutoFillBackground(False)

        # 布局
        self.layout_button.addWidget(self.btn_video)
        self.layout_button.addWidget(self.btn_quit)

        self.layout_main.addLayout(self.layout_button)
        self.layout_main.addWidget(self.label_video)

        self.setLayout(self.layout_main)
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle("人脸识别软件")

    def slot_init(self):
        self.btn_video.clicked.connect(self.btn_open_video)
        self.timer_camera.timeout.connect(self.show_video)
        self.btn_quit.clicked.connect(self.close)

    def btn_open_video(self):
        if self.timer_camera.isActive() is False:
            flag = self.video.open("./source/video/speaks.mp4")
            if flag is False:
                print("Can't open file")
            else:
                self.timer_camera.start(30)
                self.btn_video.setText("关闭相机")
        else:
            self.timer_camera.stop()
            self.video.release()
            self.label_video.clear()
            self.btn_video.setText(u'打开相机')

    def show_video(self):
        ret, img = self.video.read()
        show_0 = cv2.resize(img, (800, 600))
        show_1 = cv2.cvtColor(show_0, cv2.COLOR_BGR2RGB)

        gray_image = cv2.cvtColor(show_0, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(gray_image, 1.3, 5)  # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
        for (x, y, w, h) in faces:
            cv2.rectangle(show_1, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 画出人脸
        detect_image = QImage(show_1.data, show_1.shape[1], show_1.shape[0],
                              QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(detect_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())