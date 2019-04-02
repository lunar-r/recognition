from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys


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

        self.init_ui()

    def btn_event(self):
        print("into btn event")
        work = Work()
        work.helper.signal.connect(self.set_text)
        self.pool.start(work)
        print("start work")

    def set_text(self, text="123"):
        self.label.setText(text)
        print("Set Text: " + text)

    def init_ui(self):
        grid = QGridLayout()
        self.label = QLabel("001", self)
        self.btn = QPushButton("Scar")
        self.btn.clicked.connect(self.btn_event)
        grid.addWidget(self.label)
        grid.addWidget(self.btn)
        self.setLayout(grid)
        self.setGeometry(300, 300, 300, 300)
        self.setWindowTitle("Event Detection")

        self.show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    slot = Slot()
    sys.exit(app.exec_())

