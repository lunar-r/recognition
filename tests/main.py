from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class TestThread(QThread):
    def __init__(self):
        super(TestThread, self).__init__()
        self.num = 1

    def run(self):
        self.num = self.num + 1
        print(self.num)
        print(self.currentThreadId())


for i in range(10):
    thread = QThread()
    thread.start()
