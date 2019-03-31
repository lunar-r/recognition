import sys,time
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class BackQthread(QThread):
    #自定义信号为str参数类型
    update_date=pyqtSignal(str)

    def run( self ):
        while True:
            #获得当前系统时间
            data=QDateTime.currentDateTime()
            #设置时间显示格式
            curTime=data.toString('yyyy-MM-dd hh:mm:ss dddd')
            #发射信号
            self.update_date.emit(curTime)
            #睡眠一秒
            time.sleep(1)
class window(QDialog):
    def __init__(self):
        super(window, self).__init__()
        #设置标题与初始大小
        self.setWindowTitle('PyQt5界面实时更新的例子')
        self.resize(400,100)
        #实例化文本输入框及其初始大小
        self.input=QLabel(self)
        self.input.resize(400,100)
        self.initUI()
    def initUI( self ):
        #实例化对象
        self.backend=BackQthread()
        #信号连接到界面显示槽函数
        self.backend.update_date.connect(self.handleDisplay)
        #多线程开始
        self.backend.start()

    def handleDisplay( self,data ):
        #设置单行文本框的文本
        self.input.setText(data)

if __name__ == '__main__':
    app=QApplication(sys.argv)
    win=window()
    win.show()
    sys.exit(app.exec_())
