from PyQt5.QtGui import QKeyEvent, QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
import threading
from .plotlib import process_display_image

class QMainWindow(QWidget):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setWindowTitle("Preview")
        self.label = QLabel(self)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.label)
        self.fakes_image = None
        self.reals_image = None
        self.static_fakes_image = None
        self.fakes_list = None
        self.reals_list = None
        self.static_fakes_list = None
        self.exit_flag = False
        self.save_flag = False
        self.image_lock = threading.Lock()
        self.show()

        self.display_mode = 0 # 0 is static fake sample mode, 1 is random fake sample mode, 2 is real sample mode

    def updatePreviewImage(self):
        self.image_lock.acquire()
        if (self.fakes_list is None or self.reals_list is None or self.static_fakes_list is None):
            self.image_lock.release()
            return
        self.fakes_image = process_display_image(self.fakes_list)
        self.reals_image = process_display_image(self.reals_list)
        self.static_fakes_image = process_display_image(self.static_fakes_list)
        self.fakes_list = None
        self.reals_list = None
        self.static_fakes_list = None
        self.image_lock.release()

    def updateDisplay(self):
        if (self.fakes_image is None or self.reals_image is None or self.static_fakes_image is None):
            return 

        image = None
        if (self.display_mode == 0):
            image = self.fakes_image
        elif (self.display_mode == 1):
            image = self.reals_image
        else:
            image = self.static_fakes_image

        H, W, C = image.shape
        self.image = QPixmap.fromImage(QImage(image.data, W, H, 3 * W, QImage.Format.Format_BGR888))
        self.label.setPixmap(self.image)
        self.setFixedSize(W, H)

    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() == Qt.Key.Key_Q):
            self.exit_flag = True
            QApplication.quit()
        elif (event.key() == Qt.Key.Key_U):
            self.updatePreviewImage()
            self.updateDisplay()
        elif (event.key() == Qt.Key.Key_P):
            self.display_mode = (self.display_mode + 1) % 3
            self.updateDisplay()
        elif (event.key() == Qt.Key.Key_S):
            self.save_flag = True
        else:
            super().keyPressEvent(event)