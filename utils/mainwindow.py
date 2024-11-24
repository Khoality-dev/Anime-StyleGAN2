
from .plotlib import process_display_image, show_image

class MainWindow():
    def __init__(self):
        self.fakes_image = None
        self.reals_image = None
        self.static_fakes_image = None
        self.exit_flag = False
        self.save_flag = False
        self.update_flag = True
        
    def updatePreviewImage(self, fakes_list, reals_list, static_fakes_list):
        if (fakes_list is not None):
            self.fakes_image = process_display_image(fakes_list)
        
        if reals_list is not None:
            self.reals_image = process_display_image(reals_list)
            
        if static_fakes_list is not None:
            self.static_fakes_image = process_display_image(static_fakes_list)

    def updateDisplay(self):
        if (self.fakes_image is None or self.reals_image is None or self.static_fakes_image is None):
            return 

        show_image(self.fakes_image, "Preview_Fake_Images.jpg")
        show_image(self.static_fakes_image, "Preview_Static_Fake_Images.jpg")
        show_image(self.reals_image, "Preview_Real_Images.jpg")


try:
    from PyQt5.QtGui import QKeyEvent, QImage, QPixmap
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
    class QMainWindow(QWidget, MainWindow):
        def __init__(self):
            super(QMainWindow, self).__init__()
            self.setWindowTitle("Preview")
            self.label = QLabel(self)
            self.label.setContentsMargins(0,0,0,0)
            self.setLayout(QVBoxLayout())
            self.layout().setContentsMargins(0,0,0,0)
            self.layout().addWidget(self.label)
            self.show()
            self.display_mode = 0 # 0 is static fake sample mode, 1 is random fake sample mode, 2 is real sample mode
        
        def updateDisplay(self):
            if (self.fakes_image is None or self.reals_image is None or self.static_fakes_image is None):
                return 

            image = None
            if (self.display_mode == 0):
                image = self.fakes_image
            elif (self.display_mode == 1):
                image = self.static_fakes_image
            else:
                image = self.reals_image

            H, W, C = image.shape
            self.image = QPixmap.fromImage(QImage(image.data, W, H, 3 * W, QImage.Format.Format_BGR888))
            self.label.setPixmap(self.image)
            self.setFixedSize(W, H)

        def keyPressEvent(self, event):
            if (event.key() == Qt.Key.Key_Q):
                self.exit_flag = True
                QApplication.quit()
            elif (event.key() == Qt.Key.Key_U):
                self.update_flag = True
            elif (event.key() == Qt.Key.Key_BracketLeft):
                self.display_mode = max(self.display_mode - 1, 0)
                self.updateDisplay()
            elif (event.key() == Qt.Key.Key_BracketRight):
                self.display_mode = min(self.display_mode + 1, 2)
                self.updateDisplay()
            elif (event.key() == Qt.Key.Key_S):
                self.save_flag = True
            else:
                super().keyPressEvent(event)
except ImportError:
    ...
