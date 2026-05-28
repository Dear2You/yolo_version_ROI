import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap


def cv_to_pixmap(image, target_size):
    if image is None:
        return QPixmap()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(target_size, Qt.KeepAspectRatio, Qt.FastTransformation)
