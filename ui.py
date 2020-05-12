import sys

from PySide2 import QtCore, QtWidgets, QtGui

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk as gtk

import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

import imageio

import FILTROS as f

class MyWidget():
    def __init__(self):
        super().__init__()

        button = QtWidgets.QPushButton("CLick me!")
        self.text = QtWidgets.QLabel("Hello World")
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(button)
        self.setLayout(self.layout)

        self.progress_dialog(self)

        button.clicked.connect(self.magic)

    def magic(self):
        file_path = filedialog.askopenfilename()
        self.text.setText(file_path)

        lenna = imageio.imread(file_path)
        f.rodar_em_cor(lenna)
        f.rodar_em_cinza(lenna)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
