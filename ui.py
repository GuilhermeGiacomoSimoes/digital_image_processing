import sys

from PySide2 import QtCore, QtWidgets, QtGui

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk as gtk

import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        button = QtWidgets.QPushButton("CLick me!")
        self.text = QtWidgets.QLabel("Hello World")
        self.text.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(button)
        self.setLayout(self.layout)

        button.clicked.connect(self.magic)

    def magic(self):
        file_path = filedialog.askopenfilename()
        self.text.setText(file_path)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
