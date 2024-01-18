import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("PySide 6 Template")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        label = QLabel("Hello, PySide 6!", self)
        layout.addWidget(label)

        button = QPushButton("Click Me", self)
        button.clicked.connect(self.on_button_click)
        layout.addWidget(button)

    def on_button_click(self):
        print("Button Clicked!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())