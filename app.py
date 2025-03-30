import sys
import subprocess
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
import os

class ScriptLauncher(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Script Launcher")
        self.showFullScreen()  # Make the app full-screen
        self.setStyleSheet("background-color: #1E1E2E; color: white;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Select a script to run:")
        label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        self.eye_tracker_btn = self.create_button("🎯 Run Eye Tracker")
        self.face_recog_btn = self.create_button("🔍 Run Face Recognition")
        self.speech_recog_btn = self.create_button("🎙️ Run Speech Recognition")

        self.eye_tracker_btn.clicked.connect(lambda: self.run_script("eye_tracker_app.py"))
        self.face_recog_btn.clicked.connect(lambda: self.run_script("face_recognition_app.py"))
        self.speech_recog_btn.clicked.connect(lambda: self.run_script("speech_recog_app.py"))

        layout.addWidget(self.eye_tracker_btn)
        layout.addWidget(self.face_recog_btn)
        layout.addWidget(self.speech_recog_btn)

    def create_button(self, text):
        button = QPushButton(text)
        button.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        button.setFixedHeight(60)
        button.setStyleSheet(
            "QPushButton {"
            "background-color: #5E81AC; color: white; border-radius: 15px; padding: 15px;"
            "transition: 0.3s; font-size: 18px;"
            "}"
            "QPushButton:hover { background-color: #81A1C1; transform: scale(1.05); }"
            "QPushButton:pressed { background-color: #4C566A; transform: scale(0.95); }"
        )
        return button

    def run_script(self, script_name):
        try:
            subprocess.Popen([sys.executable, script_name])
        except Exception as e:
            print(f"Error launching {script_name}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScriptLauncher()
    window.show()
    sys.exit(app.exec())
