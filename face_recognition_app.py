
import sys
import cv2
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QMessageBox, QFrame
)
from PyQt6.QtGui import QFont, QPixmap, QImage
from PyQt6.QtCore import Qt
from deepface import DeepFace
from collections import Counter

class QuizApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.detected_emotion = "neutral"
        self.quiz_questions = []
        self.current_question_index = 0
        self.score = 0

    def initUI(self):
        self.setWindowTitle("AI Quiz App - Dark Mode")
        self.setGeometry(100, 100, 700, 500)
        self.setStyleSheet("background-color: #121212; color: #ffffff;")

        # Main Layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Title Label
        self.title_label = QLabel("AI-Driven Emotion-Based Quiz")
        self.title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Instruction Label
        self.label = QLabel("Press 'Start Quiz' to begin face & emotion detection")
        self.label.setFont(QFont("Arial", 14))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        # Start Button
        self.start_button = QPushButton("Start Quiz")
        self.start_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.start_button.setStyleSheet(self.get_button_styles())
        self.start_button.clicked.connect(self.detect_emotion)
        self.layout.addWidget(self.start_button)

        # Horizontal Line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #333;")
        self.layout.addWidget(line)

        # Question Label
        self.question_label = QLabel("")
        self.question_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.question_label)

        # Answer Input
        self.answer_input = QLineEdit()
        self.answer_input.setFont(QFont("Arial", 14))
        self.answer_input.setPlaceholderText("Enter your answer here...")
        self.answer_input.setStyleSheet(self.get_input_styles())
        self.layout.addWidget(self.answer_input)

        # Submit Button
        self.submit_button = QPushButton("Submit Answer")
        self.submit_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.submit_button.setStyleSheet(self.get_button_styles())
        self.submit_button.clicked.connect(self.check_answer)
        self.layout.addWidget(self.submit_button)
        self.submit_button.setEnabled(False)

        self.setLayout(self.layout)

    def detect_emotion(self):
        self.label.setText("Detecting face and emotion... Please wait.")
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        emotion_records = []

        for _ in range(10):  # Capture frames for emotion detection
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                try:
                    emotion_analysis = DeepFace.analyze(frame[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)
                    detected_emotion = emotion_analysis[0]['dominant_emotion']
                    emotion_records.append(detected_emotion)
                except:
                    pass

        cap.release()

        if emotion_records:
            self.detected_emotion = Counter(emotion_records).most_common(1)[0][0]
        else:
            self.detected_emotion = "neutral"

        self.label.setText(f"Detected Emotion: {self.detected_emotion}. Loading quiz...")
        self.load_quiz()

    def load_quiz(self):
        df = pd.read_csv("./math_questions.csv")
        difficulty_map = {
            "happy": "Hard", "sad": "Medium", "angry": "Easy", "fear": "Easy",
            "surprise": "Hard", "neutral": "Medium", "disgust": "Medium"
        }
        difficulty = difficulty_map.get(self.detected_emotion, "Medium")

        self.quiz_questions = df[df["Difficulty"] == difficulty].sample(
            n=min(10, len(df[df["Difficulty"] == difficulty])), 
            random_state=None
        ).reset_index(drop=True)

        self.current_question_index = 0
        self.score = 0
        self.show_question()

    def show_question(self):
        if self.current_question_index < len(self.quiz_questions):
            question_text = self.quiz_questions.iloc[self.current_question_index]["Question"]
            self.question_label.setText(f"Q{self.current_question_index+1}: {question_text}")
            self.answer_input.clear()
            self.submit_button.setEnabled(True)
        else:
            QMessageBox.information(
                self, 
                "Quiz Completed", 
                f"Your final score: {self.score}/{len(self.quiz_questions)}"
            )
            self.submit_button.setEnabled(False)

    def check_answer(self):
        user_answer = self.answer_input.text().strip()
        correct_answer = str(self.quiz_questions.iloc[self.current_question_index]["Answers"]).strip()
        
        if user_answer == correct_answer:
            self.score += 1

        self.current_question_index += 1
        self.show_question()

    def get_button_styles(self):
        return """
            QPushButton {
                background-color: #00FF00;
                color: black;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #00FF00;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00CC00;
            }
        """

    def get_input_styles(self):
        return """
            QLineEdit {
                background-color: #333;
                color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #555;
            }
            QLineEdit::placeholder {
                color: #888;
            }
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuizApp()
    window.show()
    sys.exit(app.exec())

