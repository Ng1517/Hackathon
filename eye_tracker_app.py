
import sys
import cv2
import dlib
import time
import numpy as np
import pandas as pd
from collections import Counter
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))

def get_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C)

def analyze_eye_tracking(frame, landmarks):
    left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES])
    right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES])
    left_ear = get_eye_aspect_ratio(left_eye)
    right_ear = get_eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    if avg_ear < 0.25:
        return "Drowsy or Low Engagement"
    elif avg_ear < 0.3:
        return "Neutral"
    else:
        return "Highly Engaged"

def get_most_frequent_engagement(engagement_list):
    if not engagement_list:
        return "Neutral"
    return Counter(engagement_list).most_common(1)[0][0]

def get_quiz(engagement, num_questions=10):
    df = pd.read_csv("./math_questions.csv")
    mapping = {"Drowsy or Low Engagement": "Easy", "Neutral": "Medium", "Highly Engaged": "Hard"}
    difficulty = mapping.get(engagement, "Medium")
    filt_df = df[df["Difficulty"] == difficulty]
    selected_questions = filt_df.sample(n=min(num_questions, len(filt_df)), random_state=None).reset_index(drop=True)
    return selected_questions, difficulty

class QuizApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 400)
        self.setWindowTitle("AI Adaptive Quiz")
        self.initUI()
    
    def initUI(self):
        self.setStyleSheet("background-color: #121212; color: #ffffff;")
        layout = QVBoxLayout()
        
        title = QLabel("Welcome to the AI Adaptive Quiz!")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        description = QLabel("This quiz will adapt to your engagement level, ensuring an optimized experience.")
        description.setFont(QFont("Arial", 12))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        
        font = QFont("Arial", 14)
        
        self.start_btn = QPushButton("Start Quiz")
        self.start_btn.setFont(font)
        self.start_btn.setStyleSheet("background-color: #1E88E5; color: white; padding: 10px; border-radius: 5px;")
        self.start_btn.clicked.connect(self.start_eye_tracking)
        layout.addWidget(self.start_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.engagement_label = QLabel("Detecting engagement level...")
        self.engagement_label.setFont(font)
        layout.addWidget(self.engagement_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.difficulty_label = QLabel("Difficulty Level: Not determined yet")
        self.difficulty_label.setFont(font)
        layout.addWidget(self.difficulty_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.question_label = QLabel("")
        self.question_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        layout.addWidget(self.question_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.answer_input = QLineEdit()
        self.answer_input.setFont(font)
        self.answer_input.setStyleSheet("background-color: #333333; color: white; padding: 5px; border: 2px solid #555; border-radius: 5px;")
        layout.addWidget(self.answer_input, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.next_btn = QPushButton("Next")
        self.next_btn.setFont(font)
        self.next_btn.setStyleSheet("background-color: #43A047; color: white; padding: 10px; border-radius: 5px;")
        self.next_btn.clicked.connect(self.next_question)
        layout.addWidget(self.next_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self.next_btn.hide()
        
        self.score_label = QLabel("")
        self.score_label.setFont(font)
        layout.addWidget(self.score_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.setLayout(layout)
    
    def start_eye_tracking(self):
        cap = cv2.VideoCapture(0)
        engagement_records = []
        start_time = time.time()
        duration = 15
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                landmarks = predictor(gray, face)
                engagement = analyze_eye_tracking(frame, landmarks)
                engagement_records.append(engagement)
            time.sleep(2)
            if time.time() - start_time >= duration:
                break
        cap.release()
        cv2.destroyAllWindows()
        self.engagement_level = get_most_frequent_engagement(engagement_records)
        self.engagement_label.setText(f"Your engagement level: {self.engagement_level}")
        
        self.quiz_ques, difficulty = get_quiz(self.engagement_level)
        self.difficulty_label.setText(f"Difficulty Level: {difficulty}")
        self.current_question_index = 0
        self.correct_answers = 0
        self.show_question()
    
    def show_question(self):
        if self.current_question_index < len(self.quiz_ques):
            question_text = self.quiz_ques.iloc[self.current_question_index]['Question']
            self.question_label.setText(f"Q{self.current_question_index + 1}: {question_text}")
            self.answer_input.show()
            self.answer_input.clear()
            self.next_btn.show()
        else:
            self.answer_input.hide()
            self.show_score()
    
    def next_question(self):
        correct_answer = str(self.quiz_ques.iloc[self.current_question_index]['Answers']).strip()
        user_answer = self.answer_input.text().strip()
        if user_answer == correct_answer:
            self.correct_answers += 1
        self.current_question_index += 1
        self.show_question()
    
    def show_score(self):
        self.question_label.setText("Quiz Completed!")
        self.score_label.setText(f"Your Score: {self.correct_answers} / {len(self.quiz_ques)}")
        self.next_btn.hide()

def main():
    app = QApplication(sys.argv)
    ex = QuizApp()
    ex.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

