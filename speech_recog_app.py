import sys
import random
import pandas as pd
import speech_recognition as sr
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QHBoxLayout
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from transformers import pipeline

# Load the emotion classification model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

df = pd.read_csv("./math_questions.csv")
df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion-Based Quiz Recommender 🎭")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("background-color: #2E3440; color: white; font-size: 16px;")

        self.title_label = QLabel("Emotion-Based Quiz Recommender 🎭", self)
        self.title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label = QLabel("Enter text or use voice input:", self)
        self.label.setFont(QFont("Arial", 12))
        self.textbox = QLineEdit(self)
        self.textbox.setStyleSheet("background-color: #3B4252; color: white; border-radius: 5px; padding: 5px;")

        
        self.voice_button = QPushButton("Detect Emotion (Voice) 🎙️", self)

        self.voice_button.setStyleSheet(self.button_style())


        self.voice_button.clicked.connect(self.detect_emotion_voice)

        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("color: #88C0D0;")

        self.quiz_label = QLabel("", self)
        self.quiz_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.quiz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.quiz_label.setStyleSheet("color: #A3BE8C;")

        self.question_label = QLabel("", self)
        self.question_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.question_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.question_label.setStyleSheet("color: #EBCB8B;")

        self.answer_input = QLineEdit(self)
        self.answer_input.setPlaceholderText("Enter your answer here...")
        self.answer_input.setStyleSheet("background-color: #3B4252; color: white; border-radius: 5px; padding: 5px;")

        self.submit_button = QPushButton("Submit Answer", self)
        self.submit_button.setStyleSheet(self.button_style())
        self.submit_button.clicked.connect(self.check_answer)

        self.answer_result_label = QLabel("", self)
        self.answer_result_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.answer_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.answer_result_label.setStyleSheet("color: #BF616A;")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.label)
        layout.addWidget(self.textbox)

        button_layout = QHBoxLayout()

        button_layout.addWidget(self.voice_button)
        layout.addLayout(button_layout)

        layout.addWidget(self.result_label)
        layout.addWidget(self.quiz_label)
        layout.addWidget(self.question_label)
        layout.addWidget(self.answer_input)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.answer_result_label)

        self.setLayout(layout)

        self.questions = []  
        self.current_index = 0  
        self.score = 0  

    def button_style(self):
        return """
            QPushButton {
                background-color: #5E81AC;
                color: white;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
        """

    def detect_emotion(self):
        text = self.textbox.text().strip()
        if text:
            result = emotion_classifier(text)
            emotion = result[0]['label']
            self.result_label.setText(f"Detected Emotion: {emotion}")
            self.suggest_quiz(emotion)
        else:
            self.result_label.setText("⚠️ Please enter some text.")

    def detect_emotion_voice(self):
        recognizer = sr.Recognizer()
        self.result_label.setText("🎙️ Listening... Please speak.")

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("speak now..")
                
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)

                self.textbox.setText(text)
                self.detect_emotion()
        except sr.UnknownValueError:
            self.result_label.setText("⚠️ Could not understand audio.")
        except sr.RequestError:
            self.result_label.setText("⚠️ Internet connection required.")
        except Exception as e:
            self.result_label.setText(f"⚠️ Error: {str(e)}")

    def suggest_quiz(self, emotion):
        quiz_difficulty = {
            "joy": "Hard 💪",  
            "surprise": "Hard 💪",  
            "neutral": "Medium ⚖️",
            "sadness": "Easy 🧘",
            "anger": "Easy 🧘",
            "fear": "Medium ⚖️",
            "disgust": "Medium ⚖️"
        }

        difficulty = quiz_difficulty.get(emotion.lower(), "Medium ⚖️")
        self.quiz_label.setText(f"🎯 Recommended Quiz Difficulty: {difficulty}")

        self.questions = self.get_random_questions(difficulty.split()[0])
        self.current_index = 0  
        self.score = 0  

        if self.questions:
            self.display_question()
        else:
            self.question_label.setText("⚠️ No questions available for this difficulty.")

    def get_random_questions(self, difficulty):
        filtered_df = df[df['Difficulty'].str.lower() == difficulty.lower()]
        if not filtered_df.empty:
            return filtered_df.sample(min(10, len(filtered_df))).to_dict(orient="records")
        return []

    def display_question(self):
        if self.current_index < len(self.questions):
            question_data = self.questions[self.current_index]
            self.question_label.setText(f"❓ Question {self.current_index + 1}: {question_data['Question']}")
        else:
            self.show_final_score()

    def check_answer(self):
        if self.current_index < len(self.questions):
            user_answer = self.answer_input.text().strip()
            correct_answer = str(self.questions[self.current_index]['Answers']).strip()

            if user_answer.lower() == correct_answer.lower():
                self.answer_result_label.setText("✅ Correct!")
                self.score += 1
            else:
                self.answer_result_label.setText(f"❌ Incorrect! The correct answer is: {correct_answer}")

            self.current_index += 1
            self.answer_input.clear()

            if self.current_index < len(self.questions):
                self.display_question()
            else:
                self.show_final_score()

    def show_final_score(self):
        self.question_label.setText(f"🏆 Quiz Complete! Your Score: {self.score} / {len(self.questions)}")
        self.answer_result_label.setText("🎉 Great Job!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionApp()
    window.show()
    sys.exit(app.exec())
