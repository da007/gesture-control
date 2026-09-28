from pathlib import Path

from src.core import GestureRecognition


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "gesture_recognizer.task"
)


app = GestureRecognition(MODEL_PATH)

app.run()