
import base64
import os

import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")
model = YOLO(MODEL_PATH)

HTML = """
<!doctype html>
<html>
<head>
    <title>AISS - Safety Compliance Sentinel</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; max-width: 800px; margin: 40px auto;
               padding: 20px; background: #f1f5f9; color: #172554; }
        main { background: white; padding: 24px; border-radius: 12px; }
        button { padding: 10px 18px; cursor: pointer; }
        img { max-width: 100%; margin-top: 20px; }
    </style>
</head>
<body>
<main>
    <h1>AISS Safety Compliance Sentinel</h1>
    <p>Upload an image to run AI object detection.</p>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Analyze Image</button>
    </form>
    {% if error %}<p>{{ error }}</p>{% endif %}
    {% if result %}
        <h2>Detection Results</h2>
        <p>{{ result }}</p>
        <img src="data:image/jpeg;base64,{{ image_data }}">
    {% endif %}
</main>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML, result=None, error=None)


@app.post("/")
def analyze():
    uploaded = request.files.get("image")

    if not uploaded or not uploaded.filename:
        return render_template_string(
            HTML, result=None, error="Please select an image."
        )

    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if image is None:
        return render_template_string(
            HTML, result=None, error="Invalid image file."
        )

    predictions = model.predict(image, conf=0.5, verbose=False)
    annotated = predictions[0].plot()

    success, encoded = cv2.imencode(".jpg", annotated)
    if not success:
        return render_template_string(
            HTML, result=None, error="Could not process the image."
        )

    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")
    names = predictions[0].names
    classes = predictions[0].boxes.cls.tolist()
    detected = sorted({names[int(c)] for c in classes})
    result = ", ".join(detected) if detected else "No objects detected."

    return render_template_string(
        HTML, result=result, image_data=image_data, error=None
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000"))
    )
