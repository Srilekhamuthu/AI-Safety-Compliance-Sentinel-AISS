import os
import base64
import logging
import threading

import cv2
import numpy as np
from flask import Flask, request, render_template_string
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

model = None
model_lock = threading.Lock()

HTML = """
<!doctype html>
<html>
<head>
    <title>AISS Safety Compliance Sentinel</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            background: #f1f5f9;
            color: #172554;
        }
        main {
            background: white;
            padding: 24px;
            border-radius: 12px;
        }
        button { padding: 10px 18px; cursor: pointer; }
        img { max-width: 100%; margin-top: 20px; }
        .error { color: #b91c1c; }
        .result { overflow-wrap: anywhere; }
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

    {% if error %}
        <h3 class="error">Analysis failed</h3>
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if result %}
        <h2>Detection Results</h2>
        <p class="result">{{ result }}</p>
        <img src="data:image/jpeg;base64,{{ image_data }}"
             alt="Image with detection boxes">
    {% endif %}
</main>
</body>
</html>
"""


def get_model():
    global model

    if model is None:
        with model_lock:
            if model is None:
                logger.info("Loading YOLO model...")
                model = YOLO(os.getenv("MODEL_PATH", "yolov8n.pt"))
                logger.info("YOLO model loaded successfully.")

    return model


@app.errorhandler(413)
def file_too_large(error):
    return render_template_string(
        HTML,
        result=None,
        image_data=None,
        error="Image is too large. Please upload an image smaller than 8 MB."
    ), 413


@app.get("/")
def home():
    return render_template_string(
        HTML, result=None, image_data=None, error=None
    )


@app.post("/")
def analyze():
    try:
        uploaded = request.files.get("image")

        if not uploaded or not uploaded.filename:
            return render_template_string(
                HTML, result=None, image_data=None,
                error="Please choose an image first."
            ), 400

        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if image is None:
            return render_template_string(
                HTML, result=None, image_data=None,
                error="The file is not a valid image."
            ), 400

        # Reduce memory use for large images.
        height, width = image.shape[:2]
        max_dimension = 640

        scale = min(
            max_dimension / max(height, width),
            1.0
        )

        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA
            )

        logger.info("Starting image detection.")

        detector = get_model()
        predictions = detector.predict(
            source=image,
            imgsz=640,
            conf=0.5,
            device="cpu",
            verbose=False
        )

        output = predictions[0].plot()

        success, encoded = cv2.imencode(
            ".jpg", output,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if not success:
            raise RuntimeError("Could not encode the result image.")

        image_data = base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")

        names = predictions[0].names
        boxes = predictions[0].boxes

        if boxes is not None and len(boxes) > 0:
            detected = sorted({
                names[int(class_id)]
                for class_id in boxes.cls.tolist()
            })
            result = ", ".join(detected)
        else:
            result = "No objects detected in this image."

        logger.info("Image detection completed.")

        return render_template_string(
            HTML,
            result=result,
            image_data=image_data,
            error=None
        )

    except Exception:
        logger.exception("Image analysis failed.")

        return render_template_string(
            HTML,
            result=None,
            image_data=None,
            error=(
                "The server could not process this image. "
                "Please check the Render runtime logs."
            )
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000"))
    )
