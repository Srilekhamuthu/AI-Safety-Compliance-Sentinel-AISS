
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

# Model is loaded only when the first image is analyzed.
model = None
model_lock = threading.Lock()

# Use MODEL_PATH from Render if configured.
# Otherwise, load aiss_ppe.pt from the app's working directory.
MODEL_PATH = os.getenv("MODEL_PATH", "aiss_ppe.pt")

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
        button {
            padding: 10px 18px;
            cursor: pointer;
            background: #1d4ed8;
            color: white;
            border: none;
            border-radius: 6px;
        }
        input {
            margin: 12px 0;
            max-width: 100%;
        }
        img {
            max-width: 100%;
            margin-top: 20px;
            border-radius: 8px;
        }
        .error {
            color: #b91c1c;
        }
        .result {
            overflow-wrap: anywhere;
        }
    </style>
</head>
<body>
<main>
    <h1>AISS Safety Compliance Sentinel</h1>
    <p>Upload a construction-site image to detect PPE.</p>

    <form method="post" enctype="multipart/form-data">
        <input
            type="file"
            name="image"
            accept="image/*"
            required
        >
        <br>
        <button type="submit">Analyze Image</button>
    </form>

    {% if error %}
        <h3 class="error">Analysis failed</h3>
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if result %}
        <h2>Detection Results</h2>
        <p class="result">{{ result }}</p>
        <img
            src="data:image/jpeg;base64,{{ image_data }}"
            alt="Image with detection boxes"
        >
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
                logger.info("Loading PPE model from: %s", MODEL_PATH)

                if not os.path.isfile(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Model file not found: {MODEL_PATH}. "
                        "Check that aiss_ppe.pt is deployed and "
                        "MODEL_PATH points to the correct file."
                    )

                model = YOLO(MODEL_PATH)
                logger.info("PPE model loaded successfully.")

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
        HTML,
        result=None,
        image_data=None,
        error=None
    )


@app.post("/")
def analyze():
    try:
        uploaded = request.files.get("image")

        if not uploaded or not uploaded.filename:
            return render_template_string(
                HTML,
                result=None,
                image_data=None,
                error="Please choose an image first."
            ), 400

        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if image is None:
            return render_template_string(
                HTML,
                result=None,
                image_data=None,
                error="The file is not a valid image."
            ), 400

        # Resize large images to reduce CPU and memory usage.
        height, width = image.shape[:2]
        max_dimension = 640

        scale = min(
            max_dimension / max(height, width),
            1.0
        )

        if scale < 1.0:
            image = cv2.resize(
                image,
                (
                    max(1, int(width * scale)),
                    max(1, int(height * scale))
                ),
                interpolation=cv2.INTER_AREA
            )

        logger.info("Starting PPE detection.")

        detector = get_model()

        predictions = detector.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        prediction = predictions[0]
        output = prediction.plot()

        success, encoded = cv2.imencode(
            ".jpg",
            output,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        if not success:
            raise RuntimeError("Could not encode the result image.")

        image_data = base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")

        names = prediction.names
        boxes = prediction.boxes

        if boxes is not None and len(boxes) > 0:
            detected = []

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = names[class_id]

                detected.append(
                    f"{class_name} ({confidence:.0%})"
                )

            result = ", ".join(detected)
        else:
            result = "No objects detected in this image."

        logger.info("PPE detection completed.")

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
