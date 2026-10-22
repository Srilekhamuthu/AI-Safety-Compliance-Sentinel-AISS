
import os
import io
import csv
import json
import base64
import logging
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, render_template_string, jsonify
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL_PATH = os.getenv("MODEL_PATH", "aiss_ppe.pt")
GATE_NAME = os.getenv("GATE_NAME", "Gate 1")
SUPERVISOR_EMAIL = os.getenv("SUPERVISOR_EMAIL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", SMTP_USERNAME)

# Configure the mandatory PPE list in Render using comma-separated names.
MANDATORY_PPE = [
    item.strip().lower()
    for item in os.getenv(
        "MANDATORY_PPE", "helmet,gloves,vest,boots,goggles"
    ).split(",")
    if item.strip()
]

# Persistent records require a persistent disk or external database.
# /tmp is temporary on Render; use a persistent disk path if available.
RECORDS_FILE = os.getenv("RECORDS_FILE", "/tmp/aiss_gate_records.csv")

model = None
model_lock = threading.Lock()
records_lock = threading.Lock()

PPE_CLASSES = {
    "helmet": ("helmet", "no_helmet"),
    "gloves": ("gloves", "no_gloves"),
    "vest": ("vest", "no_vest"),
    "boots": ("boots", "no_boots"),
    "goggles": ("goggles", "no_goggle"),
}


def get_model():
    global model

    if model is None:
        with model_lock:
            if model is None:
                if not os.path.isfile(MODEL_PATH):
                    raise FileNotFoundError(
                        f"Model file not found: {MODEL_PATH}"
                    )
                logger.info("Loading model: %s", MODEL_PATH)
                model = YOLO(MODEL_PATH)
                logger.info("PPE model loaded.")

    return model


def analyze_image(image):
    detector = get_model()
    prediction = detector.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    names = prediction.names
    detected = []
    confidences = {}

    if prediction.boxes is not None:
        for box in prediction.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = str(names[class_id]).strip().lower()

            detected.append(name)
            confidences[name] = max(
                confidences.get(name, 0.0), confidence
            )

    # Determine each mandatory item's status.
    # No negative class for vest exists in the current model.
    statuses = {}
    for item in MANDATORY_PPE:
        if item not in PPE_CLASSES:
            statuses[item] = "unverified"
            continue

        positive_class, negative_class = PPE_CLASSES[item]

        if negative_class in confidences:
            statuses[item] = "missing"
        elif positive_class in confidences:
            statuses[item] = "detected"
        else:
            statuses[item] = "unverified"

    missing = [
        item for item, status in statuses.items()
        if status == "missing"
    ]
    unverified = [
        item for item, status in statuses.items()
        if status == "unverified"
    ]

    if missing:
        overall = "NON-COMPLIANT - MISSING PPE"
    elif unverified:
        overall = "REVIEW REQUIRED - PPE UNVERIFIED"
    else:
        overall = "PPE DETECTED - VERIFY BEFORE ENTRY"

    annotated = prediction.plot()
    ok, encoded = cv2.imencode(
        ".jpg", annotated,
        [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    )
    if not ok:
        raise RuntimeError("Could not encode annotated image.")

    image_data = base64.b64encode(encoded.tobytes()).decode("ascii")

    return {
        "detected": detected,
        "confidences": confidences,
        "statuses": statuses,
        "missing": missing,
        "unverified": unverified,
        "overall": overall,
        "image_data": image_data
    }


def save_record(record):
    fields = [
        "timestamp", "employee_id", "gate", "status",
        "detected", "missing", "unverified", "email_status"
    ]

    path = Path(RECORDS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    with records_lock:
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                **record,
                "detected": ", ".join(record["detected"]),
                "missing": ", ".join(record["missing"]),
                "unverified": ", ".join(record["unverified"])
            })


def send_supervisor_email(record):
    if not SUPERVISOR_EMAIL:
        return "not configured"

    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.warning("SMTP credentials are not configured.")
        return "not configured"

    # Alert for missing PPE or any item that requires manual review.
    if not record["missing"] and not record["unverified"]:
        return "not required"

    message = EmailMessage()
    message["Subject"] = (
        f"AISS Safety Alert - {record['employee_id']} - {record['gate']}"
    )
    message["From"] = SENDER_EMAIL
    message["To"] = SUPERVISOR_EMAIL

    message.set_content(
        "AISS has flagged a gate-entry PPE check.\n\n"
        f"Employee ID: {record['employee_id']}\n"
        f"Gate: {record['gate']}\n"
        f"Time: {record['timestamp']}\n"
        f"Result: {record['status']}\n"
        f"Detected classes: {', '.join(record['detected']) or 'None'}\n"
        f"Missing PPE: {', '.join(record['missing']) or 'None detected'}\n"
        f"Unverified PPE: "
        f"{', '.join(record['unverified']) or 'None'}\n\n"
        "Please manually verify this result before making an entry decision."
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
        logger.info("Supervisor alert sent for %s", record["employee_id"])
        return "sent"
    except Exception:
        logger.exception("Could not send supervisor email.")
        return "failed"


HTML = """
<!doctype html>
<html>
<head>
    <title>AISS Gate Safety Check</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/html5-qrcode"></script>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px;
            margin: 24px auto; padding: 16px; background: #f1f5f9;
            color: #172554; }
        main { background: white; border-radius: 12px; padding: 22px; }
        input, button { padding: 10px; margin: 5px 0; max-width: 100%; }
        button { cursor: pointer; }
        video, img { width: 100%; max-width: 640px; margin-top: 12px; }
        .result { white-space: pre-wrap; overflow-wrap: anywhere; }
        .notice { background: #fef3c7; padding: 12px; border-radius: 8px; }
        .row { margin: 16px 0; }
    </style>
</head>
<body>
<main>
    <h1>AISS Safety Compliance Sentinel</h1>
    <p>Gate: <strong>{{ gate }}</strong></p>

    <div class="row">
        <h2>1. Scan employee ID QR code</h2>
        <div id="qr-reader" style="max-width:420px"></div>
        <button type="button" onclick="startScanner()">Start QR scanner</button>
        <button type="button" onclick="stopScanner()">Stop scanner</button>
        <p>Or enter the employee ID manually if scanning fails.</p>
        <label for="employee">Employee ID</label><br>
        <input id="employee" autocomplete="off" placeholder="e.g. EMP102" required>
    </div>

    <div class="row">
        <h2>2. Capture an image</h2>
        <video id="camera" autoplay playsinline></video><br>
        <button type="button" onclick="startCamera()">Start webcam</button>
        <button type="button" onclick="capturePhoto()">Capture photo</button>
        <canvas id="canvas" style="display:none"></canvas>
        <p>Or upload an image:</p>
        <input type="file" id="upload" accept="image/*">
    </div>

    <div class="row">
        <button type="button" onclick="submitCheck()">3. Check PPE and record gate entry</button>
    </div>

    <div class="notice">
        AI results can be incorrect. An unverified item needs manual review.
        Do not use this result alone to authorize or deny entry.
    </div>

    <div id="output" class="result"></div>
    <img id="annotated" alt="PPE detection result" style="display:none">
</main>

<script>
let scanner = null;
let cameraStream = null;
let capturedData = null;

async function startScanner() {
    try {
        if (!window.Html5Qrcode) {
            alert("QR scanner library did not load. Enter the ID manually.");
            return;
        }
        if (scanner) return;
        scanner = new Html5Qrcode("qr-reader");
        await scanner.start(
            { facingMode: "environment" },
            { fps: 10, qrbox: { width: 230, height: 230 } },
            (decodedText) => {
                // QR content should be the employee ID, not a secret.
                document.getElementById("employee").value =
                    decodedText.trim();
                stopScanner();
            },
            () => {}
        );
    } catch (error) {
        alert("Could not start QR scanner. Check camera permission or enter ID manually.");
        console.error(error);
        scanner = null;
    }
}

async function stopScanner() {
    if (scanner) {
        try {
            await scanner.stop();
            await scanner.clear();
        } catch (error) {
            console.log(error);
        }
        scanner = null;
    }
}

async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" },
            audio: false
        });
        document.getElementById("camera").srcObject = cameraStream;
    } catch (error) {
        alert("Camera permission failed. Allow camera access or upload an image.");
    }
}

function capturePhoto() {
    const video = document.getElementById("camera");
    if (!video.videoWidth) {
        alert("Start the webcam first.");
        return;
    }
    const canvas = document.getElementById("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    capturedData = canvas.toDataURL("image/jpeg", 0.85);
    document.getElementById("output").textContent = "Photo captured.";
}

async function submitCheck() {
    const employeeId = document.getElementById("employee").value.trim();
    const file = document.getElementById("upload").files[0];

    if (!employeeId) {
        alert("Scan the employee QR code or enter an employee ID.");
        return;
    }
    if (!capturedData && !file) {
        alert("Capture a webcam photo or upload an image.");
        return;
    }

    const output = document.getElementById("output");
    output.textContent = "Analyzing image. Please wait...";

    try {
        let imageData = capturedData;

        if (file) {
            imageData = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }

        const response = await fetch("/api/gate-check", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                employee_id: employeeId,
                image: imageData
            })
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");

        output.textContent =
            "Employee ID: " + data.employee_id + "\\n" +
            "Gate: " + data.gate + "\\n" +
            "Time: " + data.timestamp + "\\n" +
            "Result: " + data.status + "\\n\\n" +
            "PPE status:\\n" +
            Object.entries(data.ppe_status)
                .map(([item, status]) => item + ": " + status)
                .join("\\n") + "\\n\\n" +
            "Email status: " + data.email_status;

        const annotated = document.getElementById("annotated");
        annotated.src = "data:image/jpeg;base64," + data.image_data;
        annotated.style.display = "block";
        capturedData = null;
    } catch (error) {
        output.textContent = "Error: " + error.message;
    }
}
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML, gate=GATE_NAME)


@app.post("/api/gate-check")
def gate_check():
    try:
        payload = request.get_json(silent=True) or {}
        employee_id = str(payload.get("employee_id", "")).strip()
        image_value = payload.get("image", "")

        if not employee_id or len(employee_id) > 100:
            return jsonify(error="A valid employee ID is required."), 400

        if not image_value or not isinstance(image_value, str):
            return jsonify(error="A photo is required."), 400

        if "," in image_value and image_value.startswith("data:"):
            image_value = image_value.split(",", 1)[1]

        try:
            image_bytes = base64.b64decode(image_value, validate=True)
        except Exception:
            return jsonify(error="Invalid image data."), 400

        if len(image_bytes) > 8 * 1024 * 1024:
            return jsonify(error="Image must be smaller than 8 MB."), 413

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if image is None:
            return jsonify(error="The submitted file is not a valid image."), 400

        height, width = image.shape[:2]
        scale = min(640 / max(height, width), 1.0)
        if scale < 1:
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA
            )

        result = analyze_image(image)
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        record = {
            "timestamp": timestamp,
            "employee_id": employee_id,
            "gate": GATE_NAME,
            "status": result["overall"],
            "detected": result["detected"],
            "missing": result["missing"],
            "unverified": result["unverified"],
            "email_status": "pending"
        }

        record["email_status"] = send_supervisor_email(record)
        save_record(record)

        return jsonify(
            employee_id=employee_id,
            gate=GATE_NAME,
            timestamp=timestamp,
            status=result["overall"],
            detected=result["detected"],
            confidences=result["confidences"],
            ppe_status=result["statuses"],
            missing=result["missing"],
            unverified=result["unverified"],
            email_status=record["email_status"],
            image_data=result["image_data"]
        )

    except Exception:
        logger.exception("Gate PPE check failed.")
        return jsonify(
            error="The gate check failed. Please inspect the Render logs."
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000"))
    )
