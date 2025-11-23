import cv2
from ultralytics import YOLO
import json
import time
import requests
import datetime
import os
import uuid
import smtplib
from email.message import EmailMessage
import numpy as np

# --- STREAMLIT IMPORTS ---
import streamlit as st
import tempfile
# -------------------------

# ===================================================================
# --- 1. CONFIGURATION: FINAL SETTINGS ---
# ===================================================================

# 1. SENDER EMAIL (Your Gmail address)
SENDER_EMAIL = "lekhasri8877@gmail.com"

# 2. APP PASSWORD (Your unique 16-character code)
SENDER_PASSWORD = "zurjdrwkqrhksqhj"

# 3. RECIPIENT EMAIL (The final email address for the alert)
RECIPIENT_EMAIL = "srilekhamuthu9514@gmail.com"

# Orchestrate Simulation (Kept for the full workflow explanation)
ORCHESTRATE_API_URL = "http://localhost:8080/api/orchestrate/trigger"
IMAGE_BASE_URL = "https://aiss-project-storage.com/violations/"

# Final Contact/ID Information
MANAGER_MOBILE = "9514102647"
PERSON_ID_VIOLATOR = "P001_Contractor_North"

# Define the object IDs for detection (PLACEHOLDERS for a custom PPE model):
PERSON_ID = 0
HELMET_ID = 4
EYEGLASS_ID = 11
GLOVE_ID = 39
JACKET_ID = 27

VIOLATION_THRESHOLD_SEC = 5 # How long the violation must last before triggering.

# Variables to track the state
violation_start_time = None
violation_active = False
last_violation_id = None
violation_type = ""

# ===================================================================
# --- 2. ALERT FUNCTIONS (EMAIL & SIMULATION) ---
# ===================================================================

def send_email_alert(violation_details, incident_id, image_data):
    """
    Sends a direct email alert from the Vision Agent using Gmail SMTP.
    Takes image_data (bytes) instead of filename, as Streamlit uses temp files.
    """
    print("\n--- DIRECT EMAIL ALERT ATTEMPT ---")
    
    msg = EmailMessage()
    msg['Subject'] = f"🚨 URGENT PPE VIOLATION: {violation_details}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    body = f"""
    Safety Violation Detected by Vision Agent (ID: {incident_id[:8]})
    --------------------------------------------------
    Person ID: {PERSON_ID_VIOLATOR}
    Location: Factory Zone 3 (Near Main Entrance)
    Missing PPE: {violation_details}
    Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    Action Required: Immediate intervention.
    Evidence Image attached below.
    """
    msg.set_content(body)
    
    # Attach the evidence image using the image data (bytes)
    msg.add_attachment(image_data, maintype='image', subtype='jpeg', filename=f'violation_{incident_id[:8]}.jpg') 

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
            print("✅ Direct Email Alert SUCCESS! Message sent via SMTPLIB.")
            return True
    except Exception as e:
        print(f"❌ Email Failed. Check SENDER_EMAIL/PASSWORD/App Password setup: {e}")
        return False
    finally:
        print("-------------------------------------------\n")


def send_orchestrate_trigger_simulation(payload):
    """
    This is kept to simulate the secondary, larger workflow for the judges.
    """
    print("--- WATSONX ORCHESTRATE TRIGGER SIMULATION ---")
    print(f"⚠️ Trigger Simulated: Workflow initiated in the cloud with mobile number: {payload['manager_mobile']}")
    print("-------------------------------------------\n")


# ===================================================================
# --- 3. ML SETUP & SPLASH SCREEN FUNCTION (STREAMLIT LOGIC) ---
# ===================================================================

# NOTE: The original display_splash_screen is removed as it uses cv2.imshow, 
# which is incompatible with Streamlit. We'll use Streamlit components instead.

print("Loading YOLOv8 Model (Vision Agent)...")
model = YOLO('yolov8n.pt') 

# --- STREAMLIT INTERFACE SETUP ---
st.title("🛡️ AI Safety Compliance Sentinel Demo")
st.markdown("---")

# 1. Status Area Placeholder (Starts as splash screen equivalent)
status_box = st.empty()
status_box.info("Starting Vision Agent... Please Upload a Video File.")

# 2. File Uploader component (Replaces cv2.VideoCapture(0))
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

# 3. Main Logic runs only IF a file is uploaded
if uploaded_file is not None:

    # 1. Save the uploaded file temporarily so OpenCV can read it
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_file.write(uploaded_file.read())
    temp_video_file.close()
    video_path = temp_video_file.name
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Placeholder for the processed video output display
    frame_placeholder = st.empty()
    
    if not cap.isOpened():
        status_box.error("Error: Could not open the video file.")
        os.unlink(video_path) # Clean up temp file
        st.stop()
        
    status_box.info("Analysis Started... Processing video frames.")

    # ===================================================================
    # --- 4. REAL-TIME LOOP (Vision Agent Logic) ---
    # ===================================================================

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        detected_classes = results[0].boxes.cls.tolist()
        
        person_detected = PERSON_ID in detected_classes
        
        # --- CHECK PPE COMPLIANCE ---
        missing_items = []
        
        if person_detected:
            if not HELMET_ID in detected_classes:
                missing_items.append("Helmet")
                
            if not EYEGLASS_ID in detected_classes:
                missing_items.append("Eye Glass")
                
            if not GLOVE_ID in detected_classes:
                missing_items.append("Gloves")
                
            if not JACKET_ID in detected_classes:
                missing_items.append("Jacket")

        current_violation = len(missing_items) > 0
        current_time = time.time()
        
        # --- Default Status Setting ---
        status_text = "STATUS: CLEAR"
        color = (0, 255, 0) # Green 
        
        # --- AGENTIC STATE MANAGEMENT AND TRIGGER LOGIC ---
        if current_violation:
            violation_type = ", ".join(missing_items)
            
            if violation_start_time is None:
                violation_start_time = current_time
                print(f"⏳ Violation Detected: Missing {violation_type}. Starting timer...")
                last_violation_id = str(uuid.uuid4())
                
            elapsed_time = current_time - violation_start_time
            
            # Check for trigger condition
            if elapsed_time >= VIOLATION_THRESHOLD_SEC and not violation_active:
                # --- PERSISTENT VIOLATION CONFIRMED: TRIGGER NOW! ---
                violation_active = True
                
                # 1. Encode Image Snapshot to memory (Streamlit/Cloud friendly)
                success, buffer = cv2.imencode('.jpg', frame)
                if success:
                    image_data = buffer.tobytes()
                else:
                    image_data = None
                
                # 2. SEND THE DIRECT EMAIL ALERT! 
                if image_data:
                    send_email_alert(violation_type, last_violation_id, image_data)
                
                # 3. Trigger the Orchestrator Simulation (for presentation purposes)
                event_payload = {
                    "incident_id": last_violation_id,
                    "violation_details": violation_type,  
                    "manager_mobile": MANAGER_MOBILE, 
                    "evidence_url": os.path.join(IMAGE_BASE_URL, f"violation_{last_violation_id[:8]}.jpg").replace("\\", "/"),
                }
                send_orchestrate_trigger_simulation(event_payload)
                
        else:
            # Violation has ceased or no person detected
            violation_start_time = None
            violation_type = ""
            violation_active = False 
            last_violation_id = None
            
            
        # --- VISUAL STATUS UPDATE LOGIC ---
        if violation_active:
            status_text = f"🚨 ALERT SENT: MISSING {violation_type.upper()}"
            color = (0, 0, 255) # Red
        elif violation_start_time is not None:
            time_left = VIOLATION_THRESHOLD_SEC - (current_time - violation_start_time)
            status_text = f"MISSING {violation_type.upper()}: Holding for {time_left:.1f}s..."
            color = (0, 165, 255) # Orange/Yellow
        elif person_detected:
            status_text = "STATUS: Compliance OK (Person Present)"
            color = (0, 255, 0) # Green


        # Draw the status on the video feed
        cv2.putText(
            annotated_frame, 
            status_text, 
            (50, 50), 
            cv2.FONT_HERSHEY_DUPLEX, 
            0.65, 
            color, 
            2 
        )

        # --- REPLACE cv2.imshow with st.image ---
        # Convert the OpenCV BGR image to RGB for web display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # Update the placeholder image on the Streamlit page
        frame_placeholder.image(annotated_frame_rgb, channels="RGB", caption="Live AI Analysis", use_column_width=True)

        # Use a small waitKey equivalent (optional, but can slow down processing slightly for visual appeal)
        # st.empty() # Or similar command to prevent fast loop overwhelming the app
        
        
    # --- Cleanup and Finish ---
    cap.release()
    os.unlink(video_path) # Delete the temporary file
    
    status_box.success("✅ Video Analysis Complete! Upload a new file to continue.")

# If Streamlit is running, this code block is not used, but kept for full compatibility
# if st.util.is_running_interactive_mode() is False:
#     print("Sentinel Shut Down.")
