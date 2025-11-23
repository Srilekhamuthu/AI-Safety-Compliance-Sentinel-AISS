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

def send_email_alert(violation_details, incident_id, image_filename):
    """
    Sends a direct email alert from the Vision Agent using Gmail SMTP.
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
    
    # Attach the evidence image
    try:
        with open(image_filename, 'rb') as f:
            file_data = f.read()
            msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=image_filename) 
    except FileNotFoundError:
        print(f"❌ Error attaching image: {image_filename} not found.")

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
# --- 3. ML SETUP & SPLASH SCREEN FUNCTION (PROFESSIONAL) ---
# ===================================================================

def display_splash_screen(cap):
    """
    Displays a branded splash screen for 3 seconds with a professional dark blue background.
    """
    frame_width = 640
    frame_height = 480
    
    # Dark Blue color in BGR: (128, 0, 0)
    dark_blue_color = (128, 0, 0) 
    
    # Create a solid dark blue background
    splash_frame = np.full((frame_height, frame_width, 3), dark_blue_color, dtype='uint8')
    
    # 1. Project Title (White for professionalism)
    title = "AISS: AI SAFETY COMPLIANCE SENTINEL"
    cv2.putText(
        splash_frame, 
        title, 
        (50, 150), 
        cv2.FONT_HERSHEY_DUPLEX, 
        0.65,                         # Unified Font Size 0.65
        (255, 255, 255),              # BRIGHT WHITE
        2
    )
    
    # 2. Status Message (Deep Gold/Orange for system status)
    status_msg = "Starting Vision Agent... Please Wait"
    cv2.putText(
        splash_frame, 
        status_msg, 
        (50, 210), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.65,                         # Unified Font Size 0.65
        (0, 165, 255),                # GOLD/ORANGE
        2
    )

    # Display the screen
    cv2.imshow("AISS - Vision Agent (Live Feed)", splash_frame)
    cv2.waitKey(3000) # Show for 3000 milliseconds (3 seconds)


print("Loading YOLOv8 Model (Vision Agent)...")
model = YOLO('yolov8n.pt') 

print("Connecting to Webcam (Press 'q' to quit)...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# CALL THE SPLASH SCREEN HERE!
display_splash_screen(cap)


# ===================================================================
# --- 4. REAL-TIME LOOP (Vision Agent Logic) ---
# ===================================================================

while True:
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
            
            # 1. Save Image Snapshot
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"violation_{timestamp_str}_{last_violation_id[:8]}.jpg"
            cv2.imwrite(image_filename, frame) 
            
            # 2. SEND THE DIRECT EMAIL ALERT! (The live action)
            send_email_alert(violation_type, last_violation_id, image_filename)
            
            # 3. Trigger the Orchestrator Simulation (for presentation purposes)
            event_payload = {
                "incident_id": last_violation_id,
                "violation_details": violation_type,  
                "manager_mobile": MANAGER_MOBILE,           
                "evidence_url": os.path.join(IMAGE_BASE_URL, image_filename).replace("\\", "/"),
                # ... other payload details ...
            }
            send_orchestrate_trigger_simulation(event_payload)
            
    else:
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
        0.65,           # Unified Font Size 0.65
        color, 
        2               
    )


    # Show the Video and Quit
    cv2.imshow("AISS - Vision Agent (Live Feed)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean Up
cap.release()
cv2.destroyAllWindows()
print("Sentinel Shut Down.")
