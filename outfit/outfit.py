import cv2
from google import genai
from PIL import Image
import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from .env without overriding existing env vars."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


ROOT_DIR = Path(__file__).resolve().parents[1]
_load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    print(
        "Missing GEMINI_API_KEY. Copy .env.example to .env and set your Gemini API key.",
        file=sys.stderr,
    )
    sys.exit(1)


# ==========================================
# 0. STRUCTURED JSON SCHEMA
# ==========================================

class InterviewImageAssessment(BaseModel):
    outfit_verdict: Literal["Appropriate", "Borderline", "Not appropriate"] = Field(
        description="Whether the outfit is appropriate for an online medical school interview."
    )
    outfit_reason: str = Field(
        description="Brief reason for the outfit verdict."
    )
    setup_verdict: Literal["Good", "Acceptable", "Needs improvement"] = Field(
        description="Whether the visual setup is suitable for an online interview."
    )
    setup_reason: str = Field(
        description="Brief reason for the setup verdict."
    )
    clothing_description: str = Field(
        description="Concise description of the visible clothing and accessories."
    )
    suggestions: list[str] = Field(
        description="Up to 2 short suggestions for improvement."
    )


# ==========================================
# 1. SETUP NEW GEMINI API
# ==========================================

client = genai.Client(api_key=API_KEY)


# ==========================================
# 2. SETUP WEBCAM & FACE DETECTION
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Check your macOS Privacy & Security settings for Camera access.")
    sys.exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

captured_image_pil = None
print("Starting camera... Press 'q' to quit manually.")


# ==========================================
# 3. MAIN CAMERA LOOP
# ==========================================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    clean_frame = frame.copy()
    H, W, _ = frame.shape

    zone_x1, zone_y1 = int(W * 0.35), int(H * 0.10)
    zone_x2, zone_y2 = int(W * 0.65), int(H * 0.45)

    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Align face here",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    face_in_zone = False

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_center_x = x + w // 2
        face_center_y = y + h // 2

        if zone_x1 < face_center_x < zone_x2 and zone_y1 < face_center_y < zone_y2:
            face_in_zone = True
            cv2.putText(
                frame,
                "LOCKED ON! Capturing...",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            cv2.imshow("Webcam", frame)
            cv2.waitKey(500)

            rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
            captured_image_pil = Image.fromarray(rgb_frame)
            break

    cv2.imshow("Webcam", frame)

    if face_in_zone:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# ==========================================
# 4. SEND TO GEMINI WITH STRUCTURED OUTPUT
# ==========================================

if captured_image_pil:
    print("\nImage captured successfully! Analyzing your interview presentation...")

    prompt = """
Assess this image for an online medical school interview.

Only evaluate visible presentation.
Do not infer personality, confidence, intelligence, honesty, emotional state, or suitability for medicine.

AirPods and earbuds are acceptable and should not be treated as inappropriate.

Assess:
1. Outfit appropriateness
- professional
- neat
- modest
- not visually distracting
- suitable for an online interview

2. Visual setup
- face clearly visible
- reasonably well lit
- framed appropriately for an interview
- background clean and not overly distracting

3. Brief clothing description
- garment type
- color
- visible patterns
- visible accessories

Keep reasons concise.
Suggestions must be short and practical.
Return up to 2 suggestions only.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, captured_image_pil],
            config={
                "response_mime_type": "application/json",
                "response_json_schema": InterviewImageAssessment.model_json_schema(),
            },
        )

        # Parse and validate the JSON returned by Gemini
        result = InterviewImageAssessment.model_validate_json(response.text)

        print("\n=== GEMINI ANALYSIS (JSON) ===")
        print(result.model_dump_json(indent=2))
        print("==============================\n")

        # Optional: save to a file
        with open("result.json", "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        print("Saved to result.json")

    except Exception as e:
        print(f"An error occurred while contacting Gemini: {e}")

else:
    print("No image was captured. Exiting.")
