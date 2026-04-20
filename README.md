# Camera Detection

This project contains two camera/video analysis tools:

- `bodylang`: a MediaPipe-based video pipeline that analyzes gaze, posture, gestures, restlessness, and low-confidence detection.
- `outfit`: a webcam capture script that sends one image to Gemini for structured interview outfit/setup feedback.

## Setup

Create a local environment and install the dependencies you need.

For the body-language pipeline:

```bash
pip install -r bodylang/requirements.txt
```

For the outfit checker, also install:

```bash
pip install google-genai pillow opencv-python pydantic
```

## Environment Variables

Copy the example env file and add your Gemini API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
GEMINI_API_KEY=your_api_key_here
```

`.env` is ignored by git. Do not commit real API keys.

## Outfit Checker

Run:

```bash
python outfit/outfit.py
```

The script opens the webcam, waits until your face is inside the alignment box, captures a clean frame, and sends the prompt plus the image to Gemini.

It asks Gemini to return JSON matching this schema:

```json
{
  "outfit_verdict": "Appropriate | Borderline | Not appropriate",
  "outfit_reason": "brief reason",
  "setup_verdict": "Good | Acceptable | Needs improvement",
  "setup_reason": "brief reason",
  "clothing_description": "visible clothing/accessories",
  "suggestions": ["up to 2 practical suggestions"]
}
```

The validated result is printed and written to `result.json`.

## BodyLang Offline Pipeline

Run a single video:

```bash
python bodylang/test_run.py --video path/to/video.mp4
```

Run a directory of videos:

```bash
python bodylang/test_run.py --video_dir path/to/videos --out_dir reports
```

The output report includes:

- engagement and look-away events
- gaze drift events
- posture stability and shifts
- gesture bursts and nod count
- restlessness windows
- low-confidence frame events
- `accuracy_confidence_ai` and `accuracy_confidence_reason` based on brightness and tracking quality

## Realtime BodyLang

Run webcam inference:

```bash
python bodylang/realtime.py
```

Useful options:

```bash
python bodylang/realtime.py --infer_fps 5 --cpu
python bodylang/realtime.py --camera 1 --mirror
python bodylang/realtime.py --no_display
```

The realtime GUI shows `MID` or `LOW` accuracy warnings when lighting is dim or face/pose tracking is unreliable.

## Notes

- The body-language pipeline uses MediaPipe task models. Missing models are downloaded automatically into the configured model directory.
- Default processing uses CPU and samples video at 10 FPS.
- Generated reports and local media files are ignored by git by default.
