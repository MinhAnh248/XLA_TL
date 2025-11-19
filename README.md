# Face Recognition App

This project is a simple face recognition GUI using OpenCV and optional FaceNet embeddings.

## Main files

- `face_recognition_app.py` - main GUI application. Uses Haar cascades for detection.
- `face_dataset/` - folder with subfolders per person containing face images.

## Important libraries

- `opencv-contrib-python` (provides `cv2`)
- `Pillow` (PIL)
- `numpy`
- Optional for better recognition:
  - `torch`
  - `facenet-pytorch`

## Quick start (Windows PowerShell)

```powershell
py -3 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
# If pip fails to install torch, try the official CPU wheel index:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch
python face_recognition_app.py
```

## Notes

- If `facenet-pytorch` is installed, use the `Build Embeddings` button to compute embeddings from images in `face_dataset/`.
- After building embeddings, the app will use embeddings for recognition (more accurate than color histogram).
- MongoDB: the app will try to use the connection string in the environment variable `MONGO_URI`.
  If `MONGO_URI` is not provided the app falls back to a default local MongoDB at
  `mongodb://localhost:27017/face_recognition` and will attempt to save/load embeddings
  to/from the `face_embeddings` collection.
- `👤 Register User` collects images from your webcam into `face_dataset/<name>/`.

If you want, I can run the environment setup here (install packages) or further improve the GUI (status bar, logs).
