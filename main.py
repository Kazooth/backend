from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from filters import clean_voice

app = FastAPI(title="Noise Cleaner API")

# Permitir acceso desde cualquier origen (para desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/filter_audio")
async def filter_audio(file: UploadFile = File(...),
                       low_cut: int = Form(300),
                       high_cut: int = Form(3400)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_in:
            temp_in.write(contents)
            temp_in_path = temp_in.name

        audio_data, sr = librosa.load(temp_in_path, sr=44100, mono=True)
        cleaned = clean_voice(audio_data, sr, low_cut, high_cut)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_out:
            sf.write(temp_out.name, cleaned, sr)
            output_path = temp_out.name

        filename = os.path.basename(output_path)
        return FileResponse(output_path, filename=f"cleaned_{filename}", media_type="audio/wav")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)

@app.get("/")
def root():
    return {"message": "Noise Cleaner API running!"}
