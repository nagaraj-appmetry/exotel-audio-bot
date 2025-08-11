import os
import logging
import whisper
from datetime import datetime

logger = logging.getLogger(__name__)

class TranscriptionEngine:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        self.transcriptions_dir = "audio_storage/transcriptions"
        os.makedirs(self.transcriptions_dir, exist_ok=True)

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file and save transcription to disk"""
        try:
            result = self.model.transcribe(audio_path)
            text = result.get("text", "")
            filename = os.path.basename(audio_path).replace(".wav", ".txt")
            transcription_path = os.path.join(self.transcriptions_dir, filename)
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Transcription saved: {transcription_path}")
            return text
        except Exception as e:
            logger.error(f"Error transcribing audio {audio_path}: {e}")
            return ""
