import os
import time
import uuid
import wave
import base64
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class AudioManager:
    def __init__(self, base_dir: str = "audio_storage"):
        self.base_dir = base_dir
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories for audio storage"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "received"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "transcriptions"), exist_ok=True)
        
    def generate_unique_filename(self, stream_id: str, prefix: str = "audio") -> str:
        """Generate unique filename for audio files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{stream_id}_{timestamp}_{unique_id}.wav"
        
    def save_audio_chunk(self, stream_id: str, audio_data: bytes, chunk_index: int) -> str:
        """Save audio chunk to disk"""
        filename = self.generate_unique_filename(stream_id, f"chunk_{chunk_index}")
        filepath = os.path.join(self.base_dir, "received", filename)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(8000)  # 8kHz
            wav_file.writeframes(audio_data)
            
        logger.info(f"Saved audio chunk: {filepath}")
        return filepath
        
    def save_complete_audio(self, stream_id: str, audio_chunks: list) -> str:
        """Save complete audio from all chunks"""
        if not audio_chunks:
            return ""
            
        complete_filename = self.generate_unique_filename(stream_id, "complete")
        complete_path = os.path.join(self.base_dir, "processed", complete_filename)
        
        # Combine all chunks
        complete_audio = b''.join(audio_chunks)
        
        with wave.open(complete_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes(complete_audio)
            
        logger.info(f"Saved complete audio: {complete_path}")
        return complete_path
        
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files based on age"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        for directory in ["received", "processed", "transcriptions"]:
            dir_path = os.path.join(self.base_dir, directory)
            if os.path.exists(dir_path):
                for filename in os.listdir(dir_path):
                    filepath = os.path.join(dir_path, filename)
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old file: {filepath}")
