"""
Integration code to modify openai_realtime_sales_bot.py
This file contains the necessary changes to implement:
1. Audio saving with unique names
2. Noise suppression and audio enhancement
3. Transcription with Whisper
"""

import asyncio
import base64
import wave
import io
import os
from audio_manager import AudioManager
from transcription_engine import TranscriptionEngine

# Integration code for openai_realtime_sales_bot.py

class BotEnhancements:
    def __init__(self):
        self.audio_manager = AudioManager()
        self.transcription_engine = TranscriptionEngine()
        self.audio_buffers = {}  # Store audio chunks per stream
        self.current_audio_files = {}  # Track current audio files per stream
        
    def setup_audio_handling(self, stream_id: str):
        """Setup audio handling for a new stream"""
        self.audio_buffers[stream_id] = []
        self.current_audio_files[stream_id] = None
        
    def process_audio_chunk(self, stream_id: str, audio_data: bytes, chunk_index: int):
        """Process and save audio chunk"""
        # Save individual chunk
        filepath = self.audio_manager.save_audio_chunk(
            stream_id, 
            audio_data, 
            chunk_index
        )
        
        # Add to buffer for complete audio
        self.audio_buffers[stream_id].append(audio_data)
        
        # Transcribe chunk immediately
        transcription = self.transcription_engine.transcribe_audio(filepath)
        
        return {
            'filepath': filepath,
            'transcription': transcription
        }
        
    def save_complete_audio(self, stream_id: str) -> str:
        """Save complete audio when call ends"""
        if stream_id not in self.audio_buffers or not self.audio_buffers[stream_id]:
            return ""
            
        complete_path = self.audio_manager.save_complete_audio(
            stream_id, 
            self.audio_buffers[stream_id]
        )
        
        # Transcribe complete audio
        complete_transcription = self.transcription_engine.transcribe_audio(complete_path)
        
        # Cleanup buffer
        self.audio_buffers[stream_id] = []
        
        return complete_path
        
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files"""
        self.audio_manager.cleanup_old_files(max_age_hours)

# Usage example in openai_realtime_sales_bot.py modifications:

# 1. Add to __init__:
# self.bot_enhancements = BotEnhancements()

# 2. In handle_exotel_media:
# async def handle_exotel_media(self, stream_id: str, data: dict):
#     audio_b64 = data.get("media", {}).get("payload", "")
#     if audio_b64:
#         audio_data = base64.b64decode(audio_b64)
#         result = self.bot_enhancements.process_audio_chunk(
#             stream_id, 
#             audio_data, 
#             len(self.bot_enhancements.audio_buffers.get(stream_id, []))
#         )
#         # Use the transcription result as needed

# 3. In handle_exotel_stop:
# async def handle_exotel_stop(self, stream_id: str, data: dict):
#     complete_path = self.bot_enhancements.save_complete_audio(stream_id)
#     self.bot_enhancements.cleanup_old_files()
