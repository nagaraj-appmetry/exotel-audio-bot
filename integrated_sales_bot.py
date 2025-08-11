import asyncio
import io
import wave
import numpy as np
import websockets
import json
import logging
import base64
import time
import struct
import ssl
import os
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from config import Config
import noisereduce as nr
import soundfile as sf
import whisper
import requests
from gtts import gTTS
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegratedSalesBot:
    def __init__(self):
        # Validate configuration first
        Config.validate()
        
        self.exotel_connections: Dict[str, Dict[str, Any]] = {}
        self.audio_storage: Dict[str, Dict[str, Any]] = {}
        
        # Audio management
        self.audio_buffers: Dict[str, list] = {}
        self.chunk_counters: Dict[str, int] = {}
        self.transcription_engine = whisper.load_model("base")
        
        # Custom server configuration
        self.custom_server_url = "http://your-custom-server.com/api/chat"  # Update this
        
        # Audio settings
        self.sample_rate = Config.SAMPLE_RATE
        self.bytes_per_chunk = int((self.sample_rate * 2 * Config.BUFFER_SIZE_MS) / 1000)
        
        # Ensure directories exist
        self.setup_directories()
        
        logger.info("🤖 Integrated Sales Bot initialized!")
        logger.info(f"🔊 Audio chunk size: {Config.BUFFER_SIZE_MS}ms ({self.bytes_per_chunk} bytes)")
        logger.info(f"🏢 Company: {Config.COMPANY_NAME}")
        logger.info(f"👤 Sales Rep: {Config.SALES_REP_NAME}")

    def setup_directories(self):
        """Create necessary directories for audio storage"""
        directories = ["audio_storage", "audio_storage/received", 
                      "audio_storage/processed", "audio_storage/transcriptions"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def generate_unique_filename(self, stream_id: str, prefix: str = "audio") -> str:
        """Generate unique filename for audio files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{prefix}_{stream_id}_{timestamp}_{unique_id}.wav"

    def save_audio_chunk(self, stream_id: str, audio_data: bytes, chunk_index: int) -> str:
        """Save audio chunk to disk with noise suppression"""
        filename = self.generate_unique_filename(stream_id, f"chunk_{chunk_index}")
        filepath = os.path.join("audio_storage", "received", filename)
        
        # Apply noise suppression
        processed_audio = self.apply_noise_suppression(audio_data)
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(processed_audio)
            
        logger.info(f"💾 Saved audio chunk: {filepath}")
        return filepath

    def save_complete_audio(self, stream_id: str) -> str:
        """Save complete audio from all chunks"""
        if stream_id not in self.audio_buffers or not self.audio_buffers[stream_id]:
            return ""
            
        complete_filename = self.generate_unique_filename(stream_id, "complete")
        complete_path = os.path.join("audio_storage", "processed", complete_filename)
        
        # Combine all chunks
        complete_audio = b''.join(self.audio_buffers[stream_id])
        
        with wave.open(complete_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(complete_audio)
            
        logger.info(f"💾 Saved complete audio: {complete_path}")
        
        # Transcribe complete audio
        self.transcribe_audio(complete_path)
        
        return complete_path

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        try:
            result = self.transcription_engine.transcribe(audio_path)
            text = result.get("text", "").strip()
            
            # Save transcription
            filename = os.path.basename(audio_path).replace(".wav", ".txt")
            transcription_path = os.path.join("audio_storage", "transcriptions", filename)
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            logger.info(f"📝 Transcription saved: {transcription_path}")
            return text
        except Exception as e:
            logger.error(f"❌ Error transcribing audio: {e}")
            return ""

    async def get_response_from_custom_server(self, user_text: str) -> str:
        """Get response from your custom server"""
        try:
            payload = {
                "query": user_text,
                "context": {
                    "company": Config.COMPANY_NAME,
                    "products": Config.PRODUCTS,
                    "sales_rep": Config.SALES_REP_NAME
                }
            }
            
            response = requests.post(self.custom_server_url, json=payload, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "Thank you for your interest. How can I help you today?")
            
        except Exception as e:
            logger.error(f"❌ Error getting response from custom server: {e}")
            return "I'm here to help. What would you like to know about our products?"

    def text_to_speech(self, text: str) -> bytes:
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                tts.save(temp_file.name)
                
                # Convert MP3 to PCM
                import subprocess
                pcm_file = temp_file.name.replace('.mp3', '.pcm')
                
                subprocess.run([
                    'ffmpeg', '-i', temp_file.name, '-f', 's16le', '-ar', '8000', '-ac', '1', pcm_file
                ], check=True, capture_output=True)
                
                # Read PCM data
                with open(pcm_file, 'rb') as f:
                    pcm_data = f.read()
                
                # Cleanup
                os.unlink(temp_file.name)
                os.unlink(pcm_file)
                
                return pcm_data
                
        except Exception as e:
            logger.error(f"❌ Error converting text to speech: {e}")
            # Return a simple beep as fallback
            return self.generate_test_tone(duration_ms=500)

    def apply_noise_suppression(self, audio_data: bytes) -> bytes:
        """Apply noise suppression and audio enhancement"""
        try:
            import numpy as np
            
            # Convert to numpy array
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)
            
            # Noise gate
            noise_threshold = Config.NOISE_THRESHOLD
            audio_samples = np.where(np.abs(audio_samples) < noise_threshold, 0, audio_samples)
            
            # High-pass filter
            if len(audio_samples) > 10:
                window_size = min(5, len(audio_samples) // 2)
                moving_avg = np.convolve(audio_samples.astype(np.float32), 
                                       np.ones(window_size)/window_size, mode='same')
                audio_samples = audio_samples - moving_avg.astype(np.int16) * 0.1
            
            # Compression
            max_val = np.max(np.abs(audio_samples))
            if max_val > 0:
                compression_ratio = 0.8
                normalized = audio_samples.astype(np.float32) / max_val
                compressed = np.sign(normalized) * (np.abs(normalized) ** compression_ratio)
                audio_samples = (compressed * max_val * 0.9).astype(np.int16)
            
            return audio_samples.tobytes()
            
        except Exception as e:
            logger.error(f"❌ Error in noise suppression: {e}")
            return audio_data

    def generate_test_tone(self, frequency=440, duration_ms=1000, sample_rate=8000, amplitude=0.5):
        """Generate test tone"""
        t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
        tone = amplitude * np.sin(2 * np.pi * frequency * t)
        audio_data = np.int16(tone * 32767)
        
        with io.BytesIO() as buffer:
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            return buffer.getvalue()

    async def handle_exotel_websocket(self, websocket):
        """Handle incoming WebSocket connection from Exotel"""
        stream_id = "unknown"
        
        try:
            logger.info(f"📞 NEW SALES CALL from Exotel: {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event = data.get("event", "")
                    
                    # Extract stream ID
                    if "streamSid" in data:
                        stream_id = data["streamSid"]
                    elif "stream_sid" in data:
                        stream_id = data["stream_sid"]
                    
                    # Initialize storage for new stream
                    if stream_id not in self.audio_storage:
                        self.audio_storage[stream_id] = {
                            "websocket": websocket,
                            "start_time": time.time(),
                            "audio_chunks": [],
                            "transcriptions": []
                        }
                        self.audio_buffers[stream_id] = []
                        self.chunk_counters[stream_id] = 0
                    
                    # Handle events
                    if event == "connected":
                        await self.handle_exotel_connected(stream_id, data)
                    elif event == "start":
                        await self.handle_exotel_start(stream_id, data)
                    elif event == "media":
                        await self.handle_exotel_media(stream_id, data)
                    elif event == "mark":
                        await self.handle_exotel_mark(stream_id, data)
                    elif event == "clear":
                        await self.handle_exotel_clear(stream_id, data)
                    elif event == "stop":
                        await self.handle_exotel_stop(stream_id, data)
                        break
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing Exotel message: {e}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔚 EXOTEL CONNECTION CLOSED NORMALLY: {stream_id} (code: {e.code})")
        except Exception as e:
            logger.error(f"❌ Exotel WebSocket error: {e}")
        finally:
            logger.info(f"🧹 CLEANING UP CONNECTION: {stream_id}")
            await self.cleanup_connections(stream_id)

    async def handle_exotel_connected(self, stream_id: str, data: dict):
        """Handle Exotel connected event"""
        logger.info(f"✅ EXOTEL CONNECTED: {stream_id}")
        
        # Send test tone
        test_tone = self.generate_test_tone()
        test_audio_b64 = base64.b64encode(test_tone).decode()
        
        test_message = {
            "event": "media",
            "streamSid": stream_id,
            "media": {
                "payload": test_audio_b64,
                "timestamp": str(int(time.time() * 1000)),
                "sequenceNumber": "1"
            }
        }
        
        await self.audio_storage[stream_id]["websocket"].send(json.dumps(test_message))
        logger.info(f"🔊 TEST TONE SENT for {stream_id}")

    async def handle_exotel_start(self, stream_id: str, data: dict):
        """Handle Exotel start event"""
        logger.info(f"🚀 SALES CALL STARTED: {stream_id}")

    async def handle_exotel_media(self, stream_id: str, data: dict):
        """Handle incoming audio from Exotel customer"""
        try:
            audio_b64 = data.get("media", {}).get("payload", "")
            if not audio_b64:
                return
                
            # Decode audio
            audio_data = base64.b64decode(audio_b64)
            
            # Save chunk
            chunk_index = self.chunk_counters[stream_id]
            filepath = self.save_audio_chunk(stream_id, audio_data, chunk_index)
            self.chunk_counters[stream_id] += 1
            
            # Add to buffer
            self.audio_buffers[stream_id].append(audio_data)
            
            # Transcribe chunk
            transcription = self.transcribe_audio(filepath)
            if transcription:
                self.audio_storage[stream_id]["transcriptions"].append(transcription)
                
                # Get response from custom server
                response_text = await self.get_response_from_custom_server(transcription)
                
                # Convert to speech
                response_audio = self.text_to_speech(response_text)
                
                # Send back to user
                response_b64 = base64.b64encode(response_audio).decode()
                
                media_message = {
                    "event": "media",
                    "streamSid": stream_id,
                    "media": {
                        "payload": response_b64,
                        "timestamp": str(int(time.time() * 1000)),
                        "sequenceNumber": str(chunk_index + 1)
                    }
                }
                
                await self.audio_storage[stream_id]["websocket"].send(json.dumps(media_message))
                logger.info(f"📢 Response sent for {stream_id}: {response_text[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Error handling media event: {e}")

    async def handle_exotel_mark(self, stream_id: str, data: dict):
        """Handle Exotel mark event"""
        mark_name = data.get("mark", {}).get("name", "unknown")
        logger.info(f"📍 EXOTEL MARK: {mark_name} for {stream_id}")

    async def handle_exotel_clear(self, stream_id: str, data: dict):
        """Handle Exotel clear event"""
        logger.info(f"🧹 EXOTEL CLEAR - INTERRUPTING BOT SPEECH: {stream_id}")
        # Implementation for clearing audio buffers

    async def handle_exotel_stop(self, stream_id: str, data: dict):
        """Handle Exotel stop event"""
        logger.info(f"🛑 SALES CALL ENDED: {stream_id}")
        complete_path = self.save_complete_audio(stream_id)
        if complete_path:
            logger.info(f"💾 Complete call audio saved: {complete_path}")

    async def cleanup_connections(self, stream_id: str):
        """Clean up connections and storage"""
        try:
            if stream_id in self.audio_storage:
                del self.audio_storage[stream_id]
            if stream_id in self.audio_buffers:
                del self.audio_buffers[stream_id]
            if stream_id in self.chunk_counters:
                del self.chunk_counters[stream_id]
                
            # Cleanup old files
            self.cleanup_old_files()
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old audio files"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            for directory in ["received", "processed", "transcriptions"]:
                dir_path = os.path.join("audio_storage", directory)
                if os.path.exists(dir_path):
                    for filename in os.listdir(dir_path):
                        filepath = os.path.join(dir_path, filename)
                        if os.path.getmtime(filepath) < cutoff_time:
                            os.remove(filepath)
                            logger.info(f"🧹 Cleaned up old file: {filepath}")
                            
        except Exception as e:
            logger.error(f"❌ Error cleaning up files: {e}")

async def main():
    """Main function to start the Integrated Sales Bot"""
    try:
        bot = IntegratedSalesBot()
        
        logger.info(f"🚀 Starting Integrated Sales Bot Server on {Config.SERVER_HOST}:{Config.SERVER_PORT}")
        logger.info("📞 Ready for Exotel streaming connections!")
        
        async with websockets.serve(
            bot.handle_exotel_websocket,
            Config.SERVER_HOST,
            Config.SERVER_PORT
        ):
            logger.info(f"✅ Server running at ws://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
            logger.info("🎯 Waiting for calls...")
            await asyncio.Future()
            
    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
