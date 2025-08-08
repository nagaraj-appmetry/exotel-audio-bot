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
import os
from typing import Dict, Any, Optional
from config import Config
import noisereduce as nr
import soundfile as sf
import speech_recognition as sr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeSalesBot:
    """
    RealtimeSalesBot - OpenAI removed version.
    Preserves original class layout, logging, and flow,
    but contains NO OpenAI-related logic.
    """

    def __init__(self):
        # Validate configuration first
        Config.validate()

        # Initialize connection dictionaries
        self.exotel_connections: Dict[str, Dict[str, Any]] = {}

        # Audio buffering for better conversation flow
        self.audio_buffers: Dict[str, bytes] = {}  # Buffer audio for configured chunk sizes
        self.buffer_size_ms = Config.BUFFER_SIZE_MS
        self.sample_rate = Config.SAMPLE_RATE
        self.bytes_per_chunk = int((self.sample_rate * 2 * self.buffer_size_ms) / 1000)  # 16-bit = 2 bytes per sample
        self.silence_start_time: Dict[str, Optional[float]] = {}
        self.last_bot_reply_time: Dict[str, float] = {}
        # Sales Bot Configuration
        self.sales_instructions = Config.get_sales_instructions()

        # Silence detection state variables
        self.static_played_in_this_silence: Dict[str, bool] = {}  # stream_id -> bool flag for played static audio
        self.bot_is_speaking: Dict[str, bool] = {}
        self.last_was_silence: Dict[str, bool] = {}               # stream_id -> bool flag to track last chunk silence

        # Logging startup info
        logger.info("🤖 Realtime Sales Bot initialized (OpenAI logic removed)!")
        logger.info(f"🔊 Audio buffering: {self.buffer_size_ms}ms chunks ({self.bytes_per_chunk} bytes)")
        logger.info(f"🏢 Company: {Config.COMPANY_NAME}")
        logger.info(f"👤 Sales Rep: {Config.SALES_REP_NAME}")
        logger.info(f"📦 Products: {', '.join([p['name'] for p in Config.PRODUCTS])}")

    @staticmethod
    def reduce_noise_in_file(input_file, output_file):
        """Utility: reduce noise in a file using noisereduce (kept from original)."""
        try:
            audio_data, sr = sf.read(input_file)
            noisy_part = audio_data[0:10000] if len(audio_data) > 10000 else audio_data
            reduced_noise = nr.reduce_noise(y=audio_data, sr=sr, y_noise=noisy_part)
            sf.write(output_file, reduced_noise, sr)
            logger.info(f"🔊 Noise-reduced file written to {output_file}")
        except Exception as e:
            logger.error(f"❌ Noise reduction failed: {e}")

    async def handle_exotel_websocket(self, websocket):
        """Handle incoming WebSocket connection from Exotel"""
        stream_id = "unknown"

        try:
            logger.info(f"📞 NEW SALES CALL from Exotel: {websocket.remote_address}")

            # Set up connection keep-alive and error handling
            async for message in websocket:
                try:
                    # Primary incoming message from Exotel (JSON)
                    data = json.loads(message)
                    event = data.get("event", "")

                    # Extract stream ID (common keys used by Exotel)
                    if "streamSid" in data:
                        stream_id = data["streamSid"]
                    elif "stream_sid" in data:
                        stream_id = data["stream_sid"]
                    elif "stream_id" in data:
                        stream_id = data["stream_id"]

                    logger.info(f"🆔 STREAM ID: {stream_id}")
                    logger.info(f"🎯 EVENT: '{event}' for {stream_id}")

                    # Store Exotel connection (first time)
                    if stream_id not in self.exotel_connections:
                        self.exotel_connections[stream_id] = {
                            "websocket": websocket,
                            "start_time": time.time()
                        }
                        logger.info(f"📞 NEW EXOTEL CONNECTION: {stream_id}")

                    # Dispatch events
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
                        break  # Exit the message loop after stop event
                    else:
                        logger.info(f"🔄 UNHANDLED EVENT: {event} for {stream_id}")

                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing Exotel message: {e}")

        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"🔚 EXOTEL CONNECTION CLOSED NORMALLY: {stream_id} (code: {getattr(e, 'code', 'unknown')})")
        except Exception as e:
            logger.error(f"❌ Exotel WebSocket error: {e}")
        finally:
            logger.info(f"🧹 CLEANING UP CONNECTION: {stream_id}")
            await self.cleanup_connections(stream_id)

    async def handle_exotel_connected(self, stream_id: str, data: dict):
        """Handle Exotel connected event"""
        logger.info(f"✅ EXOTEL CONNECTED: {stream_id}")

        # Send immediate acknowledgment to Exotel (test tone)
        try:
            exotel_ws = self.exotel_connections[stream_id]["websocket"]

            # Generate a short test tone and send as media
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

            await exotel_ws.send(json.dumps(test_message))
            logger.info(f"🔊 TEST TONE SENT to confirm audio pipeline for {stream_id}")

        except Exception as e:
            logger.error(f"❌ Error sending test tone: {e}")

        # NOTE: OpenAI connection removed. This method used to call connect_to_openai(stream_id)

    async def handle_exotel_start(self, stream_id: str, data: dict):
        """Handle Exotel start event"""
        logger.info(f"🚀 SALES CALL STARTED: {stream_id}")

    # Silence detection / thresholds (kept from original)
    MIN_SPEECH_LENGTH = 96000   # ~6 sec of speech at 8kHz
    SILENCE_THRESHOLD = 300
    SILENCE_MIN_DURATION = 1.5  # seconds before bot replies
    COOLDOWN_SECONDS = 3.0      # seconds before bot can reply again

    def is_silence(self, audio_bytes: bytes, threshold=None) -> bool:
        """Return True if given PCM16 bytes are below amplitude threshold."""
        if threshold is None:
            threshold = self.SILENCE_THRESHOLD
        if not audio_bytes:
            return True  # treat empty as silence
        try:
            audio_samples = np.frombuffer(audio_bytes, dtype=np.int16)
            if audio_samples.size == 0:
                return True
            max_amplitude = np.max(np.abs(audio_samples))
            return max_amplitude < threshold
        except Exception as e:
            logger.error(f"❌ Error computing silence: {e}")
            return True

    async def send_static_audio_response(self, stream_id: str, wav_filepath: str):
        """Send static WAV file to Exotel (reads file, encodes to base64, sends 'media' event)."""
        try:
            if stream_id not in self.exotel_connections:
                logger.warning(f"⚠️ No Exotel connection for {stream_id}")
                return

            exotel_ws = self.exotel_connections[stream_id]["websocket"]

            # Read file bytes
            with open(wav_filepath, "rb") as f:
                pcm_data = f.read()

            # In some setups Exotel expects G.711-uLaw. If your file is already u-law,
            # send directly; otherwise you may need to convert. Here we send the raw bytes
            # (the original code did the same) — change if needed for your Exotel config.
            audio_b64 = base64.b64encode(pcm_data).decode()

            media_message = {
                "event": "media",
                "streamSid": stream_id,
                "media": {
                    "payload": audio_b64,
                    "timestamp": str(int(time.time() * 1000)),
                    "sequenceNumber": str(int(time.time()))
                }
            }

            await exotel_ws.send(json.dumps(media_message))
            logger.info(f"🔊 Sent static audio file to user for {stream_id}")

        except Exception as e:
            logger.error(f"❌ Error sending static audio response: {e}")

    async def handle_exotel_media(self, stream_id: str, data: dict):
        """Handle incoming media chunks from Exotel (user speech)."""
        try:
            chunk = base64.b64decode(data['media']['payload'])

            if stream_id not in self.audio_buffers:
                self.audio_buffers[stream_id] = b""
                self.static_played_in_this_silence[stream_id] = False
                self.last_was_silence[stream_id] = False
                self.bot_is_speaking[stream_id] = False
                self.silence_start_time[stream_id] = None
                self.last_bot_reply_time[stream_id] = 0

            try:
                processed = self.apply_noise_suppression(chunk)
            except Exception:
                processed = chunk

            self.audio_buffers[stream_id] += processed
            silence = self.is_silence(processed)

            static_path = os.path.join(os.path.dirname(__file__), "engines", "bot_fixed.wav")

            if silence:
                # Mark silence start time if not already
                if self.silence_start_time[stream_id] is None:
                    self.silence_start_time[stream_id] = time.time()

                silence_duration = time.time() - self.silence_start_time[stream_id]

                # Only trigger bot reply after sustained silence, speech length met, cooldown passed
                if (silence_duration >= self.SILENCE_MIN_DURATION and
                    len(self.audio_buffers[stream_id]) > self.MIN_SPEECH_LENGTH and
                    not self.static_played_in_this_silence[stream_id] and
                    not self.bot_is_speaking[stream_id] and
                    (time.time() - self.last_bot_reply_time[stream_id] > self.COOLDOWN_SECONDS)):

                    # Save customer audio
                    transcript = self.transcribe_audio_from_pcm(self.audio_buffers[stream_id])
                    self.save_customer_audio_and_transcript(stream_id, self.audio_buffers[stream_id], transcript)


                    # Transcribe
                    transcript = self.transcribe_audio(wav_path)
                    # (Optional) store transcript in DB / log
                    logger.info(f"📄 Stored transcript for {stream_id}: {transcript}")

                    # Send static audio response
                    logger.info(f"🔇 Sustained silence ({silence_duration:.2f}s) detected. Sending bot reply.")
                    await self.send_static_audio_response(stream_id, static_path)

                    self.static_played_in_this_silence[stream_id] = True
                    self.bot_is_speaking[stream_id] = True
                    self.audio_buffers[stream_id] = b""
                    self.last_bot_reply_time[stream_id] = time.time()
            else:
                # Reset silence timer and flags when speech resumes
                if self.last_was_silence.get(stream_id, False):
                    logger.info(f"🔊 Speech resumed on {stream_id}, resetting flags.")
                self.silence_start_time[stream_id] = None
                self.static_played_in_this_silence[stream_id] = False
                self.bot_is_speaking[stream_id] = False

            self.last_was_silence[stream_id] = silence

        except Exception as e:
            logger.error(f"❌ Error handling Exotel media stream {stream_id}: {e}")

    async def handle_exotel_stop(self, stream_id: str, data: dict):
        """Handle Exotel stop event"""
        logger.info(f"🛑 SALES CALL ENDED: {stream_id}")
        # Clean up resources for this stream
        await self.cleanup_connections(stream_id)

    async def handle_exotel_mark(self, stream_id: str, data: dict):
        """Handle Exotel mark event - audio playback position marker"""
        mark_name = data.get("mark", {}).get("name", "unknown")
        logger.info(f"📍 EXOTEL MARK: {mark_name} for {stream_id}")

        # Mark events can be used to synchronize audio playback
        if mark_name == "greeting_complete":
            logger.info(f"✅ GREETING COMPLETED for {stream_id}")
        elif mark_name == "response_start":
            logger.info(f"🎯 RESPONSE PLAYBACK STARTED for {stream_id}")

    async def handle_exotel_clear(self, stream_id: str, data: dict):
        """Handle Exotel clear event - clear audio buffer and STOP bot speaking"""
        logger.info(f"🧹 EXOTEL CLEAR - INTERRUPTING BOT SPEECH: {stream_id}")

        # Clear local audio buffer and reset flags
        if stream_id in self.audio_buffers:
            self.audio_buffers[stream_id] = b""
            logger.info(f"🧹 CLEARED LOCAL AUDIO BUFFER for {stream_id}")

        self.bot_is_speaking[stream_id] = False
        self.static_played_in_this_silence[stream_id] = False
        self.last_was_silence[stream_id] = False

    def convert_pcm_to_ulaw(self, pcm_data: bytes) -> bytes:
        """Convert 16-bit PCM to G.711 u-law (same sample rate 8kHz) - simplified."""
        try:
            samples_pcm = struct.unpack(f'<{len(pcm_data)//2}h', pcm_data)
        except Exception as e:
            logger.error(f"❌ convert_pcm_to_ulaw: unpack error: {e}")
            return b""

        ulaw_bytes = []
        for sample in samples_pcm:
            sample = max(-8159, min(8159, sample))
            if sample < 0:
                sample = -sample
                sign = 0x80
            else:
                sign = 0x00

            if sample < 32:
                segment = 0
                quantized = sample >> 1
            elif sample < 96:
                segment = 1
                quantized = (sample - 32) >> 2
            elif sample < 224:
                segment = 2
                quantized = (sample - 96) >> 3
            elif sample < 480:
                segment = 3
                quantized = (sample - 224) >> 4
            elif sample < 992:
                segment = 4
                quantized = (sample - 480) >> 5
            elif sample < 2016:
                segment = 5
                quantized = (sample - 992) >> 6
            elif sample < 4064:
                segment = 6
                quantized = (sample - 2016) >> 7
            else:
                segment = 7
                quantized = (sample - 4064) >> 8

            ulaw_value = sign | (segment << 4) | (quantized & 0x0F)
            ulaw_bytes.append(ulaw_value ^ 0xFF)

        return bytes(ulaw_bytes)

    def convert_ulaw_to_pcm(self, ulaw_data: bytes) -> bytes:
        """Convert G.711 u-law to 16-bit PCM (simplified)."""
        pcm_samples = []

        for ulaw_byte in ulaw_data:
            ulaw_byte ^= 0xFF
            sign = ulaw_byte & 0x80
            segment = (ulaw_byte >> 4) & 0x07
            quantized = ulaw_byte & 0x0F

            if segment == 0:
                pcm_val = (quantized << 1) + 1
            elif segment == 1:
                pcm_val = ((quantized << 2) + 33)
            elif segment == 2:
                pcm_val = ((quantized << 3) + 97)
            elif segment == 3:
                pcm_val = ((quantized << 4) + 225)
            elif segment == 4:
                pcm_val = ((quantized << 5) + 481)
            elif segment == 5:
                pcm_val = ((quantized << 6) + 993)
            elif segment == 6:
                pcm_val = ((quantized << 7) + 2017)
            else:
                pcm_val = ((quantized << 8) + 4065)

            if sign:
                pcm_val = -pcm_val

            pcm_samples.append(pcm_val)

        try:
            return struct.pack(f'<{len(pcm_samples)}h', *pcm_samples)
        except Exception as e:
            logger.error(f"❌ convert_ulaw_to_pcm: pack error: {e}")
            return b""

    def apply_noise_suppression(self, audio_data: bytes) -> bytes:
        """Apply basic noise suppression and audio enhancement for telephony."""
        try:
            audio_samples = np.frombuffer(audio_data, dtype=np.int16)

            # Basic noise gate - suppress very quiet audio (likely noise)
            noise_threshold = Config.NOISE_THRESHOLD  # Configurable noise threshold
            audio_samples = np.where(np.abs(audio_samples) < noise_threshold, 0, audio_samples)

            # Simple high-pass filter to remove low-frequency noise (< 300Hz)
            if len(audio_samples) > 10:
                window_size = min(5, len(audio_samples) // 2)
                moving_avg = np.convolve(audio_samples.astype(np.float32),
                                         np.ones(window_size)/window_size, mode='same')
                audio_samples = audio_samples - (moving_avg.astype(np.int16) * 0.1)

            # Gentle compression to normalize levels
            max_val = np.max(np.abs(audio_samples))
            if max_val > 0:
                compression_ratio = 0.8
                normalized = audio_samples.astype(np.float32) / max_val
                compressed = np.sign(normalized) * (np.abs(normalized) ** compression_ratio)
                audio_samples = (compressed * max_val * 0.9).astype(np.int16)

            return audio_samples.tobytes()

        except ImportError:
            logger.warning("📢 NumPy not available - skipping noise suppression")
            return audio_data
        except Exception as e:
            logger.error(f"❌ Error in noise suppression: {e}")
            return audio_data
        
    # def save_customer_audio(self, stream_id, pcm_bytes):
    #     folder = os.path.join(os.getcwd(), "customer_audio")
    #     os.makedirs(folder, exist_ok=True)
    #     filename = f"{stream_id}_{int(time.time())}.wav"
    #     filepath = os.path.join(folder, filename)

    #     with wave.open(filepath, 'wb') as wf:
    #         wf.setnchannels(1)
    #         wf.setsampwidth(2)  # 16-bit
    #         wf.setframerate(self.sample_rate)
    #         wf.writeframes(pcm_bytes)

    #     logger.info(f"💾 Saved customer audio: {filepath}")
    #     return filepath

    def save_customer_audio_and_transcript(self, stream_id, pcm_bytes, transcript):
        folder = os.path.join(os.getcwd(), "customer_audio")
        os.makedirs(folder, exist_ok=True)
        base_filename = f"{stream_id}_{int(time.time())}"
        
        # Save audio
        wav_path = os.path.join(folder, f"{base_filename}.wav")
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        
        # Save transcript
        txt_path = os.path.join(folder, f"{base_filename}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript or "[No speech detected]")
        
        logger.info(f"💾 Saved customer audio: {wav_path}")
        logger.info(f"📝 Saved transcript: {txt_path}")
        
        return wav_path, txt_path

    
    def transcribe_audio(self, wav_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language=Config.LANGUAGE_CODE)
            logger.info(f"📝 Transcription: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("🤷 Could not understand audio")
            return ""
        except sr.RequestError as e:
            logger.error(f"Google Speech API error: {e}")
            return ""
    

    async def cleanup_connections(self, stream_id: str):
        """Clean up Exotel connections and local buffers (OpenAI removed)."""
        try:
            # Remove Exotel connection
            if stream_id in self.exotel_connections:
                try:
                    ws = self.exotel_connections[stream_id]["websocket"]
                    if not ws.closed:
                        await ws.close()
                except Exception:
                    pass
                del self.exotel_connections[stream_id]
                logger.info(f"🧹 EXOTEL CONNECTION REMOVED: {stream_id}")

            # Clean up audio buffer
            if stream_id in self.audio_buffers:
                del self.audio_buffers[stream_id]
                logger.info(f"🧹 AUDIO BUFFER CLEARED: {stream_id}")

            # Clear flags
            self.static_played_in_this_silence.pop(stream_id, None)
            self.bot_is_speaking.pop(stream_id, None)
            self.last_was_silence.pop(stream_id, None)

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def generate_test_tone(self, frequency=440, duration_ms=200, sample_rate=8000, amplitude=0.5):
        """Generate a short test tone WAV in-memory (PCM16 mono)."""
        try:
            t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
            tone = amplitude * np.sin(2 * np.pi * frequency * t)

            # Convert to 16-bit PCM format
            audio_data = np.int16(tone * 32767)

            # Write to an in-memory WAV file
            with io.BytesIO() as buffer:
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono audio
                    wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
                wav_bytes = buffer.getvalue()

            return wav_bytes
        except Exception as e:
            logger.error(f"❌ Error generating test tone: {e}")
            return b""

async def main():
    """Main function to start the Realtime Sales Bot (OpenAI removed)."""
    try:
        # Initialize the sales bot
        sales_bot = RealtimeSalesBot()

        # Start the WebSocket server
        logger.info(f"🚀 Starting Sales Bot Server on {Config.SERVER_HOST}:{Config.SERVER_PORT}")
        logger.info("📞 Ready for Exotel streaming connections!")
        logger.info("🔐 Using secure environment-based configuration")

        async with websockets.serve(
            sales_bot.handle_exotel_websocket,
            Config.SERVER_HOST,
            Config.SERVER_PORT
        ):
            logger.info(f"✅ Sales Bot Server running at ws://{Config.SERVER_HOST}:{Config.SERVER_PORT}")
            logger.info("🎯 Waiting for calls...")

            # Keep the server running
            await asyncio.Future()  # Run forever

    except KeyboardInterrupt:
        logger.info("⏹️ Server stopped by user")
    except ValueError as e:
        logger.error(f"❌ Configuration Error: {e}")
    except Exception as e:
        logger.error(f"❌ Server Error: {e}")


if __name__ == "__main__":
    # Optional: Run a one-off noise reduction of your static file (kept from original entrypoint)
    try:
        sample_path = os.path.join(os.getcwd(), "bot_fixed.wav")
        out_path = os.path.join(os.getcwd(), "output_clean.wav")
        if os.path.exists(sample_path):
            RealtimeSalesBot.reduce_noise_in_file(sample_path, out_path)
            logger.info("Noise reduction complete. Output saved as output_clean.wav")
        else:
            logger.info("No bot_fixed.wav found in current directory — skipping noise reduction step.")
    except Exception as e:
        logger.error(f"❌ Error in initial noise reduction check: {e}")

    asyncio.run(main())
