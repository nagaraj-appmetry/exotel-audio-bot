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
import aiohttp
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO
import math


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

        # NEW: per-stream locks to avoid races
        self.stream_locks: Dict[str, asyncio.Lock] = {}

        # Silence detection state variables
        self.static_played_in_this_silence: Dict[str, bool] = {}  # stream_id -> bool flag for played static audio
        self.bot_is_speaking: Dict[str, bool] = {}
        self.last_was_silence: Dict[str, bool] = {}               # stream_id -> bool flag to track last chunk silence


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
            


    async def handle_exotel_start(self, stream_id: str, data: dict):
        """Handle Exotel start event"""
        logger.info(f"🚀 SALES CALL STARTED: {stream_id}")

    # Silence detection / thresholds (kept from original)
    MIN_SPEECH_LENGTH = 64000   # ~4 sec of speech at 8kHz
    SILENCE_THRESHOLD = 300
    SILENCE_MIN_DURATION = 0.8  # seconds before bot replies
    COOLDOWN_SECONDS = 2.0      # seconds before bot can reply again

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

            # Initialize stream-specific buffers, flags and lock
            if stream_id not in self.audio_buffers:
                self.audio_buffers[stream_id] = b""
                self.static_played_in_this_silence[stream_id] = False
                self.last_was_silence[stream_id] = False
                self.bot_is_speaking[stream_id] = False
                self.silence_start_time[stream_id] = None
                self.last_bot_reply_time[stream_id] = 0
                self.stream_locks[stream_id] = asyncio.Lock()

            try:
                processed = self.apply_noise_suppression(chunk)
            except Exception as ns_err:
                logger.warning(f"⚠️ Noise suppression failed: {ns_err}")
                processed = chunk

            self.audio_buffers[stream_id] += processed
            silence = self.is_silence(processed)

            static_path = os.path.join(os.path.dirname(__file__), "engines", "bot_fixed.wav")
            if not os.path.exists(static_path):
                logger.warning(f"⚠️ Static audio file not found at {static_path}")

            if silence:
                if self.silence_start_time[stream_id] is None:
                    self.silence_start_time[stream_id] = time.time()

                silence_duration = time.time() - self.silence_start_time[stream_id]

                if (silence_duration >= self.SILENCE_MIN_DURATION and
                    len(self.audio_buffers[stream_id]) > self.MIN_SPEECH_LENGTH and
                    not self.static_played_in_this_silence[stream_id] and
                    not self.bot_is_speaking[stream_id] and
                    (time.time() - self.last_bot_reply_time[stream_id] > self.COOLDOWN_SECONDS)):

                    # Enter critical section for this stream
                    async with self.stream_locks[stream_id]:
                        # mark bot as speaking to avoid re-entry (keeps others waiting)
                        self.bot_is_speaking[stream_id] = True

                        # 1) Transcribe off the event loop
                        transcript = await asyncio.to_thread(self.transcribe_audio_from_pcm, self.audio_buffers[stream_id])

                        # 2) Save audio + transcript (sync, lightweight)
                        wav_path, txt_path = self.save_customer_audio_and_transcript(
                            stream_id, self.audio_buffers[stream_id], transcript
                        )
                        logger.info(f"📄 Stored transcript for {stream_id}: {transcript} (saved at {txt_path})")

                        # 3) Get bot reply (async, uses aiohttp)
                        bot_reply_text = await self.get_bot_reply_text(transcript)

                        if not bot_reply_text:
                            logger.warning("⚠️ Local bot reply empty — falling back to static audio")
                            # await self.send_static_audio_response(stream_id, static_path)
                        else:
                            # 4) Produce TTS off the event loop (gTTS + pydub are blocking)
                            pcm_bytes = await asyncio.to_thread(self.tts_text_to_pcm_bytes, bot_reply_text, self.sample_rate)

                            if not pcm_bytes:
                                logger.error("❌ TTS produced no audio — falling back to static file")
                                # await self.send_static_audio_response(stream_id, static_path)
                            else:
                                # 5) Wrap PCM into a WAV container and send it (single media message)
                                with io.BytesIO() as buf:
                                    with wave.open(buf, 'wb') as wf:
                                        wf.setnchannels(1)
                                        wf.setsampwidth(2)
                                        wf.setframerate(self.sample_rate)
                                        wf.writeframes(pcm_bytes)
                                    wav_bytes = buf.getvalue()

                                # Send dynamic WAV to Exotel (same format as your static file)
                                await self.send_audio_bytes_to_exotel(stream_id, wav_bytes)

                        # update flags & clear local buffer (still in lock)
                        self.static_played_in_this_silence[stream_id] = True
                        self.audio_buffers[stream_id] = b""
                        self.last_bot_reply_time[stream_id] = time.time()

                        # Leave bot_is_speaking True until Exotel signals end of playback (mark/clear)
            else:
                if self.last_was_silence.get(stream_id, False):
                    logger.info(f"🔊 Speech resumed on {stream_id}, resetting flags.")
                self.silence_start_time[stream_id] = None
                self.static_played_in_this_silence[stream_id] = False
                self.bot_is_speaking[stream_id] = False

            self.last_was_silence[stream_id] = silence

        except Exception as e:
            import traceback
            logger.error(f"❌ Error handling Exotel media stream {stream_id}: {e}\n{traceback.format_exc()}")
    

    # ---- Helper: send raw audio bytes (WAV) to Exotel (reuse for static + dynamic) ----
    async def send_audio_bytes_to_exotel(self, stream_id: str, audio_bytes: bytes):
        if stream_id not in self.exotel_connections:
            logger.warning(f"⚠️ No Exotel connection for {stream_id} (cannot send audio)")
            return
        try:
            exotel_ws = self.exotel_connections[stream_id]["websocket"]
            audio_b64 = base64.b64encode(audio_bytes).decode()
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
            logger.info(f"🔊 Sent dynamic audio ({len(audio_bytes)} bytes) to user for {stream_id}")
        except Exception as e:
            logger.error(f"❌ Error sending audio bytes to Exotel for {stream_id}: {e}")


   
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
        
    # ---- Save customer audio and transcript to disk (for debugging) ----
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
    


    def transcribe_audio_from_pcm(self, pcm_bytes: bytes) -> str:
        """
        Write PCM16 bytes to a temporary WAV (mono, 16-bit, sample_rate self.sample_rate)
        then call transcribe_audio(wav_path). Returns the transcription string.
        """
        try:
            tmp_folder = os.path.join(os.getcwd(), "tmp_audio")
            os.makedirs(tmp_folder, exist_ok=True)
            tmp_path = os.path.join(tmp_folder, f"pcm_{int(time.time()*1000)}.wav")
            # write pcm bytes to WAV
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_bytes)
            # Use existing transcribe_audio to get text
            text = self.transcribe_audio(tmp_path)
            # Optionally remove the tmp file after transcribing
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return text or ""
        except Exception as e:
            logger.error(f"❌ transcribe_audio_from_pcm failed: {e}")
            return ""



    async def get_bot_reply_text(self, transcript: str) -> str:
        """
        Send customer transcription to local bot and get text response.
        Uses your exact curl endpoint: http://localhost:5000/get-reply
        """
        # Updated URL to match your curl endpoint
        url = "http://127.0.0.1:5000/get-reply"
        
        if not transcript:
            logger.warning("⚠️ Empty transcript provided to bot")
            return ""


        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Send customer transcription in the exact format your curl uses
                payload = {"text": transcript}
                
                async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                    if resp.status != 200:
                        logger.error(f"❌ Local bot returned status {resp.status}")
                        return ""
                        
                    data = await resp.json()
                    # Expecting response format from your bot server
                    reply_text = data.get("reply", "") or data.get("response", "") or data.get("text", "")
                    
                    logger.info(f"🔁 Customer said: {transcript[:100]}")
                    logger.info(f"🔁 Bot replied: {reply_text[:200]}")
                    return reply_text
                    
        except Exception as e:
            logger.error(f"❌ Error contacting local bot server: {e}")
            return ""


   # ---- Cleanup: remove stream lock on teardown ----
    async def cleanup_connections(self, stream_id: str):
        try:
            if stream_id in self.exotel_connections:
                try:
                    ws = self.exotel_connections[stream_id]["websocket"]
                    if not ws.closed:
                        await ws.close()
                except Exception:
                    pass
                del self.exotel_connections[stream_id]
                logger.info(f"🧹 EXOTEL CONNECTION REMOVED: {stream_id}")


            if stream_id in self.audio_buffers:
                del self.audio_buffers[stream_id]
                logger.info(f"🧹 AUDIO BUFFER CLEARED: {stream_id}")


            # Clear flags and lock
            self.static_played_in_this_silence.pop(stream_id, None)
            self.bot_is_speaking.pop(stream_id, None)
            self.last_was_silence.pop(stream_id, None)
            self.silence_start_time.pop(stream_id, None)
            self.last_bot_reply_time.pop(stream_id, None)


            # NEW: remove stream lock if present
            lock = self.stream_locks.pop(stream_id, None)
            if lock:
                # nothing special to do with the lock, just remove reference
                pass


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
        
    def tts_text_to_pcm_bytes(self, text: str, sample_rate: int = 8000) -> bytes:
        """
        Convert text -> PCM16 bytes (mono) at requested sample_rate using gTTS + pydub.
        Returns raw PCM16 bytes (no WAV header). We will wrap into WAV frames or send raw data
        depending on Exotel expectation. This function returns PCM raw bytes.
        NOTE: Requires ffmpeg for pydub to load/convert.
        """
        if not text:
            return b""


        try:
            # Generate MP3 in-memory with gTTS
            tts = gTTS(text=text, lang=getattr(Config, "LANGUAGE_CODE", "en"))
            mp3_buf = BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)


            # Load into pydub, convert to desired format
            audio = AudioSegment.from_file(mp3_buf, format="mp3")
            audio = audio.set_frame_rate(sample_rate).set_channels(1).set_sample_width(2)  # 16-bit PCM


            # raw_data is PCM16 little-endian bytes
            pcm_bytes = audio.raw_data
            return pcm_bytes
        except Exception as e:
            logger.error(f"❌ TTS conversion failed: {e}")
            return b""



    async def stream_pcm_to_exotel(self, stream_id: str, pcm_bytes: bytes, chunk_ms: int = None):
        """
        Stream PCM16 bytes back to Exotel by splitting into chunk_ms ms pieces and sending 'media' events.
        - pcm_bytes: raw PCM16 bytes (mono, 16-bit, sample_rate matching self.sample_rate)
        - chunk_ms: length of each chunk in milliseconds (defaults to Config.TTS_CHUNK_MS or 200ms)
        """
        if stream_id not in self.exotel_connections:
            logger.warning(f"⚠️ No Exotel connection for {stream_id} (cannot stream audio)")
            return


        exotel_ws = self.exotel_connections[stream_id]["websocket"]
        chunk_ms = chunk_ms or getattr(Config, "TTS_CHUNK_MS", 200)


        bytes_per_sample = 2  # 16-bit
        samples_per_ms = self.sample_rate / 1000.0
        samples_per_chunk = int(samples_per_ms * chunk_ms)
        bytes_per_chunk = samples_per_chunk * bytes_per_sample
        if bytes_per_chunk <= 0:
            bytes_per_chunk = 1600  # fallback


        total_len = len(pcm_bytes)
        total_chunks = math.ceil(total_len / bytes_per_chunk)
        seq = int(time.time())  # sequence base; incremental sequenceNumber preferred


        logger.info(f"📤 Streaming TTS audio to {stream_id}: {total_len} bytes in {total_chunks} chunks ({chunk_ms}ms each)")


        try:
            offset = 0
            chunk_index = 0
            while offset < total_len:
                out_chunk = pcm_bytes[offset: offset + bytes_per_chunk]
                offset += bytes_per_chunk
                chunk_index += 1

                payload_bytes = out_chunk

                audio_b64 = base64.b64encode(payload_bytes).decode()

                media_message = {
                    "event": "media",
                    "streamSid": stream_id,
                    "media": {
                        "payload": audio_b64,
                        "timestamp": str(int(time.time() * 1000)),
                        "sequenceNumber": str(seq + chunk_index)
                    }
                }

                await exotel_ws.send(json.dumps(media_message))

                # small sleep to pace chunks (equal to chunk length)
                await asyncio.sleep(chunk_ms / 1000.0)

            logger.info(f"✅ Finished streaming TTS audio to {stream_id}")
        except Exception as e:
            logger.error(f"❌ Error streaming TTS to Exotel: {e}")    


async def main():
    """Main function to start the Realtime Sales Bot."""
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
        sample_path = os.path.join(os.getcwd(), "engines", "bot_fixed.wav")
        out_path = os.path.join(os.getcwd(), "output_clean.wav")
        if os.path.exists(sample_path):
            RealtimeSalesBot.reduce_noise_in_file(sample_path, out_path)
            logger.info("Noise reduction complete. Output saved as output_clean.wav")
        else:
            logger.info("No bot_fixed.wav found in current directory — skipping noise reduction step.")
    except Exception as e:
        logger.error(f"❌ Error in initial noise reduction check: {e}")


    asyncio.run(main())