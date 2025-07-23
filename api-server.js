const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const http = require("http");
const WebSocket = require("ws");
const { spawn } = require("child_process");

const app = express();
const port = 3000;

// Setup multer for file uploads
const upload = multer({ dest: "uploads/" });

// Function to send audio buffer to Whisper CLI and get transcription
const wav = require("wav");

const { v4: uuidv4 } = require("uuid");

function transcribeWithWhisper(audioBuffer, callback) {
  // Generate unique temp filename for each transcription
  const tempAudioPath = path.join(
    __dirname,
    "audio",
    `temp_input_${uuidv4()}.wav`
  );

  const fileWriter = new wav.FileWriter(tempAudioPath, {
    channels: 1,
    sampleRate: 8000,
    bitDepth: 16,
  });

  fileWriter.write(audioBuffer);
  fileWriter.end();

  fileWriter.on("finish", () => {
    console.log(`WAV file written to ${tempAudioPath}`);
    // Spawn Whisper CLI process (assumes whisper is installed and in PATH)
    const whisperProcess = spawn("whisper", [
      tempAudioPath,
      "--model",
      "base",
      "--language",
      "en",
      "--fp16",
      "False",
    ]);

    let transcription = "";
    whisperProcess.stdout.on("data", (data) => {
      transcription += data.toString();
    });

    whisperProcess.stderr.on("data", (data) => {
      console.error(`Whisper error: ${data}`);
    });

    whisperProcess.on("close", (code) => {
      if (code === 0) {
        callback(null, transcription);
      } else {
        callback(new Error(`Whisper process exited with code ${code}`));
      }
      // Delete temp file here
      console.log(
        `⏳ Scheduling deletion of temp file in 5 minutes: ${tempAudioPath}`
      );
      setTimeout(() => {
        fs.unlink(tempAudioPath, (err) => {
          if (err) {
            console.warn(
              `❌ Failed to delete temp audio file after 5 minutes: ${tempAudioPath}`,
              err
            );
          } else {
            console.log(
              `🗑️ Temp file deleted after 5 minutes: ${tempAudioPath}`
            );
          }
        });
      }, 5 * 60 * 1000); // 5 minutes in milliseconds
    });
  });
}

// HTTP API endpoint to accept input audio file and return bot response audio file
app.post("/api/audio", upload.single("inputAudio"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No input audio file uploaded");
  }

  // Move uploaded file to audio/input.mp3
  const inputAudioPath = path.join(__dirname, "audio", "input.mp3");
  fs.renameSync(req.file.path, inputAudioPath);

  // Send back the bot-response audio file
  const botResponsePath = path.join(__dirname, "audio", "bot1.wav");

  if (!fs.existsSync(botResponsePath)) {
    return res.status(500).send("Bot response audio file not found");
  }

  res.setHeader("Content-Type", "audio/mpeg");
  res.setHeader("Content-Disposition", "attachment; filename=bot1.wav");

  const readStream = fs.createReadStream(botResponsePath);
  readStream.pipe(res);
});

// Create HTTP server and attach Express app
const server = http.createServer(app);

// Setup WebSocket server on the same HTTP server
const wss = new WebSocket.Server({ server, path: "/media" });

console.log(`API server listening on http://localhost:${port}`);

wss.on("connection", (ws) => {
  console.log("✅ WebSocket client connected");

  let audioChunks = [];
  let partialAudioBuffer = Buffer.alloc(0);
  let sequenceNumber = 0;

  ws.on("message", (message) => {
    try {
      console.log("Received message:", message.toString());
      const msg = JSON.parse(message.toString());

      if (msg.event === "start") {
        console.log(`📞 Stream started: ${msg.start.stream_sid}`);
        audioChunks = [];
        partialAudioBuffer = Buffer.alloc(0);
        sequenceNumber = 0;
      } else if (msg.event === "media") {
        console.log(
          `Received media chunk: sequence_number=${msg.sequence_number}, stream_sid=${msg.stream_sid}`
        );
        const payload = Buffer.from(msg.media.payload, "base64");
        audioChunks.push(payload);

        // Append to partial buffer
        partialAudioBuffer = Buffer.concat([partialAudioBuffer, payload]);

        // If partial buffer exceeds threshold (e.g., 1 second of audio ~16000 bytes for 16kHz 16bit mono)
        if (partialAudioBuffer.length >= 16000) {
          console.log(
            "Partial buffer threshold reached, transcribing partial audio..."
          );
          // Transcribe partial buffer
          transcribeWithWhisper(partialAudioBuffer, (err, transcription) => {
            if (err) {
              console.error("Error transcribing partial audio:", err);
              ws.send(JSON.stringify({ event: "error", message: err.message }));
            } else {
              console.log("Partial Transcription:", transcription);
              ws.send(
                JSON.stringify({
                  event: "transcription",
                  transcript: transcription,
                  sequence_number: sequenceNumber++,
                  partial: true,
                })
              );
              // Send bot response audio after partial transcription
              sendBotAudio(ws);
            }
            saveTranscriptionToFile(transcription, true);
          });
          // Reset partial buffer
          partialAudioBuffer = Buffer.alloc(0);
        }
      } else if (msg.event === "stop") {
        console.log("🛑 Stream stopped.");

        // Concatenate all audio chunks
        const fullAudioBuffer = Buffer.concat(audioChunks);

        console.log("Transcribing full audio buffer...");
        // Transcribe full audio buffer
        transcribeWithWhisper(fullAudioBuffer, (err, transcription) => {
          if (err) {
            console.error("Error transcribing audio:", err);
            ws.send(JSON.stringify({ event: "error", message: err.message }));
          } else {
            console.log("Final Transcription:", transcription);
            ws.send(
              JSON.stringify({
                event: "transcription",
                transcript: transcription,
                sequence_number: sequenceNumber++,
                partial: false,
              })
            );
          }
          saveTranscriptionToFile(transcription, false);
        });

        // Send bot response audio after a delay
        setTimeout(() => {
          console.log("Sending bot response audio...");
          sendBotAudio(ws);
        }, 1000);
      } else if (msg.event === "dtmf") {
        console.log(`DTMF received: ${msg.dtmf.digits}`);
      } else if (msg.event === "clear") {
        console.log("Clear event received");
        ws.close();
      }
    } catch (err) {
      console.error("Error processing message:", err);
    }
  });

  ws.on("close", () => {
    console.log("❌ WebSocket client disconnected");
  });
});

// Function to send bot response audio over WebSocket
function sendBotAudio(ws) {
  // Use symbol on socket to track audio send status
  if (ws.isBotAudioSending) {
    console.log("⚠️ Bot audio already sending for this socket, skipping...");
    return;
  }

  ws.isBotAudioSending = true;

  const audioPath = path.join(__dirname, "audio", "bot1.wav");
  const readStream = fs.createReadStream(audioPath, { highWaterMark: 1024 });

  let sequence = 0;

  const interval = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) {
      clearInterval(interval);
      ws.isBotAudioSending = false;
      console.log("WebSocket closed, stopped sending bot response audio.");
      return;
    }

    const chunk = readStream.read();
    if (chunk) {
      const payload = chunk.toString("base64");
      ws.send(
        JSON.stringify({
          event: "media",
          stream_sid: "bot-response",
          media: { payload },
          sequence_number: sequence++,
        })
      );
    } else {
      clearInterval(interval);
      ws.isBotAudioSending = false;
      console.log("✅ Sent full bot response audio.");
    }
  }, 40);
}

function saveTranscriptionToFile(transcript, isPartial = false) {
  const folder = path.join(__dirname, "transcripts");
  if (!fs.existsSync(folder)) {
    fs.mkdirSync(folder);
  }

  const fileName = `${isPartial ? "partial" : "final"}_${Date.now()}.txt`;
  const filePath = path.join(folder, fileName);

  fs.writeFile(filePath, transcript, (err) => {
    if (err) {
      console.error(`❌ Failed to save transcription: ${filePath}`, err);
    } else {
      console.log(`📝 Transcription saved to: ${filePath}`);
    }
  });
}

server.listen(port);
