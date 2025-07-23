const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const http = require("http");
const WebSocket = require("ws");

const app = express();
const port = 3000;

// Setup multer for file uploads
const upload = multer({ dest: "uploads/" });

// HTTP API endpoint to accept input audio file and return bot response audio file
app.post("/api/audio", upload.single("inputAudio"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No input audio file uploaded");
  }

  // Move uploaded file to audio/input.mp3
  const inputAudioPath = path.join(__dirname, "audio", "input.mp3");
  fs.renameSync(req.file.path, inputAudioPath);

  // Send back the bot-response.mp3 file
  const botResponsePath = path.join(__dirname, "audio", "bot-response.mp3");

  if (!fs.existsSync(botResponsePath)) {
    return res.status(500).send("Bot response audio file not found");
  }

  res.setHeader("Content-Type", "audio/mpeg");
  res.setHeader("Content-Disposition", "attachment; filename=bot-response.mp3");

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

  const fileStream = fs.createWriteStream(
    path.join(__dirname, "audio", "input.mp3"),
    { flags: "w" }
  );

  ws.on("message", (message) => {
    try {
      const msg = JSON.parse(message.toString());

      if (msg.event === "start") {
        console.log(`📞 Stream started: ${msg.start.stream_sid}`);
      } else if (msg.event === "media") {
        const payload = Buffer.from(msg.media.payload, "base64");
        fileStream.write(payload);
      } else if (msg.event === "stop") {
        console.log("🛑 Stream stopped. Closing file.");
        fileStream.end();

        // After receiving full audio, send response audio
        setTimeout(() => {
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
    fileStream.end();
  });
});

// Function to send bot response audio over WebSocket
function sendBotAudio(ws) {
  const audioPath = path.join(__dirname, "audio", "bot-response.mp3");

  const readStream = fs.createReadStream(audioPath, { highWaterMark: 1024 });

  let sequence = 0;

  const interval = setInterval(() => {
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
      console.log("✅ Sent full bot response audio.");
    }
  }, 40);
}

server.listen(port);
