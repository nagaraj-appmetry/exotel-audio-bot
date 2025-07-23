const express = require("express");
const multer = require("multer");
const path = require("path");
const fs = require("fs");

const app = express();
const port = 3000;

// Setup multer for file uploads
const upload = multer({ dest: "uploads/" });

// API endpoint to accept input audio file and return bot response audio file
app.post("/api/audio", upload.single("inputAudio"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No input audio file uploaded");
  }

  // Move uploaded file to audio/input.mp3 (or input.wav if needed)
  const inputAudioPath = path.join(__dirname, "audio", "input.mp3");
  fs.renameSync(req.file.path, inputAudioPath);

  // Simulate processing delay or call to bot logic here
  // For now, just send back the bot-response.mp3 file

  const botResponsePath = path.join(__dirname, "audio", "bot-response.mp3");

  // Check if bot response file exists
  if (!fs.existsSync(botResponsePath)) {
    return res.status(500).send("Bot response audio file not found");
  }

  res.setHeader("Content-Type", "audio/mpeg");
  res.setHeader("Content-Disposition", "attachment; filename=bot-response.mp3");

  const readStream = fs.createReadStream(botResponsePath);
  readStream.pipe(res);
});

app.listen(port, () => {
  console.log(`API server listening at http://localhost:${port}`);
});
