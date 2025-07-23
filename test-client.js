const WebSocket = require("ws");
const fs = require("fs");
const path = require("path");

const ws = new WebSocket("ws://localhost:5001/media");

ws.on("open", () => {
  console.log("🧪 Connected to local bot");

  // 1️⃣ Send "start" event
  ws.send(
    JSON.stringify({
      event: "start",
      start: {
        stream_sid: "test-stream",
      },
    })
  );

  // 2️⃣ Stream WAV audio as base64 chunks
  const filePath = path.join(__dirname, "audio", "input.mp3");
  const stream = fs.createReadStream(filePath, { highWaterMark: 1024 });

  stream.on("data", (chunk) => {
    const payload = chunk.toString("base64");
    ws.send(
      JSON.stringify({
        event: "media",
        stream_sid: "test-stream",
        media: {
          payload: payload,
        },
      })
    );
  });

  stream.on("end", () => {
    // 3️⃣ Send "stop" event after audio is sent
    ws.send(
      JSON.stringify({
        event: "stop",
        stream_sid: "test-stream",
      })
    );
    console.log("✅ Test input sent. Waiting for bot reply...");
  });
});

const outputPath = path.join(__dirname, "audio", "response-from-bot.mp3");

// Remove previous response file if exists
if (fs.existsSync(outputPath)) {
  fs.unlinkSync(outputPath);
}

ws.on("message", (msg) => {
  try {
    const message = JSON.parse(msg.toString());
    if (message.event === "media" && message.media && message.media.payload) {
      const audioBuffer = Buffer.from(message.media.payload, "base64");
      fs.appendFileSync(outputPath, audioBuffer);
      console.log("📥 Received and saved audio chunk to response-from-bot.wav");
    } else {
      console.log("📥 Bot sent message:", msg.toString());
    }
  } catch (err) {
    console.error("Error parsing message:", err);
  }
});
