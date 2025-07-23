const WebSocket = require("ws");
const fs = require("fs");
const path = require("path");
const { Writable } = require("stream");
const { sendBotAudio } = require("./send-audio");

const wss = new WebSocket.Server({ port: 5001, path: "/media" });
console.log("🚀 WebSocket Voicebot listening on ws://localhost:5001/media");

wss.on("connection", function connection(ws) {
  console.log("✅ Voicebot connected!");

const fileStream = fs.createWriteStream(
  path.join(__dirname, "audio", "response", "input.mp3"),
  {
    flags: "w",
  }
);

  ws.on("message", function incoming(message) {
    const msg = JSON.parse(message.toString());

    if (msg.event === "start") {
      console.log(`📞 Stream started: ${msg.start.stream_sid}`);
    }

    if (msg.event === "media") {
      const payload = Buffer.from(msg.media.payload, "base64");
      fileStream.write(payload);
    }

    if (msg.event === "stop") {
      console.log("🛑 Stream stopped. Closing file.");
      fileStream.end();

      // After receiving full audio, send response
      setTimeout(() => {
        sendBotAudio(ws);
      }, 1000); // small delay to ensure file saved
    }
  });

  ws.on("close", () => {
    console.log("❌ Voicebot disconnected");
    fileStream.end();
  });
});
