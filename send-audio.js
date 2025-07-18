const fs = require("fs");
const path = require("path");

function sendBotAudio(ws) {
  const audioPath = path.join(__dirname, "audio", "bot-response.wav");

  const readStream = fs.createReadStream(audioPath, {
    highWaterMark: 320, // approx. 20ms of audio at 16kHz mono 16-bit
  });

  let sequence = 0;

  const interval = setInterval(() => {
    const chunk = readStream.read();
    if (chunk) {
      const payload = chunk.toString("base64");
      ws.send(
        JSON.stringify({
          event: "media",
          stream_sid: "bot-response",
          media: {
            payload: payload,
          },
          sequence_number: sequence++,
        })
      );
    } else {
      clearInterval(interval);
      console.log("✅ Sent full bot response audio.");
    }
  }, 20); // emulate 20ms packet sending
}

module.exports = { sendBotAudio };
