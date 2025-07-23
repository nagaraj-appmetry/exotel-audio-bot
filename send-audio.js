const fs = require("fs");
const path = require("path");

function sendBotAudio(ws) {
  const audioPath = path.join(__dirname, "audio", "bot-response.mp3");

  const readStream = fs.createReadStream(audioPath, {
    highWaterMark: 1024, // chunk size for mp3 streaming
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
  }, 40); // adjusted interval for mp3 streaming
}

module.exports = { sendBotAudio };
