import type { ApiExampleKind } from "./types";

export type ApiExampleContext = {
  base: string;
  root: string;
  key: string;
  speakerVoice: string;
};
export function buildApiExample(kind: ApiExampleKind, context: ApiExampleContext): string {
  const { base, root, key, speakerVoice } = context;
  if (kind === "python") {
    return `import json
import requests

API_KEY = "${key}"
BASE_URL = "${base}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Health check, no auth required.
print(requests.get("${root}/health", timeout=10).json())

# Status and installed model folders.
print(requests.get(f"{BASE_URL}/status", headers=HEADERS, timeout=10).json())
print(requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=10).json())
print(requests.get(f"{BASE_URL}/higgs/speakers", headers=HEADERS, timeout=10).json())

# Plain TTS. Finished-file routes support "wav" or "mp3".
speech = requests.post(
    f"{BASE_URL}/audio/speech",
    headers=HEADERS,
    json={
        "model": "current",
        "input": "Hello from the Higgs Studio API.",
        "voice": "default",
        "response_format": "mp3",
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "seed": 1234,
    },
    timeout=600,
)
speech.raise_for_status()
open("speech.mp3", "wb").write(speech.content)

# Saved speaker identity clone through the OpenAI-style route.
speaker = requests.post(
    f"{BASE_URL}/audio/speech",
    headers=HEADERS,
    json={
        "model": "current",
        "input": "This uses a saved speaker identity.",
        "voice": "${speakerVoice}",
        "response_format": "wav",
        "max_tokens": 1024,
    },
    timeout=600,
)
speaker.raise_for_status()
open("speaker.wav", "wb").write(speaker.content)

# Voice clone.
clone = requests.post(
    f"{BASE_URL}/higgs/voice-clone",
    headers=HEADERS,
    json={
        "input": "This uses the reference speaker.",
        "reference_audio_path": r"C:\\voices\\speaker.wav",
        "reference_text": "Optional transcript of the reference audio.",
        "response_format": "wav",
        "max_tokens": 1024,
    },
    timeout=600,
)
clone.raise_for_status()
open("clone.wav", "wb").write(clone.content)

# Continue speech.
continued = requests.post(
    f"{BASE_URL}/higgs/continue-speech",
    headers=HEADERS,
    json={
        "audio_path": r"C:\\voices\\start.wav",
        "continuation_text": "and this is the continuation.",
        "response_format": "wav",
        "max_tokens": 1024,
    },
    timeout=600,
)
continued.raise_for_status()
open("continued.wav", "wb").write(continued.content)

# Streaming TTS. Each response line is one JSON event. Audio events carry wavBase64 chunks;
# decode/play chunks as they arrive. The final event can be wavBase64 or mp3Base64.
with requests.post(
    f"{BASE_URL}/higgs/audio/stream",
    headers=HEADERS,
    json={
        "input": "This starts returning audio chunks before the final file is ready.",
        "voice": "default",
        "response_format": "mp3",
        "max_tokens": 1024,
    },
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    for line in stream.iter_lines(decode_unicode=True):
        if not line:
            continue
        event = json.loads(line)
        print(event["event"], event.get("phase", event.get("sampleCount", "")))

# Cancel the active job if needed.
# requests.post(f"{BASE_URL}/higgs/cancel", headers=HEADERS, timeout=10)`;
  }
  if (kind === "javascript") {
    return `// Node 18+ example.
import { writeFile } from "node:fs/promises";

const API_KEY = "${key}";
const BASE_URL = "${base}";
const headers = {
  "Authorization": \`Bearer \${API_KEY}\`,
  "Content-Type": "application/json",
};

const health = await fetch("${root}/health");
console.log(await health.json());

const status = await fetch(\`\${BASE_URL}/status\`, { headers });
console.log(await status.json());

const models = await fetch(\`\${BASE_URL}/models\`, { headers });
console.log(await models.json());

const speakers = await fetch(\`\${BASE_URL}/higgs/speakers\`, { headers });
console.log(await speakers.json());

const speech = await fetch(\`\${BASE_URL}/audio/speech\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    model: "current",
    input: "Hello from the Higgs Studio API.",
    voice: "default", // Or "${speakerVoice}" for a saved speaker identity.
    response_format: "mp3",
    max_tokens: 1024,
    temperature: 1.0,
    top_p: 0.95,
    top_k: 50,
    seed: 1234,
  }),
});
if (!speech.ok) throw new Error(await speech.text());
await writeFile("speech.mp3", Buffer.from(await speech.arrayBuffer()));

const savedSpeaker = await fetch(\`\${BASE_URL}/audio/speech\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    model: "current",
    input: "This uses a saved speaker identity.",
    voice: "${speakerVoice}",
    response_format: "wav",
    max_tokens: 1024,
  }),
});
if (!savedSpeaker.ok) throw new Error(await savedSpeaker.text());
await writeFile("speaker.wav", Buffer.from(await savedSpeaker.arrayBuffer()));

const clone = await fetch(\`\${BASE_URL}/higgs/voice-clone\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    input: "This uses the reference speaker.",
    reference_audio_path: "C:\\\\voices\\\\speaker.wav",
    reference_text: "Optional transcript of the reference audio.",
    response_format: "wav",
    max_tokens: 1024,
  }),
});
if (!clone.ok) throw new Error(await clone.text());
await writeFile("clone.wav", Buffer.from(await clone.arrayBuffer()));

const continued = await fetch(\`\${BASE_URL}/higgs/continue-speech\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    audio_path: "C:\\\\voices\\\\start.wav",
    continuation_text: "and this is the continuation.",
    response_format: "wav",
    max_tokens: 1024,
  }),
});
if (!continued.ok) throw new Error(await continued.text());
await writeFile("continued.wav", Buffer.from(await continued.arrayBuffer()));

const streamed = await fetch(\`\${BASE_URL}/higgs/audio/stream\`, {
  method: "POST",
  headers,
  body: JSON.stringify({
    input: "This streams progress and wav-base64 chunks.",
    voice: "default",
    response_format: "mp3",
    max_tokens: 1024,
  }),
});
if (!streamed.ok || !streamed.body) throw new Error(await streamed.text());
const reader = streamed.body.getReader();
const decoder = new TextDecoder();
let pending = "";
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  pending += decoder.decode(value, { stream: true });
  const lines = pending.split("\\n");
  pending = lines.pop() || "";
  for (const line of lines) {
    if (!line.trim()) continue;
    const event = JSON.parse(line);
    // event.event === "audio" has wavBase64 for live playback.
    // event.event === "final" has wavBase64 or mp3Base64 for the finished clean file.
    console.log(event.event, event.phase || event.sampleCount || "");
  }
}

// await fetch(\`\${BASE_URL}/higgs/cancel\`, { method: "POST", headers });`;
  }
  if (kind === "powershell") {
    return `$ApiKey = "${key}"
$BaseUrl = "${base}"
$Headers = @{
  Authorization = "Bearer $ApiKey"
  "Content-Type" = "application/json"
}

# Health check, no auth required.
Invoke-RestMethod -Uri "${root}/health" -Method Get

# Status and installed model folders.
Invoke-RestMethod -Uri "$BaseUrl/status" -Method Get -Headers $Headers
Invoke-RestMethod -Uri "$BaseUrl/models" -Method Get -Headers $Headers
Invoke-RestMethod -Uri "$BaseUrl/higgs/speakers" -Method Get -Headers $Headers

# Plain TTS. Finished-file routes support "wav" or "mp3".
$SpeechBody = @{
  model = "current"
  input = "Hello from the Higgs Studio API."
  voice = "default"
  response_format = "mp3"
  max_tokens = 1024
  temperature = 1.0
  top_p = 0.95
  top_k = 50
  seed = 1234
} | ConvertTo-Json
Invoke-WebRequest -Uri "$BaseUrl/audio/speech" -Method Post -Headers $Headers -Body $SpeechBody -OutFile "speech.mp3"

# Saved speaker identity clone through the OpenAI-style route.
$SpeakerBody = @{
  model = "current"
  input = "This uses a saved speaker identity."
  voice = "${speakerVoice}"
  response_format = "wav"
  max_tokens = 1024
} | ConvertTo-Json
Invoke-WebRequest -Uri "$BaseUrl/audio/speech" -Method Post -Headers $Headers -Body $SpeakerBody -OutFile "speaker.wav"

# Voice clone.
$CloneBody = @{
  input = "This uses the reference speaker."
  reference_audio_path = "C:\\voices\\speaker.wav"
  reference_text = "Optional transcript of the reference audio."
  response_format = "wav"
  max_tokens = 1024
} | ConvertTo-Json
Invoke-WebRequest -Uri "$BaseUrl/higgs/voice-clone" -Method Post -Headers $Headers -Body $CloneBody -OutFile "clone.wav"

# Continue speech.
$ContinueBody = @{
  audio_path = "C:\\voices\\start.wav"
  continuation_text = "and this is the continuation."
  response_format = "wav"
  max_tokens = 1024
} | ConvertTo-Json
Invoke-WebRequest -Uri "$BaseUrl/higgs/continue-speech" -Method Post -Headers $Headers -Body $ContinueBody -OutFile "continued.wav"

# Streaming TTS. Each line is a JSON event. Audio events include wavBase64 chunks for playback;
# final includes wavBase64 or mp3Base64 for the clean finished file.
$StreamBody = @{
  input = "This streams progress and wav-base64 chunks."
  voice = "default"
  response_format = "mp3"
  max_tokens = 1024
} | ConvertTo-Json
curl.exe -N -X POST "$BaseUrl/higgs/audio/stream" \`
  -H "Authorization: Bearer $ApiKey" \`
  -H "Content-Type: application/json" \`
  -d $StreamBody

# Cancel the active job if needed.
# Invoke-RestMethod -Uri "$BaseUrl/higgs/cancel" -Method Post -Headers $Headers`;
  }
  return `# Health check, no auth required.
curl "${root}/health"

# Status and installed model folders.
curl "${base}/status" \\
  -H "Authorization: Bearer ${key}"

curl "${base}/models" \\
  -H "Authorization: Bearer ${key}"

curl "${base}/higgs/speakers" \\
  -H "Authorization: Bearer ${key}"

# Plain TTS. Finished-file routes support "wav" or "mp3".
curl -X POST "${base}/audio/speech" \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"current","input":"Hello from the Higgs Studio API.","voice":"default","response_format":"mp3","max_tokens":1024,"temperature":1.0,"top_p":0.95,"top_k":50,"seed":1234}' \\
  --output speech.mp3

# Saved speaker identity clone through the OpenAI-style route.
curl -X POST "${base}/audio/speech" \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"current","input":"This uses a saved speaker identity.","voice":"${speakerVoice}","response_format":"wav","max_tokens":1024}' \\
  --output speaker.wav

# Voice clone.
curl -X POST "${base}/higgs/voice-clone" \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"input":"This uses the reference speaker.","reference_audio_path":"C:\\\\voices\\\\speaker.wav","reference_text":"Optional transcript of the reference audio.","response_format":"wav","max_tokens":1024}' \\
  --output clone.wav

# Continue speech.
curl -X POST "${base}/higgs/continue-speech" \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"audio_path":"C:\\\\voices\\\\start.wav","continuation_text":"and this is the continuation.","response_format":"wav","max_tokens":1024}' \\
  --output continued.wav

# Streaming TTS as newline-delimited JSON events. Audio events include wavBase64 chunks
# for live playback; final includes wavBase64 or mp3Base64 for the clean finished file.
curl -N -X POST "${base}/higgs/audio/stream" \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"input":"This streams progress and wav-base64 chunks.","voice":"default","response_format":"mp3","max_tokens":1024}'

# Cancel the active job if needed.
curl -X POST "${base}/higgs/cancel" \\
  -H "Authorization: Bearer ${key}"`;
}
