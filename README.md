# audio2mqtt 1.5

Version: 1.5.0

`audio2mqtt` receives a continuous FFmpeg audio stream, performs local speech recognition in a Docker container, and publishes transcript JSON to stdout, a JSONL file, MQTT and/or webhooks.



## Pipeline

```text
Audio device
  -> FFmpeg
  -> TCP raw PCM 16 kHz mono s16le
TeamSpeak server
  -> audio2mqtt TeamSpeak bot client
  -> per-speaker received audio
  -> Docker container: audio2mqtt
  -> Rust service
  -> Silero VAD V5
  -> whisper-rs / whisper.cpp CUDA
  -> JSONL stdout + ./output/transcripts.jsonl
  -> MQTT client and/or webhook POST
  -> Web admin UI
```

## Start
Tested only in Windows. Feel free to test in Linux and make pull request for documentation and scripts :) 

From PowerShell in the project root:

```powershell
docker compose build --no-cache
docker compose up
```

Web admin:

```text
http://localhost:8080
```

TCP audio input:

```text
localhost:15000
```

## Windows audio devices

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

List devices:

```powershell
.\scripts\list_audio_devices.ps1
```

Copy the exact device name from a line ending with `(audio)`.

## Start FFmpeg stream

```powershell
.\scripts\start_ffmpeg_stream.ps1 -Device "Microphone Array (Realtek(R) Audio)"
```

FFmpeg sends raw PCM to the container TCP port:

```text
16 kHz
mono
signed 16-bit little endian
```

## Web admin

Open:

```text
http://localhost:8080
```

You can configure:

- ASR language, thread count and beam size
- VAD threshold and segment timing
- active model
- model download URL profiles
- Hugging Face GGML model profiles by repo name
- logging level
- frame-level VAD debug on/off
- MQTT enabled/disabled
- MQTT host, port, topic, username, password and QoS
- TeamSpeak receiver enabled/disabled
- TeamSpeak server, nickname, identity and channel selection
- webhook endpoints

Settings are stored in:

```text
./config/config.json
```

## Logging

Default logging is `info` and VAD frame debug is disabled.

Web UI logging options:

```text
error
warn
info
debug
trace
```

VAD frame debug is a separate checkbox because it can produce a lot of logs on a continuous stream.

## Hugging Face models

The web admin includes a built-in profile for:

```text
Finnish-NLP/Finnish-finetuned-whisper-models-ggml-format
ggml-model-fi-large-v3.bin
```

You can also add other Hugging Face GGML Whisper models by repo name in the web admin. Use `owner/model` form, and optionally set a `.bin` file name if the repo contains more than one candidate.

## REST transcription

Endpoint:

```http
POST /api/transcribe
Content-Type: application/json
```

Request body for WAV:

```json
{
  "audio": {
    "format": "wav",
    "encoding": "base64",
    "data": "UklGRiQAAABXQVZF..."
  },
  "context": {
    "device_id": "frontdesk-mic-01",
    "session_id": "abc-789"
  }
}
```

Request body for raw PCM:

```json
{
  "audio": {
    "format": "pcm_s16le",
    "encoding": "base64",
    "sample_rate": 16000,
    "channels": 1,
    "data": "..."
  },
  "context": {
    "source_system": "test-client"
  }
}
```

Supported REST audio formats:

```text
wav: 16 kHz PCM16 WAV, mono or multi-channel. Multi-channel is downmixed to mono.
pcm_s16le: 16 kHz mono signed 16-bit little-endian PCM.
```

REST responses use the same transcript schema as stream events.

## TeamSpeak input

The TeamSpeak receiver connects as a normal client to an external TeamSpeak server. It does not implement or run a TeamSpeak server.

The Docker image includes Opus runtime libraries for TeamSpeak audio decoding.

This build uses a local `tsproto` patch under `vendor/tsproto-0.2.0`. The patch accepts intermediate TeamSpeak license handshake data larger than `0x7f`, which is needed for servers that report no server license and otherwise fail with `Failed to parse license: Invalid data 0x104 in intermediate license`.

For debugging TeamSpeak license handshake parsing, start the container with:

```text
TSPROTO_DUMP_LICENSE=1
```

This writes one raw license blob dump to the container logs during connection setup. Disable it after collecting the log because the dump is verbose.

In the web admin:

- set the TeamSpeak server address, nickname and optional passwords
- keep the generated identity or paste another identity into the identity field
- connect once to fetch the channel list
- select a channel from the dropdown and save it

The selected channel is stored in `./config/config.json` by id and path. On restart the service first tries the saved channel id, then the saved path, and finally falls back to the server default channel so the bot can still connect and refresh the channel list. If the selected channel no longer exists, the TeamSpeak status is shown as `channel_missing`.

TeamSpeak transcript events use the same MQTT topic, webhooks and JSONL output as other inputs. The source identifies the origin:

```json
{
  "source": {
    "type": "teamspeak",
    "src": "ts.example.com:9987",
    "meta": {
      "transport": "teamspeak",
      "client_id": 42,
      "client_name": "Speaker",
      "channel_id": 7,
      "channel_name": "Default",
      "channel_path": "Default",
      "sample_rate_original": 48000,
      "sample_rate": 16000,
      "channels": 1
    }
  }
}
```

## MQTT

No local broker is started by this bundle.

If your MQTT broker runs on the Windows host, use this in the web UI:

```text
host: host.docker.internal
port: 1883
topic: audio2mqtt/transcripts
```

If the broker is elsewhere on the network, use its IP or DNS name.

## Webhooks

Webhook row format in the UI:

```text
name|enabled|url|bearer_token
```

Example without token:

```text
local-api|true|http://host.docker.internal:9000/asr|
```

Example with bearer token:

```text
prod|true|https://example.com/asr|secret-token
```

## Watch transcripts

```powershell
.\scripts\watch_transcripts.ps1
```

Or Docker logs:

```powershell
docker compose logs -f audio2mqtt
```

## GPU test

```powershell
.\scripts\gpu_test.ps1
```

## VRAM monitor

```powershell
.\scripts\watch_gpu.ps1
```

## Output JSON example: stream

```json
{
  "type": "transcript",
  "schema_version": "1.0",
  "id": "9f8c0f14-61e6-49b8-830c-59e84ce8fa9a",
  "ts": "2026-05-10T15:22:31.842Z",
  "source": {
    "type": "stream",
    "src": "192.168.1.42:53421",
    "meta": {
      "transport": "tcp",
      "format": "pcm_s16le",
      "sample_rate": 16000,
      "channels": 1
    }
  },
  "context": null,
  "model": {
    "id": "large-v3",
    "name": "Whisper large-v3",
    "path": "/models/ggml-large-v3.bin"
  },
  "audio": {
    "start_sec": 128.384,
    "end_sec": 136.672,
    "duration_sec": 8.288
  },
  "vad": {
    "enabled": true,
    "engine": "silero",
    "threshold": 0.55
  },
  "result": {
    "language": "fi",
    "text": "This is a transcript from the continuous stream.",
    "segments": [
      {
        "start_sec": 128.384,
        "end_sec": 136.672,
        "text": "This is a transcript from the continuous stream."
      }
    ]
  }
}
```

## Output JSON example: REST

```json
{
  "type": "transcript",
  "schema_version": "1.0",
  "id": "d4a8b4e9-4a75-4e1a-ae33-1787cb6f94fd",
  "ts": "2026-05-10T15:25:02.191Z",
  "source": {
    "type": "rest",
    "src": "192.168.1.100:60142",
    "meta": {
      "endpoint": "/api/transcribe",
      "format": "wav",
      "encoding": "base64",
      "sample_rate": 16000,
      "channels": 1
    }
  },
  "context": {
    "device_id": "frontdesk-mic-01",
    "session_id": "abc-789"
  },
  "model": {
    "id": "large-v3",
    "name": "Whisper large-v3",
    "path": "/models/ggml-large-v3.bin"
  },
  "audio": {
    "start_sec": 0.0,
    "end_sec": 4.736,
    "duration_sec": 4.736
  },
  "vad": {
    "enabled": false,
    "engine": "none",
    "threshold": null
  },
  "result": {
    "language": "fi",
    "text": "This is a transcript from a REST audio clip.",
    "segments": []
  }
}
```
