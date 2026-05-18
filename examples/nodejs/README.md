# audiovec — Node.js ONNX Inference

Run audiovec's emotion recognition model directly from Node.js using
[ONNX Runtime](https://onnxruntime.ai/).

## Prerequisites

- Node.js 18+
- The ONNX model exported from the trained PyTorch checkpoint:
  ```
  uv run python -m audiovec.export_onnx
  ```
  (produces `models/crnn-transformer.onnx`)

## Setup

```bash
cd examples/nodejs
npm install
```

## Usage

### Basic prediction

```bash
node predict.js speech.wav
```

Output:

```
  😌  CALM  (89.2%)

  Top emotions:
    calm       89.2%  ██████████████████████████████
    neutral     6.1%  ██
    happy       2.9%  █

  Embedding dimension: 256
  Embedding range:     [0.0000, 3.2841]
```

### Options

```bash
# Show top 5 emotions
node predict.js speech.wav --top-k 5

# Output raw JSON (logits + embedding)
node predict.js speech.wav --json
```

## How it works

1. **Read WAV** — uses `wav-decoder` to parse PCM samples.
2. **Mel-spectrogram** — a pure-JS implementation replicates
   `librosa.feature.melspectrogram` with the same parameters
   (22050 Hz, 128 mel bands, 8 kHz max freq, 2048 FFT, 512 hop).
3. **ONNX inference** — the spectrogram is fed to `crnn-transformer.onnx`
   via `onnxruntime-node`. The model returns both logits (8 emotions) and the
   256-d embedding vector.
4. **Softmax** — logits are converted to probabilities in JS.

## API Reference

### `predict.js` (CLI)

```bash
node predict.js <path-to-wav> [options]
```

| Option       | Description                  | Default |
|--------------|------------------------------|---------|
| `--top-k <n>` | Show top-N emotions         | 3       |
| `--json`      | Output raw JSON              | false   |
| `--help`      | Show help                    |         |

### `melspectrogram.js` (library)

```js
const { audioToSpectrogram } = require("./melspectrogram");

const samples = new Float64Array([...]); // PCM mono audio
const sr = 22050;                         // sample rate
const { data, nMels, nFrames } = audioToSpectrogram(samples, sr);
// data is a Float64Array of length 128 * 228, normalized to [0, 1]
```

## Files

| File              | Purpose                                      |
|-------------------|----------------------------------------------|
| `predict.js`      | CLI inference script                         |
| `melspectrogram.js` | Pure-JS mel-spectrogram computation        |
| `package.json`    | npm dependencies (`onnxruntime-node`, `wav-decoder`) |

## Notes

- Only **WAV** files are supported. Convert MP3/M4A/etc. with `ffmpeg` first:
  ```
  ffmpeg -i input.mp3 -ar 22050 -ac 1 output.wav
  ```
- The model expects mono audio at 22050 Hz. Input is automatically resampled
  if the sample rate differs.
- The ONNX model was exported with opset 18 and validated against the original
  PyTorch model.
