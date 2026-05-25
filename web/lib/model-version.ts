/**
 * model-version.ts — Resolve the current model version label.
 *
 * The version string identifies *which* deployment of the ONNX model is
 * running, so Langfuse traces can be segmented by model version.
 *
 * Resolution order (first wins):
 *   1. process.env.MODEL_VERSION          — Vercel env var override
 *   2. web/app/api/predict/models/version.json  — written by deploy_pipeline.py
 *   3. "unknown"                           — fallback when neither is available
 */

import fs from "fs";
import path from "path";

let _version = "";  // cached version string

/** Path to version.json in the bundled serverless output directory. */
const VERSION_JSON_PATH = path.resolve(__dirname, "models", "version.json");

export function getModelVersion(): string {
  if (_version) return _version;

  // 1. Env var override
  const envVersion = process.env.MODEL_VERSION;
  if (envVersion) {
    _version = envVersion;
    return _version;
  }

  // 2. version.json (bundled with the app or copied by deploy_pipeline.py)
  try {
    if (fs.existsSync(VERSION_JSON_PATH)) {
      const raw = fs.readFileSync(VERSION_JSON_PATH, "utf-8");
      const parsed = JSON.parse(raw);
      if (parsed.version && typeof parsed.version === "string") {
        _version = parsed.version;
        return _version;
      }
    }
  } catch {
    // Ignore read/parse errors — fall through to default
  }

  // 3. Fallback
  _version = "unknown";
  return _version;
}
