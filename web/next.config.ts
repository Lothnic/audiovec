import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // onnxruntime-node is a native module used only in API routes.
  // This prevents Next.js from trying to bundle it with client code.
  serverExternalPackages: ["onnxruntime-node"],

  // Best-effort: try to include the ONNX model files via file tracing.
  // Inference.ts has a runtime download fallback if tracing doesn't pick them up.
  experimental: {
    outputFileTracingIncludes: {
      "/api/predict": [
        "app/api/predict/models/crnn-transformer.onnx",
        "app/api/predict/models/crnn-transformer.onnx.data",
        "app/api/predict/models/version.json",
      ],
    },
  },
};

export default nextConfig;
