import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // onnxruntime-node is a native module used only in API routes.
  // This prevents Next.js from trying to bundle it with client code.
  serverExternalPackages: ["onnxruntime-node"],

  // Best-effort: try to include the ONNX model via file tracing.
  // Inference.ts has a runtime download fallback if this doesn't work.
  experimental: {
    outputFileTracingIncludes: {
      "/api/predict": ["app/api/predict/models/crnn-transformer.onnx"],
    },
  },
};

export default nextConfig;
