import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // onnxruntime-node is a native module used only in API routes.
  // This prevents Next.js from trying to bundle it with client code.
  serverExternalPackages: ["onnxruntime-node"],

  // Ensure the ONNX model file is included in the serverless function bundle.
  // The path is relative to this file (the Next.js project root, i.e. web/).
  // The model lives at ../models/crnn-transformer.onnx and is loaded dynamically
  // by inference.ts via path.resolve(process.cwd(), "..", "models", ...).
  experimental: {
    outputFileTracingIncludes: {
      "/api/predict": ["../models/crnn-transformer.onnx"],
    },
  },
};

export default nextConfig;
