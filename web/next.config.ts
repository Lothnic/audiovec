import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // onnxruntime-node is a native module used only in API routes.
  // This prevents Next.js from trying to bundle it with client code.
  serverExternalPackages: ["onnxruntime-node"],

  // Ensure the ONNX model file is included in the serverless function bundle.
  // The model lives at web/app/api/predict/models/ alongside the route handler
  // so Next.js traces it automatically into the built function output.
  // inference.ts resolves it via path.resolve(__dirname, "models", ...).
  experimental: {
    outputFileTracingIncludes: {
      "/api/predict": ["app/api/predict/models/crnn-transformer.onnx"],
    },
  },
};

export default nextConfig;
