import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // onnxruntime-node is a native module used only in API routes.
  // This prevents Next.js from trying to bundle it with client code.
  serverExternalPackages: ["onnxruntime-node"],

  // Ensure the ONNX model is included in the serverless function bundle.
  // Both this config and web/vercel.json target the same file as a belt-and-suspenders approach.
  experimental: {
    outputFileTracingIncludes: {
      "/api/predict": ["app/api/predict/models/crnn-transformer.onnx"],
    },
  },
};

export default nextConfig;
