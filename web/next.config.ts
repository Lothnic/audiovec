import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // onnxruntime-node is a native module used only in API routes.
  // This prevents Next.js from trying to bundle it with client code.
  serverExternalPackages: ["onnxruntime-node"],

  // Model is copied into the serverless function output manually by the
  // vercel-build script in package.json (post-next-build copy step).
  // inference.ts resolves it via path.resolve(__dirname, "models", ...).
};

export default nextConfig;
