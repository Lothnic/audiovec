/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    if (isServer) {
      // onnxruntime-node is a native package — defeat Next.js eager bundling
      config.externals.push(
        { "onnxruntime-node": "commonjs onnxruntime-node" },
        /\.node$/
      );
    }
    return config;
  },
};

export default nextConfig;
