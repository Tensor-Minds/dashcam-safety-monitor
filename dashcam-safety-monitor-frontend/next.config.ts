import type { NextConfig } from "next";

const backendUrl = (
  process.env.INTERNAL_BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
  async rewrites() {
    return [
      {
        source: "/ws/:path*",
        destination: `${backendUrl}/ws/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${backendUrl}/static/:path*`,
      },
    ];
  },
};

export default nextConfig;
