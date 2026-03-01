import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  turbopack: {
    // Tell Next.js that this webapp/ dir is the workspace root,
    // preventing it from picking up the parent repo's lockfile.
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
