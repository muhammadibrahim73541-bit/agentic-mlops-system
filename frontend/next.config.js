/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: true },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://didactic-giggle-x5q77xwprg5v26xjv-8000.app.github.dev/:path*',
      },
    ]
  },
}

module.exports = nextConfig
