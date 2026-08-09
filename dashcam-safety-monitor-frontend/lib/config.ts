/**
 * Dynamic resolution of Backend API and WebSocket URLs.
 * Works seamlessly for:
 * 1. Local development (localhost:3000 <-> localhost:8000)
 * 2. Single-port exposure via tunnels (Ngrok, Cloudflare, Localtunnel) where Next.js proxies to backend
 * 3. Custom environment variables (NEXT_PUBLIC_API_URL / NEXT_PUBLIC_WS_URL)
 * 4. URL query overrides (?api=... or ?ws=...)
 */
export function getBackendUrls(): { apiUrl: string; wsUrl: string } {
  if (typeof window === "undefined") {
    const apiUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/+$/, "");
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/video";
    return { apiUrl, wsUrl };
  }

  // 1. Check URL query parameters (e.g., ?api=https://... or ?ws=wss://...)
  const params = new URLSearchParams(window.location.search);
  const queryApi = params.get("api") || params.get("apiUrl");
  const queryWs = params.get("ws") || params.get("wsUrl");

  const isHttpsPage = window.location.protocol === "https:";
  const hostname = window.location.hostname;
  const isLocalhost = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "[::1]";

  // Determine API Base URL
  let apiUrl = queryApi || process.env.NEXT_PUBLIC_API_URL || "";
  if (!apiUrl) {
    if (!isLocalhost) {
      // Remote host or tunnel (Ngrok, Cloudflare, etc.): use frontend origin so Next.js proxies to backend
      apiUrl = `${window.location.protocol}//${window.location.host}`;
    } else {
      // Local development fallback
      apiUrl = `${window.location.protocol}//${hostname}:8000`;
    }
  }

  // Upgrade HTTP to HTTPS if main page is loaded over HTTPS
  if (isHttpsPage && apiUrl.startsWith("http://")) {
    apiUrl = apiUrl.replace("http://", "https://");
  }

  // Strip trailing slashes
  apiUrl = apiUrl.replace(/\/+$/, "");

  // Determine WebSocket URL
  let wsUrl = queryWs || process.env.NEXT_PUBLIC_WS_URL || "";
  if (!wsUrl) {
    const wsProtocol = apiUrl.startsWith("https://") || isHttpsPage ? "wss:" : "ws:";
    const cleanHost = apiUrl.replace(/^https?:\/\//, "");
    wsUrl = `${wsProtocol}//${cleanHost}/ws/video`;
  } else {
    // Upgrade WS to WSS if page is loaded over HTTPS
    if (isHttpsPage && wsUrl.startsWith("ws://")) {
      wsUrl = wsUrl.replace("ws://", "wss://");
    }
  }

  return { apiUrl, wsUrl };
}
