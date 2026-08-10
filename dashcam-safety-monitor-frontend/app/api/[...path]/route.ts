import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = (
  process.env.INTERNAL_BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");

async function proxyToBackend(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const backendPath = `/api/${path.join("/")}`;
  const url = new URL(request.url);
  const backendUrl = `${BACKEND_URL}${backendPath}${url.search}`;

  try {
    const headers = new Headers();
    request.headers.forEach((value, key) => {
      if (key.toLowerCase() !== "host" && key.toLowerCase() !== "connection") {
        headers.set(key, value);
      }
    });

    const fetchOptions: RequestInit = {
      method: request.method,
      headers,
      signal: AbortSignal.timeout(30000),
    };

    if (request.method !== "GET" && request.method !== "HEAD") {
      fetchOptions.body = await request.arrayBuffer();
    }

    const backendResponse = await fetch(backendUrl, fetchOptions);

    const contentType = backendResponse.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const data = await backendResponse.json();
      return NextResponse.json(data, { status: backendResponse.status });
    }

    const responseBody = await backendResponse.arrayBuffer();
    return new NextResponse(responseBody, {
      status: backendResponse.status,
      headers: {
        "Content-Type": contentType,
      },
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Backend proxy error";
    console.error(`[API Proxy] ${request.method} ${backendPath} error:`, message);
    return NextResponse.json(
      { status: "error", message },
      { status: 502 }
    );
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyToBackend(request, context);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyToBackend(request, context);
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyToBackend(request, context);
}

export async function DELETE(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxyToBackend(request, context);
}
