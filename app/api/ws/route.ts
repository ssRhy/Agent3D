// app/api/ws/route.ts
import { NextResponse } from "next/server";

export async function GET() {
  // 获取WebSocket服务地址
  // 通常在生产环境中，这会通过环境变量或配置文件获取
  const websocketUrl =
    process.env.WEBSOCKET_URL ||
    `ws://${process.env.HOSTNAME || "localhost"}:3001`;

  return NextResponse.json({
    websocket_url: websocketUrl,
    status: "ok",
  });
}
