"use client";

import { useRef, useEffect } from "react";

// 定义消息类型
export interface Message {
  role: "user" | "agent" | "system" | "assistant";
  content: string;
  type?:
    | "thinking"
    | "error"
    | "warning"
    | "success"
    | "screenshot"
    | "analysis"
    | "code_execution";
  imageUrl?: string;
}

interface ConversationLogProps {
  messages: Message[];
}

export default function ConversationLog({
  messages = [],
}: ConversationLogProps) {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // 自动滚动到最新消息
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className="flex flex-col space-y-4">
      {messages.length === 0 ? (
        <div className="text-center text-gray-500 p-4">
          <p>还没有对话，请输入描述开始创建3D场景</p>
        </div>
      ) : (
        messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            {message.role === "user" ? (
              <div className="bg-blue-100 p-3 rounded-lg">
                <div className="font-semibold text-blue-800 mb-1">用户</div>
                <div className="text-gray-800">{message.content}</div>
              </div>
            ) : message.role === "agent" ? (
              <div
                className={`p-3 rounded-lg ${
                  message.type === "thinking" ? "bg-gray-100" : "bg-green-100"
                }`}
              >
                <div className="font-semibold text-green-800 mb-1">
                  {message.type === "thinking" ? "Agent思考中..." : "Agent"}
                </div>
                <div className="text-gray-800">{message.content}</div>
              </div>
            ) : (
              <div className="bg-gray-100 p-3 rounded-lg">
                <div className="font-semibold text-gray-800 mb-1">系统</div>
                <div className="text-gray-700">{message.content}</div>
                {message.type === "screenshot" && message.imageUrl && (
                  <div className="mt-2"></div>
                )}
              </div>
            )}
          </div>
        ))
      )}
      <div ref={messagesEndRef} />
    </div>
  );
}
