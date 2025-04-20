"use client";

import React, { useState, useRef, useEffect } from "react";
import ThreeCanvas, { ThreeCanvasRef } from "./ThreeCanvas";
import CodeEditor from "./CodeEditor";
import ConversationLog, { Message } from "./ConversationLog";

// 定义WebSocket消息类型
interface WebSocketMessage {
  type: string;
  content?: string;
  code?: string;
  requestId?: string;
  description?: string;
  message?: string;
  isValid?: boolean;
  status?: string;
  stage?: string;
  result?: string;
  error?: string;
  quality?: string;
  view?: string;
  detail?: string;
  focus?: string;
}

// 定义挂起请求的回调类型
type PendingRequestCallback = (result: any) => void;

export default function AgentWorkspace() {
  const [userInput, setUserInput] = useState<string>("");
  const [conversation, setConversation] = useState<Message[]>([]);
  const [isAgentWorking, setIsAgentWorking] = useState<boolean>(false);
  const [currentCode, setCurrentCode] = useState<string>("");
  const threeCanvasRef = useRef<ThreeCanvasRef | null>(null);
  const [socketReady, setSocketReady] = useState<boolean>(false);
  const requestIdRef = useRef<number>(0);
  const pendingRequestsRef = useRef<Record<string, PendingRequestCallback>>({});
  const webSocketRef = useRef<WebSocket | null>(null);
  const [canvasReady, setCanvasReady] = useState<boolean>(false);

  // 初始化WebSocket连接
  useEffect(() => {
    let wsInstance: WebSocket | null = null;
    let retryCount = 0;
    let retryTimeout: NodeJS.Timeout | null = null;
    const MAX_RETRIES = 5;

    // 创建WebSocket连接函数
    const connectWebSocket = (url: string): WebSocket => {
      console.log(
        `尝试连接WebSocket: ${url} (重试: ${retryCount}/${MAX_RETRIES})`
      );

      // 确保URL包含/ws路径
      let wsUrl = url;

      const socket = new WebSocket(wsUrl);
      webSocketRef.current = socket;

      // 添加全局引用供Agent使用
      if (typeof window !== "undefined") {
        (window as any)._threeJsAgentWebSocket = socket;
      }

      socket.onopen = () => {
        console.log("WebSocket连接已建立");
        setSocketReady(true);
        retryCount = 0; // 重置重试计数
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          console.log("收到WebSocket消息:", message.type);
          handleMessage(message);
        } catch (error) {
          console.error("解析WebSocket消息出错:", error);
        }
      };

      socket.onclose = (event) => {
        console.log(
          `WebSocket连接已关闭 (code: ${event.code}, reason: ${event.reason})`
        );
        setSocketReady(false);
        retryConnection();
      };

      socket.onerror = (error) => {
        console.error("WebSocket错误:", error);
        setSocketReady(false);

        if (retryCount === 0) {
          addToConversation({
            role: "system",
            content: "WebSocket连接出错，正在尝试重新连接...",
            type: "error",
          });
        }
      };

      return socket;
    };

    // 重试连接函数
    const retryConnection = () => {
      if (retryCount < MAX_RETRIES) {
        retryCount++;
        const delay = Math.min(1000 * Math.pow(2, retryCount), 10000); // 指数退避策略，最大10秒

        console.log(`将在 ${delay}ms 后重试WebSocket连接...`);
        if (retryTimeout) clearTimeout(retryTimeout);

        retryTimeout = setTimeout(() => {
          if (wsInstance) {
            try {
              wsInstance.close();
            } catch (e) {
              // 忽略关闭错误
            }
          }
          wsInstance = connectWebSocket(wsUrl);
        }, delay);
      } else {
        console.error(`WebSocket连接失败，已达到最大重试次数 (${MAX_RETRIES})`);
        addToConversation({
          role: "system",
          content: "WebSocket连接失败，请刷新页面重试或检查服务器是否运行",
          type: "error",
        });
      }
    };

    // 首先获取WebSocket服务地址
    let wsUrl = "";
    fetch("/api/ws")
      .then((res) => res.json())
      .then((data) => {
        console.log("WebSocket服务信息:", data);
        wsUrl = data.websocket_url;
        wsInstance = connectWebSocket(wsUrl);
      })
      .catch((error) => {
        console.error("获取WebSocket服务信息失败:", error);

        // 使用默认WebSocket地址
        console.log("使用默认WebSocket地址");
        wsUrl = `ws://${window.location.hostname}:3001`;
        wsInstance = connectWebSocket(wsUrl);

        addToConversation({
          role: "system",
          content: "无法获取WebSocket服务信息，尝试使用默认连接",
          type: "warning",
        });
      });

    return () => {
      // 清理函数
      if (retryTimeout) clearTimeout(retryTimeout);

      // 关闭WebSocket连接
      if (wsInstance) {
        try {
          wsInstance.close();
        } catch (e) {
          // 忽略关闭错误
        }
      }

      webSocketRef.current = null;
      if (typeof window !== "undefined") {
        (window as any)._threeJsAgentWebSocket = null;
      }
    };
  }, []);

  // 处理收到的消息
  function handleMessage(message: WebSocketMessage): void {
    try {
      // 处理各种消息类型
      switch (message.type) {
        case "agent_thinking":
        case "ai_thinking":
          addToConversation({
            role: "agent",
            content: message.message || message.content || "AI正在思考...",
            type: "thinking",
          });
          break;

        case "agent_message":
          // 检查消息内容中是否包含代码块
          const codeRegex = /```(?:html|javascript)?\s*([\s\S]*?)```/g;
          let codeMatches: string[] = [];
          let match: RegExpExecArray | null;
          let content = message.content || "";

          // 提取所有代码块
          while ((match = codeRegex.exec(content)) !== null) {
            codeMatches.push(match[1]);
          }

          // 如果找到代码块，提取Three.js相关代码并执行
          if (codeMatches.length > 0) {
            // 查找HTML代码块
            const htmlCode = codeMatches.find(
              (code) =>
                code.includes("<!DOCTYPE html>") ||
                code.includes("<html") ||
                (code.includes("<script") && code.includes("THREE"))
            );

            if (htmlCode) {
              console.log("找到Three.js HTML代码，准备执行");

              // 改进的脚本提取正则表达式，能够处理带属性的script标签和不带属性的script标签
              const scriptRegex =
                /<script(?! src=|.*?src=).*?>([\s\S]*?)<\/script>/g;
              const scriptMatches: string[] = [];
              let scriptMatch: RegExpExecArray | null;

              while ((scriptMatch = scriptRegex.exec(htmlCode)) !== null) {
                if (scriptMatch[1] && scriptMatch[1].trim().length > 0) {
                  scriptMatches.push(scriptMatch[1]);
                }
              }

              // 如果找到<script>标签内容，执行代码
              if (scriptMatches.length > 0) {
                // 合并所有脚本内容
                let threeJsCode = scriptMatches.join("\n\n");

                // 预处理代码，移除不兼容的部分
                threeJsCode = threeJsCode
                  // 移除renderer创建和添加到DOM的代码
                  .replace(
                    /const\s+renderer\s*=\s*new\s+THREE\.WebGLRenderer[^;]*;/g,
                    "// renderer已由ThreeCanvas提供"
                  )
                  .replace(
                    /document\.body\.appendChild\s*\(\s*renderer\.domElement\s*\)\s*;/g,
                    "// canvas已由ThreeCanvas提供"
                  )
                  .replace(
                    /renderer\.setSize\s*\([^;]*\)\s*;/g,
                    "// 尺寸已由ThreeCanvas管理"
                  )
                  // 修改动画循环
                  .replace(
                    /function\s+animate\s*\(\s*\)\s*{[\s\S]*?}/g,
                    function (match) {
                      // 保留动画循环内部逻辑，但移除requestAnimationFrame调用
                      return match.replace(
                        /requestAnimationFrame\s*\(\s*animate\s*\)\s*;/g,
                        "// 动画循环由ThreeCanvas管理"
                      );
                    }
                  )
                  // 移除animate()调用
                  .replace(/animate\s*\(\s*\)\s*;/g, "// 动画由ThreeCanvas管理")
                  // 移除window事件监听
                  .replace(
                    /window\.addEventListener\s*\(\s*['"]resize['"][^;]*;/g,
                    "// 窗口大小调整由ThreeCanvas管理"
                  );

                // 添加必要的资源跟踪代码
                threeJsCode += `
                // 将创建的对象添加到资源跟踪
                if (typeof apple !== 'undefined') {
                  trackResource(apple, 'object');
                }
                if (typeof appleGeometry !== 'undefined') {
                  trackResource(appleGeometry, 'geometry');
                }
                if (typeof appleMaterial !== 'undefined') {
                  trackResource(appleMaterial, 'material');
                }
                `;

                console.log("处理后的Three.js代码:", threeJsCode);

                // 设置当前代码用于显示
                setCurrentCode(threeJsCode);

                // 执行代码
                if (canvasReady && threeCanvasRef.current) {
                  try {
                    threeCanvasRef.current.executeCode(threeJsCode);

                    // 添加执行成功消息
                    addToConversation({
                      role: "system",
                      content: `执行Three.js代码成功`,
                      type: "success",
                    });
                  } catch (error) {
                    console.error("执行Three.js代码失败:", error);
                    addToConversation({
                      role: "system",
                      content: `执行Three.js代码失败: ${error}`,
                      type: "error",
                    });
                  }
                } else {
                  console.error("Three.js Canvas未准备好");
                }
              }
            }
          }

          // 添加AI响应到对话
          addToConversation({
            role: "assistant",
            content: message.content || "",
          });
          setIsAgentWorking(false);
          break;

        case "execute":
          console.log("收到执行代码消息:", message.code);
          if (message.code) setCurrentCode(message.code);

          // 确保Canvas已准备好再执行代码
          if (canvasReady && threeCanvasRef.current && message.code) {
            executeCode(
              message.code,
              message.requestId || Date.now().toString()
            );
          } else {
            console.warn("Canvas未准备好或代码为空，延迟执行");
            setTimeout(() => {
              if (threeCanvasRef.current && message.code) {
                console.log("延迟执行代码");
                executeCode(
                  message.code,
                  message.requestId || Date.now().toString()
                );
              } else {
                console.error("Canvas仍未准备好或代码为空，无法执行代码");
                addToConversation({
                  role: "system",
                  content: "3D渲染环境未准备好，无法执行代码",
                  type: "error",
                });
              }
            }, 1500);
          }
          break;

        case "code_generated":
          console.log("收到生成的代码:", message.code);
          if (message.code) setCurrentCode(message.code);

          // 从代码中提取Three.js脚本部分
          if (message.code) {
            const extractedCode = extractThreeJsCode(message.code);
            if (extractedCode && canvasReady && threeCanvasRef.current) {
              console.log("执行提取的Three.js代码");
              executeCode(extractedCode, Date.now().toString());
            }
          }
          break;

        case "tool_response":
          // 处理工具响应
          if (message.requestId) {
            const callback = pendingRequestsRef.current[message.requestId];
            if (callback) {
              callback(message.result);
              delete pendingRequestsRef.current[message.requestId];
            }
          }
          break;
      }
    } catch (error) {
      const err = error as Error;
      console.error("处理消息错误:", err);
    }
  }

  // 提取HTML中的Three.js代码
  function extractThreeJsCode(htmlCode: string): string | null {
    if (!htmlCode || typeof htmlCode !== "string") return null;

    // 检查是否是HTML代码
    if (
      htmlCode.includes("<!DOCTYPE html>") ||
      htmlCode.includes("<html") ||
      (htmlCode.includes("<script") && htmlCode.includes("THREE"))
    ) {
      // 提取script标签内容
      const scriptRegex = /<script(?! src=|.*?src=).*?>([\s\S]*?)<\/script>/g;
      const scriptMatches: string[] = [];
      let scriptMatch: RegExpExecArray | null;

      while ((scriptMatch = scriptRegex.exec(htmlCode)) !== null) {
        if (scriptMatch[1] && scriptMatch[1].trim().length > 0) {
          scriptMatches.push(scriptMatch[1]);
        }
      }

      if (scriptMatches.length > 0) {
        // 合并所有脚本内容
        let threeJsCode = scriptMatches.join("\n\n");

        // 预处理代码
        threeJsCode = threeJsCode
          // 移除renderer创建和添加到DOM的代码
          .replace(
            /const\s+renderer\s*=\s*new\s+THREE\.WebGLRenderer[^;]*;/g,
            "// renderer已由ThreeCanvas提供"
          )
          .replace(
            /document\.body\.appendChild\s*\(\s*renderer\.domElement\s*\)\s*;/g,
            "// canvas已由ThreeCanvas提供"
          )
          .replace(
            /renderer\.setSize\s*\([^;]*\)\s*;/g,
            "// 尺寸已由ThreeCanvas管理"
          )
          // 保留动画循环的内部逻辑
          .replace(
            /function\s+animate\s*\(\s*\)\s*{[\s\S]*?}/g,
            function (match) {
              return match.replace(
                /requestAnimationFrame\s*\(\s*animate\s*\)\s*;/g,
                "// 动画循环由ThreeCanvas管理"
              );
            }
          )
          // 移除animate()调用
          .replace(/animate\s*\(\s*\)\s*;/g, "// 动画由ThreeCanvas管理")
          // 移除window事件监听
          .replace(
            /window\.addEventListener\s*\(\s*['"]resize['"][^;]*;/g,
            "// 窗口大小调整由ThreeCanvas管理"
          );

        return threeJsCode;
      }
    }

    // 如果不是HTML或没有script标签，可能是纯JS代码，直接返回
    return htmlCode;
  }

  // 通过WebSocket发送消息
  function sendWebSocketMessage(message: WebSocketMessage): boolean {
    if (
      !webSocketRef.current ||
      webSocketRef.current.readyState !== WebSocket.OPEN
    ) {
      console.error("WebSocket未连接");
      addToConversation({
        role: "system",
        content: "WebSocket未连接，无法发送消息",
        type: "error",
      });
      return false;
    }

    try {
      console.log("发送WebSocket消息:", message);
      webSocketRef.current.send(JSON.stringify(message));
      return true;
    } catch (error) {
      const err = error as Error;
      console.error("发送WebSocket消息出错:", err);
      addToConversation({
        role: "system",
        content: `发送消息失败: ${err.message}`,
        type: "error",
      });
      return false;
    }
  }

  // 添加消息到对话
  function addToConversation(message: Message): void {
    setConversation((prev) => [...prev, message]);
  }

  // 启动Agent
  async function startAgent(): Promise<void> {
    if (!socketReady || !userInput.trim()) return;

    setIsAgentWorking(true);
    addToConversation({
      role: "user",
      content: userInput,
    });

    // 通过WebSocket发送用户输入
    const success = sendWebSocketMessage({
      type: "user_input",
      content: userInput,
    });

    if (!success) {
      setIsAgentWorking(false);
    }

    setUserInput("");
  }

  // 执行Three.js代码
  async function executeCode(code: string, requestId: string): Promise<void> {
    if (!threeCanvasRef.current) {
      console.error("Three.js Canvas组件未初始化");
      sendWebSocketMessage({
        type: "tool_response",
        requestId,
        result: JSON.stringify({
          success: false,
          error: "Three.js Canvas组件未初始化",
        }),
      });
      return;
    }

    try {
      console.log("执行Three.js代码");
      const result = await threeCanvasRef.current.executeCode(code);

      // 返回执行结果
      sendWebSocketMessage({
        type: "tool_response",
        requestId,
        result: JSON.stringify(result),
      });

      // 显示执行结果
      addToConversation({
        role: "system",
        content: `3D场景${result.success ? "成功" : "失败"}渲染`,
        type: "code_execution",
      });
    } catch (error) {
      const err = error as Error;
      console.error("执行代码错误:", err);

      sendWebSocketMessage({
        type: "tool_response",
        requestId,
        result: JSON.stringify({
          success: false,
          error: err.message,
        }),
      });
    }
  }

  // 处理Canvas引用设置
  const handleCanvasRef = (ref: ThreeCanvasRef | null) => {
    threeCanvasRef.current = ref;
    if (ref) {
      setCanvasReady(true);
    }
  };

  return (
    <div className="flex h-screen">
      {/* 左侧面板：对话和代码编辑器 */}
      <div className="w-1/2 flex flex-col border-r border-gray-200">
        {/* 对话历史 */}
        <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
          <ConversationLog messages={conversation} />
        </div>

        {/* 用户输入 */}
        <div className="p-4 border-t border-gray-200">
          <textarea
            className="w-full p-2 border rounded"
            rows={3}
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="描述你想创建的3D场景..."
            disabled={isAgentWorking}
          />
          <div className="flex justify-between mt-2">
            <button
              className={`px-4 py-2 rounded ${
                isAgentWorking ? "bg-gray-300" : "bg-blue-500 text-white"
              }`}
              onClick={startAgent}
              disabled={isAgentWorking || !socketReady}
            >
              {isAgentWorking ? "代理思考中..." : "开始生成"}
            </button>
            <button
              className="px-4 py-2 text-gray-700 border rounded"
              onClick={() => setConversation([])}
            >
              清空对话
            </button>
          </div>
        </div>

        {/* 代码编辑器 */}
        <div className="h-1/3 border-t border-gray-200">
          <CodeEditor
            code={currentCode}
            readOnly={isAgentWorking}
            onChange={setCurrentCode}
            onExecute={(code) => {
              if (code && threeCanvasRef.current) {
                const requestId = Date.now().toString();
                executeCode(code, requestId);
              } else {
                addToConversation({
                  role: "system",
                  content: "无法执行代码：代码为空或渲染器未准备好",
                  type: "error",
                });
              }
            }}
          />
        </div>
      </div>

      {/* 右侧面板：Three.js渲染区域 */}
      <div className="w-1/2 bg-black">
        <ThreeCanvas ref={handleCanvasRef} />
      </div>
    </div>
  );
}
