import dotenv from "dotenv";
dotenv.config();

import { StringOutputParser } from "@langchain/core/output_parsers";

import { AzureChatOpenAI } from "@langchain/openai";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";

import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { RunnableSequence } from "@langchain/core/runnables";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { BufferMemory } from "langchain/memory";
import readline from "readline";
import fs from "fs";
import path from "path";
import { z } from "zod";
import { WebSocketServer, WebSocket } from "ws";
import * as diff from "diff";
import * as DiffMatchPatch from "diff-match-patch";

// 创建WebSocket服务器
const wss = new WebSocketServer({
  port: process.env.WS_PORT ? parseInt(process.env.WS_PORT) : 3001,
});
let activeConnections = [];

// 保存当前代码的状态
let currentThreeJsCode = "";

// 初始化BufferMemory用于存储对话历史和代码变更
const memory = new BufferMemory({
  memoryKey: "chat_history",
  returnMessages: true,
  inputKey: "input",
  outputKey: "output",
});

// 初始化diff-match-patch
const dmp = new DiffMatchPatch.diff_match_patch();

// WebSocket服务器连接事件
wss.on("connection", (ws) => {
  console.log("Client connected to WebSocket");
  activeConnections.push(ws);

  // 设置心跳间隔
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
      console.log("Ping sent to client");
    }
  }, 30000); // 每30秒发送一次ping

  ws.on("close", () => {
    console.log("Client disconnected from WebSocket");
    clearInterval(pingInterval); // 清除心跳间隔
    activeConnections = activeConnections.filter((conn) => conn !== ws);
  });

  ws.on("message", (message) => {
    try {
      const data = JSON.parse(message.toString());
      console.log("Received message from client:", data);

      // 处理来自前端的消息
      if (data.type === "user_input" || data.type === "user_prompt") {
        console.log("User input:", data.content);
        // 通知客户端正在处理请求
        ws.send(
          JSON.stringify({
            type: "agent_thinking",
            content: "正在生成您的Three.js代码...",
          })
        );

        // 发送generation_started消息
        ws.send(
          JSON.stringify({
            type: "generation_started",
          })
        );

        // 调用优化后的mainAgent处理用户输入
        mainAgent(data.content)
          .then((result) => {
            // 发送处理结果状态消息
            ws.send(
              JSON.stringify({
                type: "agent_message",
                content: `已处理请求: ${
                  result.status === "success" ? "成功" : "需要修正"
                }`,
              })
            );

            // 如果代码生成成功，确保代码被发送
            if (result.status === "success" && result.code) {
              // 使用code_generated类型，与前端接收逻辑匹配
              ws.send(
                JSON.stringify({
                  type: "code_generated",
                  code: result.code,
                  diffInfo: result.diffInfo,
                })
              );
              console.log("发送代码完成，长度:", result.code.length);
            } else if (result.status === "needs_revision") {
              // 发送错误反馈信息
              ws.send(
                JSON.stringify({
                  type: "error",
                  content: result.feedback,
                })
              );
            }
          })
          .catch((error) => {
            console.error("Error processing request:", error);
            ws.send(
              JSON.stringify({
                type: "error",
                content: "处理请求时发生错误，请重试",
              })
            );
          });
      }
    } catch (error) {
      console.error("Error parsing message:", error);
    }
  });
});

// 配置 Azure OpenAI
const llm = new AzureChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
  azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
  azureOpenAIApiDeploymentName: process.env.AZURE_OPENAI_API_DEPLOYMENT_NAME,
  azureOpenAIApiVersion: "2024-02-15-preview",
});

// ==================== 用户输入分析 prompt ====================
const userInputAnalysisPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `你是一名3D场景分析专家，负责理解用户的需求,将用户需求转化拆解为每一个小的部分。遵循以下原则：
    -理解用户请求的3d模型
    -分析3d模型由什么组成部分组成的
    专注于场景分析。`,
  ],
  ["human", "{input}"],
]);

// 创建用户输入分析链
const userInputAnalysisChain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
  },
  userInputAnalysisPrompt,
  llm,
  new StringOutputParser(),
]);

// ==================== 代码生成 prompt ====================
const codeGenerationPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `你是一名专业的Three.js代码生成专家，需要基于用户的需求，对现有代码进行精确修改。请严格遵循以下规则：

  DIFF GENERATION APPROACH:
      - 仔细阅读现有代码，理解其结构和功能
      - 只添加、修改或删除必要的代码部分，而不是重写整个文件
      - 保持现有的代码风格、缩进和注释风格
      - 确保你的修改与现有代码无缝集成
      - 在需要时重用现有的组件、函数和变量

  CODE STRUCTURE:
      - 如果已有现有场景代码，你应该在其基础上添加新的元素，而不是创建全新的场景
      - 保留已有代码的核心结构，只修改或添加用户请求的新功能
      - 确保场景初始化代码只出现一次
      - 完整的HTML文档，包含适当的头部和主体部分
      - 使用Three.js 0.153.0版本的ES模块导入
      - 场景/相机/渲染器初始化
      - 用户请求的对象和功能
      - 动画循环和事件处理程序
      - 单HTML文件实现，便于在线平台直接运行
      - 内联CSS样式
   
   TECHNICAL REQUIREMENTS:
      - 使用importmap的现代ES模块导入
      - 兼容性的ES模块填充
      - 正确的组件初始化顺序
      - 具有调整大小处理程序的响应式设计
      - 清晰、有注释的代码结构
      - 处理模型加载的异步情况，包括加载进度和错误处理
  
   专注于将描述转换为有效代码，只修改必要的部分。
`,
  ],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

// 创建代码生成链并集成记忆
const codeGenerationChain = RunnableSequence.from([
  {
    input: (initialInput) => initialInput.input,
    chat_history: async () =>
      memory.loadMemoryVariables({}).then((res) => res.chat_history),
  },
  codeGenerationPrompt,
  llm,
  new StringOutputParser(),
]);

// 应用差异到代码
function applyDiff(originalText, patchText) {
  try {
    // 首先尝试使用diff库的applyPatch
    const result = diff.applyPatch(originalText, patchText);
    if (result !== false) {
      return { applied: true, result };
    }

    // 如果失败，尝试使用diff-match-patch
    console.log("使用diff.applyPatch失败，尝试使用diff-match-patch");
    const patches = dmp.patch_fromText(patchText);
    const [patchResult, appliedPatches] = dmp.patch_apply(
      patches,
      originalText
    );

    if (appliedPatches.every((applied) => applied)) {
      return { applied: true, result: patchResult };
    }

    console.log("使用diff-match-patch失败，尝试手动应用");
    // 如果上述方法都失败，可以尝试更加定制化的差异应用方法
    // 暂时返回失败
    return { applied: false, error: "无法应用差异" };
  } catch (error) {
    console.error("应用差异时出错:", error);
    return { applied: false, error: error.message };
  }
}

// ==================== 工具集 ====================
const tools = [
  // 优化: 合并代码生成和验证为一个工具
  new DynamicStructuredTool({
    name: "generate_and_validate_code",
    description:
      "基于用户需求和现有代码生成新的Three.js代码，并自动验证其有效性",
    schema: z.object({
      userRequest: z.string().describe("用户的原始请求"),
      existingCode: z.string().describe("现有的Three.js代码，如果没有则为空"),
    }),
    func: async ({ userRequest, existingCode }) => {
      // 分析用户输入
      const analyzedRequest = await userInputAnalysisChain.invoke({
        input: userRequest,
      });

      // 构建输入
      let input = analyzedRequest;
      if (existingCode && existingCode !== "尚无现有代码") {
        input = `在现有场景基础上添加以下内容：${analyzedRequest}\n\n现有代码：\n${existingCode}`;
      }

      // 调用代码生成链
      const generatedCode = await codeGenerationChain.invoke({ input });

      // 保存对话历史到内存
      await memory.saveContext(
        { input: userRequest },
        { output: generatedCode }
      );

      // 生成差异
      let diffText = null;
      if (existingCode && existingCode !== "尚无现有代码") {
        diffText = diff.createPatch(
          "scene.html",
          existingCode || "",
          generatedCode,
          "原始代码",
          "修改后的代码"
        );
      }

      // 验证代码
      const requiredElements = [
        "THREE.Scene",
        "PerspectiveCamera",
        "WebGLRenderer",
        "animate()",
        "requestAnimationFrame",
      ];

      const missing = requiredElements.filter(
        (el) => !generatedCode.includes(el)
      );
      const validationResult =
        missing.length > 0
          ? `缺少关键元素: ${missing.join(", ")}`
          : "代码验证通过";

      // 更新当前代码，供下一步使用
      if (validationResult === "代码验证通过") {
        currentThreeJsCode = generatedCode;
      }

      return JSON.stringify({
        code: generatedCode,
        validationResult: validationResult,
        diffText: diffText,
      });
    },
  }),

  // 添加场景质量评估与优化工具
  new DynamicStructuredTool({
    name: "evaluate_and_improve_scene",
    description: "评估Three.js场景质量并自动改进不满意的部分",
    schema: z.object({
      code: z.string().describe("需要评估和改进的Three.js代码"),
      userRequest: z.string().describe("用户的原始请求，用于理解期望效果"),
    }),
    func: async ({ code, userRequest }) => {
      console.time("场景评估和优化时间");

      // 1. 评估场景质量
      const evaluationPrompt = `请评估以下Three.js场景代码的质量，重点关注:
1. 是否完全满足用户需求: "${userRequest}"
2. 视觉效果是否足够好（光照、材质、颜色等）
3. 场景结构是否合理
4. 代码质量是否高


回答格式:
- 评分(1-10): [分数]
- 主要问题: [问题列表]
- 改进建议: [具体建议]

代码:
${code}`;

      const evaluation = await llm.invoke(evaluationPrompt);
      console.log("场景评估结果:", evaluation.text);

      // 解析评分 (简单提取1-10之间的数字)
      const scoreMatch = evaluation.text.match(/评分.*?(\d+)/);
      const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

      // 2. 如果评分低于阈值，自动改进
      if (score < 5) {
        console.log(`场景评分(${score})不满意，开始自动改进...`);

        const improvementPrompt = `请改进以下Three.js场景代码，使其更好地满足用户需求:"${userRequest}"
根据评估，当前场景存在以下问题:
${evaluation.text}

请对代码进行修改，重点改进以下方面:
1. 满足用户需求的所有功能和视觉元素
2. 提升光照效果、材质质量和整体美观性
3. 使场景符合analyze_user_input分析出来的需求
4. 不要使用本地贴图和gtlf模型
修改后返回完整的、优化过的代码:

${code}`;

        const improvedResult = await llm.invoke(improvementPrompt);
        const improvedCode = improvedResult.text;

        // 生成改进差异
        const improvementDiff = diff.createPatch(
          "scene.html",
          code,
          improvedCode,
          "评估前",
          "评估后"
        );

        // 改进后直接发送到客户端
        await sendCodeToWebsocketDirect(
          improvedCode,
          userRequest,
          improvementDiff
        );

        console.timeEnd("场景评估和优化时间");
        return JSON.stringify({
          originalScore: score,
          evaluation: evaluation.text,
          improvedCode: improvedCode,
          improvementDiff: improvementDiff,
          isImproved: true,
        });
      } else {
        console.log(`场景评分(${score})良好，无需改进`);

        // 直接发送到客户端
        await sendCodeToWebsocketDirect(code, userRequest);

        console.timeEnd("场景评估和优化时间");
        return JSON.stringify({
          originalScore: score,
          evaluation: evaluation.text,
          improvedCode: code,
          isImproved: false,
        });
      }
    },
  }),
];

// ==================== 创建优化后的agent ====================
const optimizedAgentPrompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `你是一个高效的Three.js代码生成和管理Agent，负责完成从需求理解到代码生成和发送的整个流程。你的主要任务包括：

1. 生成并验证代码 - 使用generate_and_validate_code工具一步完成代码生成和验证
2. 评估和改进场景 - 使用evaluate_and_improve_scene工具评估场景质量并自动改进
3. 如果生成的代码验证未通过，请报告问题

如果现有代码可用，确保在其基础上进行修改和添加，而不是创建全新的场景。`,
  ],
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

// 创建优化后的Agent
const optimizedAgent = createToolCallingAgent({
  llm,
  tools,
  prompt: optimizedAgentPrompt,
});

// 创建Agent执行器
const optimizedAgentExecutor = new AgentExecutor({
  agent: optimizedAgent,
  tools,
  verbose: true,
});

// ==================== 新增函数: 直接发送代码到WebSocket，无需LLM调用 ====================
async function sendCodeToWebsocketDirect(code, userInput, diffText = null) {
  try {
    // 保存代码到文件
    const outputPath = path.join(process.cwd(), "output.html");
    fs.writeFileSync(outputPath, code, "utf-8");
    console.log(`代码已保存到: ${outputPath}`);

    // 如果有差异信息，保存到文件
    if (diffText) {
      const diffPath = path.join(process.cwd(), "latest.diff");
      fs.writeFileSync(diffPath, diffText, "utf-8");
      console.log(`差异已保存到: ${diffPath}`);
    }

    // 更新当前代码状态
    currentThreeJsCode = code;

    // 发送代码到所有连接的WebSocket客户端
    const connectedClients = activeConnections.length;
    console.log(`准备向${connectedClients}个客户端发送代码`);

    // 准备差异信息
    const diffInfo = diffText
      ? {
          diffText: diffText,
          userRequest: userInput,
        }
      : null;

    // 不再重复输出差异信息到控制台，统一由interactiveLoop处理

    activeConnections.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        try {
          // 先发送处理完成消息
          client.send(
            JSON.stringify({
              type: "agent_message",
              content: "已处理请求: 成功",
            })
          );

          // 再发送生成的代码
          const codeMessage = JSON.stringify({
            type: "code_generated",
            code: code,
            diffInfo: diffInfo,
            requestId: Date.now().toString(),
          });

          client.send(codeMessage);
          console.log(`已发送代码到客户端`);
        } catch (wsError) {
          console.error("向客户端发送消息时出错:", wsError);
        }
      } else {
        console.log(`客户端连接状态不是OPEN: ${client.readyState}`);
      }
    });

    return {
      status: "success",
      code: code,
      diffInfo: diffInfo,
      message: `代码已发送到客户端并保存`,
    };
  } catch (error) {
    console.error("发送代码到WebSocket出错:", error);
    return {
      status: "error",
      feedback: "发送代码过程中发生错误: " + error.message,
    };
  }
}

// ==================== 修改mainAgent函数逻辑 ====================
async function mainAgent(userInput) {
  try {
    console.log("处理用户请求:", userInput);

    // 获取当前代码
    const existingCode = currentThreeJsCode || "尚无现有代码";

    // 直接发送模式 - 如果用户请求明确指示直接发送现有代码
    if (
      userInput.toLowerCase().includes("直接发送") ||
      userInput.toLowerCase().includes("发送现有代码") ||
      userInput.toLowerCase().includes("无需修改")
    ) {
      if (existingCode === "尚无现有代码") {
        return {
          status: "needs_revision",
          feedback: "尚无现有代码可发送，请先生成代码",
          original_code: "",
        };
      }

      console.log("直接发送现有代码，跳过LLM调用");
      return await sendCodeToWebsocketDirect(existingCode, userInput);
    }

    // 正常LLM生成模式
    const result = await optimizedAgentExecutor.invoke({
      input: `处理用户请求：${userInput}。请严格按照以下流程完成：
1. 首先获取当前代码（使用get_current_code工具）
2. 使用generate_and_validate_code工具生成和验证代码，该工具现在使用BufferMemory保存对话历史
3. 如果代码验证通过，必须使用evaluate_and_improve_scene工具评估场景质量并自动改进
   （evaluate_and_improve_scene工具会负责将最终代码发送到客户端）
4. 如果生成的代码验证未通过，请报告问题
`,
    });

    console.log("Agent执行结果：", result.output);

    // 尝试提取差异信息
    let diffInfo = null;
    if (result.output.includes("diffText")) {
      try {
        // 尝试从输出中提取JSON
        const jsonMatch = result.output.match(/\{[\s\S]*diffText[\s\S]*\}/);
        if (jsonMatch) {
          const jsonStr = jsonMatch[0];
          const parsedData = JSON.parse(jsonStr);
          if (parsedData.diffText) {
            diffInfo = {
              diffText: parsedData.diffText,
              userRequest: userInput,
            };
          }
        }
      } catch (e) {
        console.log("无法从输出中提取差异信息", e);
      }
    }

    // 解析结果
    if (result.output.includes("代码验证通过")) {
      return {
        status: "success",
        code: currentThreeJsCode,
        diffInfo: diffInfo,
        preview: result.output,
      };
    } else {
      return {
        status: "needs_revision",
        feedback: result.output,
        original_code: currentThreeJsCode,
      };
    }
  } catch (error) {
    console.error("Agent执行错误:", error);
    return {
      status: "error",
      feedback: "执行过程中发生错误: " + error.message,
    };
  }
}

// ==================== 交互界面 ====================
async function interactiveLoop() {
  console.log("Three.js 智能生成系统正在运行...");
  console.log("WebSocket服务器监听端口:", process.env.WS_PORT || 3001);

  // 保留命令行接口作为备用
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const ask = () => {
    rl.question("\n请输入物体描述（或输入exit退出）：", async (input) => {
      if (input.toLowerCase() === "exit") {
        rl.close();
        return;
      }

      console.time("代码生成耗时");
      const result = await mainAgent(input);
      console.timeEnd("代码生成耗时");

      if (result.status === "success") {
        console.log("\n代码生成成功！");

        // 优化差异信息显示格式
        if (result.diffInfo && result.diffInfo.diffText) {
          console.log("\n===== 代码差异信息 =====");
          // 提取有意义的部分：文件信息和差异行
          const lines = result.diffInfo.diffText.split("\n");
          const relevantLines = lines.filter(
            (line) =>
              line.startsWith("+++") ||
              line.startsWith("---") ||
              line.startsWith("@@") ||
              line.startsWith("+") ||
              line.startsWith("-")
          );

          // 美化输出
          const formattedLines = relevantLines.map((line) => {
            if (line.startsWith("+") && !line.startsWith("+++")) {
              return `\x1b[32m${line}\x1b[0m`; // 绿色表示添加
            } else if (line.startsWith("-") && !line.startsWith("---")) {
              return `\x1b[31m${line}\x1b[0m`; // 红色表示删除
            } else if (line.startsWith("@@")) {
              return `\x1b[36m${line}\x1b[0m`; // 青色表示行号信息
            } else if (line.startsWith("+++") || line.startsWith("---")) {
              return `\x1b[33m${line}\x1b[0m`; // 黄色表示文件信息
            }
            return line;
          });

          console.log(formattedLines.join("\n") || "差异格式不可读");
          console.log("=======================\n");
        }
      } else {
        console.log("\n需要修正：", result.feedback);
      }

      ask();
    });
  };

  ask();
}

// 启动系统
interactiveLoop();
