
'use client';

import React, { useState, useRef, useEffect } from 'react';
import styles from "./page.module.css";

// Define message type
type Message = {
  id: string;
  content: string;
  sender: 'user' | 'ai';
  timestamp: Date;
};

export default function Home() {
  // State for chat messages
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: '欢迎使用3D AI助手！请描述您想要创建的3D场景。',
      sender: 'ai',
      timestamp: new Date(),
    },
  ]);
  
  // State for input message
  const [inputMessage, setInputMessage] = useState('');
  
  // State for code display
  const [code, setCode] = useState('// 生成的Three.js代码将显示在这里');
  
  // State to track if code has been executed
  const [codeExecuted, setCodeExecuted] = useState(false);
  
  // State to store HTML content for iframe
  const [iframeContent, setIframeContent] = useState('');
  
  // Ref for the render container
  const renderContainerRef = useRef<HTMLDivElement>(null);
  
  // Ref for chat messages container to auto-scroll
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // WebSocket connection
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Initialize WebSocket connection
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:${process.env.NEXT_PUBLIC_WS_PORT || 3001}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received message:', data);
        
        // Handle different message types from server
        if (data.type === 'ai_response') {
          addMessage(data.content, 'ai');
        } else if (data.type === 'code_generated') {
          setCode(data.code || '// 没有代码生成');
          addMessage('我已经生成了Three.js代码，请查看代码区域。', 'ai');
          setCodeExecuted(false); // Reset code execution state
        }
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };
    
    setSocket(ws);
    
    // Cleanup on unmount
    return () => {
      ws.close();
    };
  }, []);
  
  // Add a new message to the chat
  const addMessage = (content: string, sender: 'user' | 'ai') => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      sender,
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, newMessage]);
  };
  
  // Handle sending a message
  const handleSendMessage = () => {
    if (!inputMessage.trim() || !socket || socket.readyState !== WebSocket.OPEN) return;
    
    // Add user message to chat
    addMessage(inputMessage, 'user');
    
    // Send message to server
    socket.send(JSON.stringify({
      type: 'user_prompt',
      content: inputMessage,
    }));
    
    // Clear input field
    setInputMessage('');
  };
  
  // Handle input keypress (send on Enter)
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };
  
  // Function to execute the generated Three.js code
  const executeCode = () => {
    if (!code) return;
    
    try {
      // 设置iframe内容
      setIframeContent(code);
      setCodeExecuted(true);
      addMessage('Three.js代码已成功执行，请查看右侧渲染区域。', 'ai');
    } catch (error) {
      console.error('Error executing code:', error);
      addMessage(`执行代码时出错: ${error instanceof Error ? error.message : String(error)}`, 'ai');
    }
  };

  return (
    <div className={styles.page}>
      {/* Left Panel - Chat and Code */}
      <div className={styles.leftPanel}>
        {/* Chat Container */}
        <div className={styles.chatContainer}>
          <div className={styles.chatMessages}>
            {messages.map((message) => (
              <div 
                key={message.id} 
                className={message.sender === 'user' ? styles.messageUser : styles.messageAI}
              >
                {message.content}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className={styles.chatInput}>
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder="描述您想要的3D场景..."
              disabled={!isConnected}
            />
            <button 
              onClick={handleSendMessage}
              disabled={!isConnected}
            >
              发送
            </button>
          </div>
        </div>
        
        {/* Code Container */}
        <div className={styles.codeContainer}>
          <div className={styles.codeHeader}>
            <span>生成的代码</span>
            <div>
              <button 
                onClick={executeCode}
                disabled={!code || code === '// 生成的Three.js代码将显示在这里'}
                style={{ marginRight: '0.5rem' }}
              >
                执行代码
              </button>
              <button onClick={() => {
                // Copy code to clipboard
                navigator.clipboard.writeText(code);
              }}>
                复制代码
              </button>
            </div>
          </div>
          <pre className={styles.codeContent}>
            {code}
          </pre>
        </div>
      </div>
      
      {/* Right Panel - 3D Rendering */}
      <div className={styles.renderContainer}>
        <div className={styles.renderHeader}>
          <h3>3D渲染区域</h3>
          <button 
            className={styles.executeButton}
            onClick={executeCode}
            disabled={!isConnected || !code || code === '// 生成的Three.js代码将显示在这里'}
          >
            执行代码
          </button>
        </div>
        <div className={styles.renderContent} id="scene-container" ref={renderContainerRef}>
          {!isConnected && (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              color: 'var(--foreground)',
            }}>
              正在连接到服务器...
            </div>
          )}
          {isConnected && !codeExecuted && (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              height: '100%',
              color: 'var(--foreground)',
            }}>
              生成代码后，点击"执行代码"按钮来渲染3D场景
            </div>
          )}
          {codeExecuted && (
            <iframe
              style={{
                width: '100%',
                height: '100%',
                border: 'none',
              }}
              sandbox="allow-scripts allow-same-origin"
              title="Three.js Rendering"
              srcDoc={iframeContent}
            />
          )}
        </div>
      </div>
    </div>
  );
}
