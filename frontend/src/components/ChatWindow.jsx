import { useState, useRef, useEffect } from "react"

export default function ChatWindow({ messages, onSend, loading, ready }) {
  const [input, setInput] = useState("")
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, loading])

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (!input.trim() || loading || !ready) return
    onSend(input.trim())
    setInput("")
    textareaRef.current.style.height = "auto"
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const autoResize = (e) => {
    e.target.style.height = "auto"
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px"
  }

  return (
    <main className="chat-main">
      <div className="chat-messages">
        {messages.map(msg => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}

        {loading && (
          <div className="bubble-row assistant">
            {/* FIX 1: Thiếu avatar → bubble lệch trái so với tin nhắn assistant khác */}
            <div className="avatar assistant-avatar">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/>
              </svg>
            </div>
            <div className="bubble assistant typing-bubble">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-wrapper"><form className="chat-input-bar" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="chat-input"
          placeholder={ready ? "Nhập câu hỏi... (Enter để gửi)" : "Upload tài liệu để bắt đầu..."}
          value={input}
          onChange={e => { setInput(e.target.value); autoResize(e) }}
          onKeyDown={handleKeyDown}
          disabled={!ready || loading}
          rows={1}
        />
        <button
          type="submit"
          className={`send-btn ${(!input.trim() || !ready || loading) ? "disabled" : ""}`}
          disabled={!input.trim() || !ready || loading}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form></div>
    </main>
  )
}

function MessageBubble({ msg }) {
  const [expanded, setExpanded] = useState(false)
  const isUser = msg.role === "user"

  // FIX 2: renderText cũ chỉ xử lý **bold**, bỏ qua \n
  // → câu trả lời dài từ T5 hiện thành 1 khối chữ không xuống dòng
  const renderText = (text) => {
    return text.split("\n").map((line, lineIdx) => (
      <span key={lineIdx}>
        {lineIdx > 0 && <br />}
        {line.split(/(\*\*.*?\*\*)/).map((part, i) =>
          part.startsWith("**") && part.endsWith("**")
            ? <strong key={i}>{part.slice(2, -2)}</strong>
            : part
        )}
      </span>
    ))
  }

  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      {!isUser && (
        <div className="avatar assistant-avatar">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/>
          </svg>
        </div>
      )}

      <div className="bubble-content">
        <div className={`bubble ${isUser ? "user" : msg.error ? "error" : msg.isSystem ? "system" : "assistant"}`}>
          <p>{renderText(msg.text)}</p>
        </div>

        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="sources-section">
            {/* FIX 3: Thiếu type="button" → click toggle nguồn tham khảo trigger form submit
                → trang reload, mất toàn bộ chat history */}
            <button
              type="button"
              className="sources-toggle"
              onClick={() => setExpanded(!expanded)}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
              </svg>
              {msg.sources.length} nguồn tham khảo
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                style={{ transform: expanded ? "rotate(180deg)" : "none", transition: "transform .2s" }}>
                <polyline points="6 9 12 15 18 9"/>
              </svg>
            </button>

            {expanded && (
              <div className="sources-list">
                {msg.sources.map((s, i) => (
                  <div key={i} className="source-item">
                    <div className="source-header">
                      <span className="source-file">{s.source}</span>
                      <span className="source-score">{(s.score * 100).toFixed(0)}% match</span>
                    </div>
                    <p className="source-text">{s.text}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {isUser && (
        <div className="avatar user-avatar">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
          </svg>
        </div>
      )}
    </div>
  )
}