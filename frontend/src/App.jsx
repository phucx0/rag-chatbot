import { useState, useEffect } from "react"
import ChatWindow from "./components/ChatWindow"
import Sidebar from "./components/Sidebar"
import "./index.css"

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000"

export default function App() {
  const [status, setStatus] = useState({ ready: false, num_chunks: 0, documents: [] })
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: "assistant",
      text: "Xin chào! Tôi là RAG Chatbot. Hãy upload tài liệu PDF hoặc TXT để bắt đầu hỏi đáp.",
      sources: [],
    },
  ])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)

  // Poll status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`)
        const data = await res.json()
        setStatus(data)
      } catch {
        setStatus(prev => ({ ...prev, ready: false }))
      }
    }
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const sendMessage = async (question) => {
    if (!question.trim() || loading) return

    const userMsg = { id: Date.now(), role: "user", text: question, sources: [] }
    setMessages(prev => [...prev, userMsg])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 4 }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || "Lỗi server")
      }

      const data = await res.json()
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: data.answer,
          sources: data.sources || [],
        },
      ])
    } catch (e) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: `Lỗi: ${e.message}`,
          sources: [],
          error: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const uploadFile = async (file) => {
    setUploading(true)
    const form = new FormData()
    form.append("file", file)

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.detail)

      setMessages(prev => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          text: `Đã upload thành công: **${file.name}** (${data.num_chunks} chunks). Bạn có thể hỏi về nội dung tài liệu này rồi!`,
          sources: [],
          isSystem: true,
        },
      ])

      // Refresh status
      const s = await fetch(`${API_BASE}/status`)
      setStatus(await s.json())
    } catch (e) {
      setMessages(prev => [
        ...prev,
        {
          id: Date.now(),
          role: "assistant",
          text: `Upload thất bại: ${e.message}`,
          sources: [],
          error: true,
        },
      ])
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="app-layout">
      <Sidebar
        status={status}
        onUpload={uploadFile}
        uploading={uploading}
        apiBase={API_BASE}
      />
      <ChatWindow
        messages={messages}
        onSend={sendMessage}
        loading={loading}
        ready={status.ready}
      />
    </div>
  )
}
