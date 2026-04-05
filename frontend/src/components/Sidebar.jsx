import { useRef, useState } from "react"

export default function Sidebar({ status, onUpload, uploading }) {
  const fileRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = (file) => {
    if (!file) return
    const ext = file.name.split(".").pop().toLowerCase()
    // FIX 4: Thiếu "json" trong danh sách cho phép
    // → user upload train.json bị chặn ngay ở frontend dù server hỗ trợ .json
    if (!["pdf", "txt", "json"].includes(ext)) {
      alert("Chỉ hỗ trợ file .pdf, .txt hoặc .json")
      return
    }
    onUpload(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo">
          <div className="logo-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <div>
            <div className="logo-name">RAG Chatbot</div>
            <div className="logo-sub">NLP Transformer</div>
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="status-card">
        <div className="status-row">
          <span className={`status-dot ${status.ready ? "online" : "offline"}`} />
          <span className="status-label">{status.ready ? "Sẵn sàng" : "Chưa có tài liệu"}</span>
        </div>
        {status.ready && (
          <div className="status-stats">
            <div className="stat">
              <span className="stat-val">{status.num_chunks}</span>
              <span className="stat-key">chunks</span>
            </div>
            <div className="stat">
              <span className="stat-val">{status.documents?.length || 0}</span>
              <span className="stat-key">tài liệu</span>
            </div>
          </div>
        )}
      </div>

      {/* Upload */}
      <div className="upload-section">
        <div className="section-title">Tải tài liệu lên</div>
        <div
          className={`drop-zone ${dragOver ? "drag-over" : ""} ${uploading ? "uploading" : ""}`}
          onClick={() => !uploading && fileRef.current.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          {/* FIX 4 (tiếp): accept input cũng phải có .json */}
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.txt,.json"
            style={{ display: "none" }}
            onChange={e => handleFile(e.target.files[0])}
          />
          {uploading ? (
            <>
              <div className="upload-spinner" />
              <span>Đang xử lý...</span>
            </>
          ) : (
            <>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
              <span>Click hoặc kéo thả file</span>
              {/* FIX 4 (tiếp): hint text cũng cập nhật */}
              <span className="drop-hint">PDF, TXT, JSON • Tối đa 10MB</span>
            </>
          )}
        </div>
      </div>

      {/* Document list */}
      {status.documents && status.documents.length > 0 && (
        <div className="doc-list-section">
          <div className="section-title">Tài liệu đã index</div>
          <div className="doc-list">
            {status.documents.map((doc, i) => (
              <div key={i} className="doc-item">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <span>{doc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Architecture info */}
      <div className="arch-section">
        <div className="section-title">Kiến trúc</div>
        <div className="arch-list">
          {[
            { label: "Encoder", val: "Sentence-BERT" },
            { label: "Vector DB", val: "FAISS" },
            { label: "Generator", val: "ViT5" },
            { label: "Server", val: "FastAPI" },
          ].map(({ label, val }) => (
            <div key={label} className="arch-item">
              <span className="arch-label">{label}</span>
              <span className="arch-val">{val}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}