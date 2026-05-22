import { useState, useRef, useEffect } from "react"
import { ChatPanel } from "./components/ChatPanel"
import { StatusBar } from "./components/StatusBar"

export interface Message {
  role: "user" | "assistant"
  content: string
}

export interface ConstrainedMeta {
  deviation: number
  intervened: boolean
  phase: string
  tokens: number
}

function App() {
  const [rawMessages, setRawMessages] = useState<Message[]>([])
  const [constrainedMessages, setConstrainedMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [rawLoading, setRawLoading] = useState(false)
  const [constrainedLoading, setConstrainedLoading] = useState(false)
  const [constrainedMeta, setConstrainedMeta] = useState<ConstrainedMeta | null>(null)
  const [statusInfo, setStatusInfo] = useState({ deviation: 0, shaCount: 0, cycleCount: 0, phase: "生" })
  const [showWelcome, setShowWelcome] = useState(true)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const sendingRef = useRef(false)

  useEffect(() => {
    fetchState()
    inputRef.current?.focus()
  }, [])

  const fetchState = async () => {
    try {
      const res = await fetch("/api/state")
      const data = await res.json()
      // new format: { raw: [...], constrained: [...], constrained_state: {...} }
      setRawMessages(data.raw || data.raw_history || [])
      setConstrainedMessages(data.constrained || data.constrained_history || [])
      const cs = data.constrained_state || data
      setStatusInfo({
        deviation: cs.deviation || 0,
        shaCount: cs.sha_count || 0,
        cycleCount: cs.cycle_count || 0,
        phase: cs.phase || "生",
      })
    } catch {}
  }

  const sendMessage = async () => {
    const msg = input.trim()
    if (!msg || sendingRef.current) return
    sendingRef.current = true

    setInput("")
    setShowWelcome(false)
    setConstrainedMeta(null)

    const userMsg: Message = { role: "user", content: msg }
    setRawMessages(prev => [...prev, userMsg])
    setConstrainedMessages(prev => [...prev, userMsg])

    try {
      // Step 1: raw (unconstrained) first
      setRawLoading(true)
      const rawRes = await fetch("/api/raw", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      })
      const rawData = await rawRes.json()
      const rawAssistant: Message = { role: "assistant", content: rawData.response || "(empty)" }
      setRawMessages(prev => [...prev, rawAssistant])
      setRawLoading(false)

      // Step 2: constrained second (alternating, no parallel)
      setConstrainedLoading(true)
      const conRes = await fetch("/api/constrained", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      })
      const conData = await conRes.json()
      const conAssistant: Message = { role: "assistant", content: conData.response || "(empty)" }
      setConstrainedMessages(prev => [...prev, conAssistant])
      setConstrainedMeta({
        deviation: conData.deviation || 0,
        intervened: conData.intervened || false,
        phase: conData.phase || "生",
        tokens: conData.tokens || 0,
      })
      setStatusInfo(prev => ({
        ...prev,
        deviation: conData.deviation || prev.deviation,
        phase: conData.phase || prev.phase,
      }))
    } catch {
      if (rawLoading) {
        setRawMessages(prev => [...prev, { role: "assistant", content: "[连接失败]" }])
      }
      if (constrainedLoading) {
        setConstrainedMessages(prev => [...prev, { role: "assistant", content: "[连接失败]" }])
      }
    } finally {
      setRawLoading(false)
      setConstrainedLoading(false)
      sendingRef.current = false
    }
  }

  const handleReset = async () => {
    await fetch("/api/reset")
    setRawMessages([])
    setConstrainedMessages([])
    setShowWelcome(true)
    setStatusInfo({ deviation: 0, shaCount: 0, cycleCount: 0, phase: "生" })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const suggestedPrompts = [
    "什么是机器学习？",
    "解释一下意识的本质",
    "请写一段关于AI安全的论述",
    "量子计算如何改变世界？",
  ]

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">
      {/* Header */}
      <header className="flex-none border-b border-border bg-card px-4 py-3">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-xl">⚖</span>
            <div>
              <h1 className="text-sm font-semibold text-foreground">象语言 思维引导引擎</h1>
              <p className="text-[11px] text-muted-foreground">Qwen3.5-4B · AMD RX 6650 XT · Vulkan</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusBar
              deviation={statusInfo.deviation}
              shaCount={statusInfo.shaCount}
              cycleCount={statusInfo.cycleCount}
              phase={statusInfo.phase}
            />
            <button
              onClick={handleReset}
              className="text-xs px-3 py-1.5 rounded-md bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors"
            >
              重置
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex gap-0 overflow-hidden max-w-[1400px] mx-auto w-full">
        {/* Raw Panel */}
        <ChatPanel
          title="原生模型"
          subtitle="无约束 · 自由生成"
          accent="raw"
          messages={rawMessages}
          loading={rawLoading}
          constrainedMeta={null}
          showWelcome={showWelcome}
          suggestedPrompts={suggestedPrompts}
          onPromptClick={(p) => { setInput(p); setTimeout(sendMessage, 100) }}
        />

        {/* Divider */}
        <div className="w-px bg-border flex-none" />

        {/* Constrained Panel */}
        <ChatPanel
          title="约束引擎"
          subtitle="CangVM · Logit Bias"
          accent="constrained"
          messages={constrainedMessages}
          loading={constrainedLoading}
          constrainedMeta={constrainedMeta}
          showWelcome={showWelcome}
          suggestedPrompts={suggestedPrompts}
          onPromptClick={(p) => { setInput(p); setTimeout(sendMessage, 100) }}
        />
      </main>

      {/* Input */}
      <footer className="flex-none border-t border-border bg-card px-4 py-3">
        <div className="max-w-[1400px] mx-auto flex gap-3 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入消息，回车发送..."
            rows={1}
            className="flex-1 bg-secondary text-foreground rounded-lg px-4 py-2.5 text-sm resize-none outline-none focus:ring-1 focus:ring-ring placeholder:text-muted-foreground/50 scrollbar-thin"
            style={{ minHeight: 40, maxHeight: 120 }}
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = "auto"
              el.style.height = Math.min(el.scrollHeight, 120) + "px"
            }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || sendingRef.current}
            className="px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            发送
          </button>
        </div>
      </footer>
    </div>
  )
}

export default App
