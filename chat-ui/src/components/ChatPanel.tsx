import { useRef, useEffect } from "react"
import type { Message, ConstrainedMeta } from "../App"
import { cn } from "../lib/utils"

interface ChatPanelProps {
  title: string
  subtitle: string
  accent: "raw" | "constrained"
  messages: Message[]
  loading: boolean
  constrainedMeta: ConstrainedMeta | null
  showWelcome: boolean
  suggestedPrompts: string[]
  onPromptClick: (prompt: string) => void
}

export function ChatPanel({
  title,
  subtitle,
  accent,
  messages,
  loading,
  constrainedMeta,
  showWelcome,
  suggestedPrompts,
  onPromptClick,
}: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, loading])

  const accentColor = accent === "raw"
    ? "border-raw-accent/30 bg-raw-accent/5"
    : "border-constrained-accent/30 bg-constrained-accent/5"

  const headerBorder = accent === "raw"
    ? "border-b-raw-accent/20"
    : "border-b-constrained-accent/20"

  const badgeBg = accent === "raw"
    ? "bg-raw-accent/15 text-red-400"
    : "bg-constrained-accent/15 text-green-400"

  return (
    <div className="flex-1 flex flex-col min-w-0">
      {/* Panel Header */}
      <div className={cn("flex-none px-4 py-2.5 border-b flex items-center justify-between", headerBorder)}>
        <div>
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          <p className="text-[11px] text-muted-foreground">{subtitle}</p>
        </div>
        {accent === "constrained" && constrainedMeta && (
          <div className="flex items-center gap-2">
            {constrainedMeta.intervened && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-medium">
                干预
              </span>
            )}
            <span className="text-[11px] text-muted-foreground font-mono">
              δ={constrainedMeta.deviation.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin"
      >
        {showWelcome ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-6">
            <div className={cn("w-12 h-12 rounded-full flex items-center justify-center text-lg mb-4", accent === "raw" ? "bg-red-500/10" : "bg-green-500/10")}>
              {accent === "raw" ? "🔴" : "🟢"}
            </div>
            <h3 className="text-sm font-medium text-foreground mb-1">
              {title} — 开始对话
            </h3>
            <p className="text-xs text-muted-foreground mb-6 max-w-xs">
              {accent === "raw"
                ? "直接调用模型，无任何约束干预"
                : "经 CangVM + Logit Bias 约束引擎处理"}
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {suggestedPrompts.map((p, i) => (
                <button
                  key={i}
                  onClick={() => onPromptClick(p)}
                  className="text-xs px-3 py-1.5 rounded-full border border-border bg-card hover:bg-secondary transition-colors text-muted-foreground hover:text-foreground"
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => (
              <div key={i} className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}>
                <div
                  className={cn(
                    "max-w-[85%] rounded-lg px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap",
                    msg.role === "user"
                      ? "bg-secondary text-secondary-foreground"
                      : cn("border", accentColor)
                  )}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            {loading && (
              <div className="flex justify-start">
                <div className="max-w-[85%] rounded-lg px-4 py-3 border border-border bg-card">
                  <div className="flex gap-1.5">
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                    <span className="typing-dot" />
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Status Bar */}
      {!showWelcome && messages.length > 0 && (
        <div className={cn("flex-none px-4 py-1.5 border-t border-border flex items-center gap-3", accent === "constrained" ? "justify-between" : "justify-end")}>
          {accent === "constrained" && (
            <>
              <span className={cn("text-[10px] px-1.5 py-0.5 rounded font-mono", badgeBg)}>
                卦 {statusGua(constrainedMeta?.deviation ?? 0)}
              </span>
              {constrainedMeta && (
                <span className="text-[10px] text-muted-foreground font-mono">
                  {constrainedMeta.tokens} tok
                </span>
              )}
            </>
          )}
          <span className="text-[10px] text-muted-foreground">
            {messages.length / 2} 轮
          </span>
        </div>
      )}
    </div>
  )
}

function statusGua(deviation: number): string {
  if (deviation > 0.9) return "䷀ 乾"
  if (deviation > 0.7) return "䷛ 大过"
  if (deviation > 0.5) return "䷺ 涣"
  if (deviation > 0.3) return "䷿ 未济"
  if (deviation > 0.1) return "䷊ 泰"
  return "䷗ 复"
}
