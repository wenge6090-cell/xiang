interface StatusBarProps {
  deviation: number
  shaCount: number
  cycleCount: number
  phase: string
}

export function StatusBar({ deviation, shaCount, cycleCount, phase }: StatusBarProps) {
  const barColor = deviation > 0.7 ? "bg-red-500" : deviation > 0.4 ? "bg-amber-500" : "bg-green-500"

  return (
    <div className="flex items-center gap-3 text-[11px]">
      {/* Deviation bar */}
      <div className="flex items-center gap-1.5">
        <span className="text-muted-foreground">偏离</span>
        <div className="w-16 h-1.5 rounded-full bg-secondary overflow-hidden">
          <div
            className={cn("h-full rounded-full transition-all duration-300", barColor)}
            style={{ width: `${Math.min(deviation * 100, 100)}%` }}
          />
        </div>
        <span className="font-mono text-muted-foreground w-8 text-right">
          {(deviation * 100).toFixed(0)}%
        </span>
      </div>
      <span className="text-muted-foreground/30">|</span>
      <span className="text-muted-foreground">
        卦 {phase}
      </span>
      <span className="text-muted-foreground/30">|</span>
      <span className="text-muted-foreground">
        剪 {shaCount}
      </span>
      <span className="text-muted-foreground/30">|</span>
      <span className="text-muted-foreground">
        周 {cycleCount}
      </span>
    </div>
  )
}

function cn(...classes: (string | false | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ")
}
