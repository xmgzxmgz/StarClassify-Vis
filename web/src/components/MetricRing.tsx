/**
 * 渲染环形指标可视化。
 * @param props 组件输入参数。
 * @returns 指标环形图组件。
 */
export default function MetricRing(props: {
  value: number;
  label: string;
  tone?: "blue" | "green" | "amber";
}) {
  const pct = Math.max(0, Math.min(1, props.value));
  const angle = Math.round(pct * 360);
  const color =
    props.tone === "green"
      ? "#22C55E"
      : props.tone === "amber"
        ? "#F59E0B"
        : "#3B82F6";

  return (
    <div className="flex flex-col items-center gap-3">
      <div
        className="relative flex h-32 w-32 items-center justify-center rounded-full ring-2 ring-slate-300 dark:ring-white/10 ring-glow"
        style={{
          background: `conic-gradient(${color} ${angle}deg, rgba(255,255,255,0.08) 0deg)`,
        }}
      >
        <div className="absolute inset-2 rounded-full bg-white dark:bg-[#0B1220]"></div>
        <div className="relative text-center">
          <div className="text-xl font-semibold text-slate-900 dark:text-white">
            {Math.round(pct * 100)}%
          </div>
          <div className="text-xs text-slate-600 dark:text-white/60">{props.label}</div>
        </div>
      </div>
    </div>
  );
}
