/**
 * 渲染类别分布条形图。
 * @param props 组件输入参数。
 * @returns 分布条形图组件。
 */
export default function DistributionBars(props: {
  title: string;
  items: { label: string; value: number }[];
  tone?: "blue" | "purple";
}) {
  const max = Math.max(1, ...props.items.map((i) => i.value));
  const color =
    props.tone === "purple"
      ? "from-purple-500/80 to-fuchsia-500/70"
      : "from-blue-500/80 to-cyan-400/70";

  return (
    <div className="space-y-3">
      <div className="text-sm font-semibold text-slate-900 dark:text-white">{props.title}</div>
      <div className="space-y-2">
        {props.items.map((item) => (
          <div key={item.label} className="space-y-1">
            <div className="flex items-center justify-between text-xs text-slate-600 dark:text-white/60">
              <span>{item.label}</span>
              <span>{item.value}</span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-200 dark:bg-white/10">
              <div
                className={`h-2 rounded-full bg-gradient-to-r ${color} shimmer`}
                style={{ width: `${Math.round((item.value / max) * 100)}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
