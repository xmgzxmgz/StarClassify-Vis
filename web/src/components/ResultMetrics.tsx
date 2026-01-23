/**
 * 渲染核心指标卡片。
 * @param props 组件输入参数。
 * @returns 指标卡片组件。
 */
export default function ResultMetrics(props: {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}) {
  /**
   * 格式化指标数值。
   * @param v 指标数值。
   * @returns 格式化后的字符串。
   */
  function fmt(v: number) {
    return new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 4,
    }).format(v);
  }

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <div className="rounded-lg bg-gradient-to-br from-blue-500/15 via-white/5 to-transparent p-3 glow-panel">
        <div className="text-xs text-white/50">Accuracy</div>
        <div className="mt-1 text-lg font-semibold text-white">
          {fmt(props.accuracy)}
        </div>
      </div>
      <div className="rounded-lg bg-gradient-to-br from-emerald-500/15 via-white/5 to-transparent p-3 glow-panel">
        <div className="text-xs text-white/50">Precision</div>
        <div className="mt-1 text-lg font-semibold text-white">
          {fmt(props.precision)}
        </div>
      </div>
      <div className="rounded-lg bg-gradient-to-br from-cyan-500/15 via-white/5 to-transparent p-3 glow-panel">
        <div className="text-xs text-white/50">Recall</div>
        <div className="mt-1 text-lg font-semibold text-white">
          {fmt(props.recall)}
        </div>
      </div>
      <div className="rounded-lg bg-gradient-to-br from-purple-500/15 via-white/5 to-transparent p-3 glow-panel">
        <div className="text-xs text-white/50">F1</div>
        <div className="mt-1 text-lg font-semibold text-white">
          {fmt(props.f1)}
        </div>
      </div>
    </div>
  );
}
