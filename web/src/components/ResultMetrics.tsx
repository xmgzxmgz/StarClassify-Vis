function InfoBadge({ children, tip }: { children: React.ReactNode; tip: string }) {
  return (
    <div className="group relative inline-flex items-center gap-1">
      {children}
      <span className="cursor-help text-blue-500 opacity-60 hover:opacity-100">ℹ️</span>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-50 w-64 p-3 text-xs text-white bg-slate-800 rounded-lg shadow-lg">
        {tip}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800"></div>
      </div>
    </div>
  );
}

export default function ResultMetrics(props: {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}) {
  function fmt(v: number) {
    return new Intl.NumberFormat(undefined, {
      maximumFractionDigits: 4,
    }).format(v);
  }

  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <div className="rounded-lg bg-gradient-to-br from-blue-100 via-white to-transparent dark:from-blue-500/15 dark:via-white/5 dark:to-transparent p-3 glow-panel">
        <div className="text-xs text-slate-600 dark:text-white/50">
          <InfoBadge tip="准确率 = 预测正确的样本数 / 总样本数。例如：准确率0.78表示模型正确分类了78%的恒星。适合快速了解模型整体表现，但无法反映类别间的详细差异。">
            Accuracy
          </InfoBadge>
        </div>
        <div className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
          {fmt(props.accuracy)}
        </div>
      </div>
      <div className="rounded-lg bg-gradient-to-br from-emerald-100 via-white to-transparent dark:from-emerald-500/15 dark:via-white/5 dark:to-transparent p-3 glow-panel">
        <div className="text-xs text-slate-600 dark:text-white/50">
          <InfoBadge tip="精确率 = 预测为某类且预测正确的数量 / 预测为该类的总数量。高精确率意味着模型「报假警」的情况较少。">
            Precision
          </InfoBadge>
        </div>
        <div className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
          {fmt(props.precision)}
        </div>
      </div>
      <div className="rounded-lg bg-gradient-to-br from-cyan-100 via-white to-transparent dark:from-cyan-500/15 dark:via-white/5 dark:to-transparent p-3 glow-panel">
        <div className="text-xs text-slate-600 dark:text-white/50">
          <InfoBadge tip="召回率 = 预测为某类且预测正确的数量 / 该类的实际总数量。高召回率意味着模型「漏检」的情况较少。">
            Recall
          </InfoBadge>
        </div>
        <div className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
          {fmt(props.recall)}
        </div>
      </div>
      <div className="rounded-lg bg-gradient-to-br from-purple-100 via-white to-transparent dark:from-purple-500/15 dark:via-white/5 dark:to-transparent p-3 glow-panel">
        <div className="text-xs text-slate-600 dark:text-white/50">
          <InfoBadge tip="F1分数 = 2 × 精确率 × 召回率 / (精确率 + 召回率)。是精确率和召回率的调和平均，平衡考虑了两者，适合评估类别不平衡的数据集。">
            F1
          </InfoBadge>
        </div>
        <div className="mt-1 text-lg font-semibold text-slate-900 dark:text-white">
          {fmt(props.f1)}
        </div>
      </div>
    </div>
  );
}
