function InfoBadge({ children, tip }: { children: React.ReactNode; tip: string }) {
  return (
    <div className="group relative inline-flex items-center gap-1">
      {children}
      <span className="cursor-help text-blue-500 opacity-60 hover:opacity-100">ℹ️</span>
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-50 w-72 p-3 text-xs text-white bg-slate-800 rounded-lg shadow-lg">
        {tip}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800"></div>
      </div>
    </div>
  );
}

export default function ConfusionMatrixTable(props: {
  labels: string[];
  matrix: number[][];
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-white/80">
        <InfoBadge tip="混淆矩阵是评估分类模型的重要工具。矩阵的每一行代表模型预测的结果，每一列代表真实的类别。对角线上的数字越大越好，表示预测正确的数量。">
          🧩 混淆矩阵
        </InfoBadge>
      </div>
      <div className="overflow-auto rounded-lg border border-slate-300 dark:border-white/10">
        <table className="min-w-full border-separate border-spacing-0 text-sm">
          <thead>
            <tr className="bg-slate-100 dark:bg-white/5">
              <th className="sticky left-0 z-10 w-32 border-b border-slate-300 dark:border-white/10 bg-slate-100 dark:bg-white/5 px-3 py-2 text-left font-medium text-slate-700 dark:text-white/80">
                <InfoBadge tip="矩阵的左侧列显示每种恒星类型的真实数量。">
                  实际类别 ↓
                </InfoBadge>
              </th>
              {props.labels.map((l) => (
                <th
                  key={l}
                  className="border-b border-slate-300 dark:border-white/10 px-3 py-2 text-left font-medium text-slate-700 dark:text-white/80"
                >
                  <InfoBadge tip={`模型预测为「${l}」的恒星数量。`}>
                    预测:{l}
                  </InfoBadge>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {props.matrix.map((row, i) => (
              <tr key={props.labels[i] ?? String(i)} className="hover:bg-slate-50 dark:hover:bg-white/5">
                <td className="sticky left-0 z-10 border-b border-slate-300 dark:border-white/5 bg-white dark:bg-[#111A2E] px-3 py-2 font-medium text-slate-700 dark:text-white/80">
                  <InfoBadge tip={`测试集中实际为「${props.labels[i]}」类型的恒星总数（包含正确预测和错误预测）。`}>
                    实际:{props.labels[i] ?? "?"}
                  </InfoBadge>
                </td>
                {row.map((v, j) => {
                  const isCorrect = i === j;
                  return (
                    <td
                      key={j}
                      className={`border-b border-slate-300 dark:border-white/5 px-3 py-2 text-slate-600 dark:text-white/70 ${
                        isCorrect
                          ? "bg-green-50 dark:bg-green-500/10 font-semibold text-green-600 dark:text-green-400"
                          : "bg-red-50/50 dark:bg-red-500/5"
                      }`}
                    >
                      <InfoBadge
                        tip={
                          isCorrect
                            ? `✅ 正确预测！${v}颗恒星被正确分类为「${props.labels[i]}」。`
                            : `❌ 错误预测！${v}颗恒星实际是「${props.labels[i]}」但被错误预测为「${props.labels[j]}」。`
                        }
                      >
                        {v}
                      </InfoBadge>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-slate-500 dark:text-white/50 italic">
        <InfoBadge tip="💡 解读技巧：对角线（绿色）上的数字越大越好，表示正确预测越多。非对角线（红色）的数字越小越好，表示错误预测越少。">
          💡 绿色单元格表示正确分类，数字越大越好；红色单元格表示错误分类，数字越小越好
        </InfoBadge>
      </div>
    </div>
  );
}
