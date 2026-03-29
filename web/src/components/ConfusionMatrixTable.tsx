export default function ConfusionMatrixTable(props: {
  labels: string[];
  matrix: number[][];
}) {
  return (
    <div className="overflow-auto rounded-lg border border-slate-300 dark:border-white/10">
      <table className="min-w-full border-separate border-spacing-0 text-sm">
        <thead>
          <tr className="bg-slate-100 dark:bg-white/5">
            <th className="sticky left-0 z-10 w-32 border-b border-slate-300 dark:border-white/10 bg-slate-100 dark:bg-white/5 px-3 py-2 text-left font-medium text-slate-700 dark:text-white/80">
              混淆矩阵
            </th>
            {props.labels.map((l) => (
              <th
                key={l}
                className="border-b border-slate-300 dark:border-white/10 px-3 py-2 text-left font-medium text-slate-700 dark:text-white/80"
              >
                预测:{l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.matrix.map((row, i) => (
            <tr key={props.labels[i] ?? String(i)} className="hover:bg-slate-50 dark:hover:bg-white/5">
              <td className="sticky left-0 z-10 border-b border-slate-300 dark:border-white/5 bg-white dark:bg-[#111A2E] px-3 py-2 font-medium text-slate-700 dark:text-white/80">
                实际:{props.labels[i] ?? "?"}
              </td>
              {row.map((v, j) => (
                <td
                  key={j}
                  className="border-b border-slate-300 dark:border-white/5 px-3 py-2 text-slate-600 dark:text-white/70"
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
