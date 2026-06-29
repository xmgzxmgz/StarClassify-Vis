import Card from "@/components/Card";
import EmptyState from "@/components/EmptyState";
import ErrorBanner from "@/components/ErrorBanner";
import ConfusionMatrixTable from "@/components/ConfusionMatrixTable";
import type { RunResult } from "@/types";

const STAR_CLASS_INFO: Record<string, { name: string; color: string; bgColor: string; description: string; examples: string }> = {
  "COOL_DWARF": {
    name: "冷矮星",
    color: "text-red-400",
    bgColor: "bg-red-100 dark:bg-red-500/20",
    description: "温度较低的小型恒星，表面温度约3500-5000K",
    examples: "如红矮星、橙矮星，银河系中最多的一类恒星"
  },
  "HOT_MAIN_SEQUENCE": {
    name: "热主序星",
    color: "text-blue-400",
    bgColor: "bg-blue-100 dark:bg-blue-500/20",
    description: "温度较高的年轻恒星，表面温度超过6000K",
    examples: "如A型、F型恒星，呈蓝白色，寿命较短"
  },
  "OTHER": {
    name: "其他类型",
    color: "text-gray-500",
    bgColor: "bg-gray-100 dark:bg-gray-500/20",
    description: "不符合标准MK分类的特殊天体",
    examples: "包括特殊恒星、双星系统等特殊情况"
  },
  "RED_GIANT": {
    name: "红巨星",
    color: "text-orange-400",
    bgColor: "bg-orange-100 dark:bg-orange-500/20",
    description: "已经离开主序阶段的膨胀恒星，体积巨大但温度较低",
    examples: "如心宿二、毕宿五，表面温度约3500-5000K但光度极高"
  },
  "SOLAR_TYPE": {
    name: "类太阳恒星",
    color: "text-yellow-400",
    bgColor: "bg-yellow-100 dark:bg-yellow-500/20",
    description: "与太阳相似的恒星，表面温度约5000-6000K",
    examples: "如织女星、天狼星，质量约0.8-1.2倍太阳质量"
  },
  "HOT_GIANT": {
    name: "热巨星",
    color: "text-cyan-400",
    bgColor: "bg-cyan-100 dark:bg-cyan-500/20",
    description: "温度极高的巨大恒星，表面温度超过7000K",
    examples: "如蓝巨星、猎户座参宿七，体积可达太阳数十倍"
  },
  "VERY_COOL": {
    name: "极冷恒星",
    color: "text-purple-400",
    bgColor: "bg-purple-100 dark:bg-purple-500/20",
    description: "温度极低的恒星或准恒星天体",
    examples: "如M型矮星、褐矮星候选体，表面温度低于3500K"
  },
  "UNKNOWN": {
    name: "未知类型",
    color: "text-slate-500",
    bgColor: "bg-slate-100 dark:bg-slate-500/20",
    description: "参数缺失或无法分类的恒星",
    examples: "需要更多观测数据才能确定其类型"
  }
};

function StarClassBadge({ englishClass, count }: { englishClass: string; count: number }) {
  const info = STAR_CLASS_INFO[englishClass] || {
    name: englishClass,
    color: "text-slate-500",
    bgColor: "bg-slate-100 dark:bg-slate-500/20",
    description: "",
    examples: ""
  };

  return (
    <div className={`rounded-lg p-3 ${info.bgColor} transition-all hover:scale-[1.02] cursor-default`}>
      <div className="flex items-center justify-between mb-1">
        <span className={`font-bold text-sm ${info.color}`}>{info.name}</span>
        <span className={`text-lg font-bold ${info.color}`}>{count}</span>
      </div>
      <div className="text-xs text-slate-600 dark:text-white/60 mb-1">{info.description}</div>
      <div className="text-xs text-slate-500 dark:text-white/40 italic">{info.examples}</div>
    </div>
  );
}

function MetricCard({ label, value, description }: { label: string; value: number; description: string }) {
  return (
    <div className="bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-700/50 dark:to-slate-800/50 rounded-xl p-4 border border-slate-200 dark:border-white/10">
      <div className="text-xs text-slate-500 dark:text-white/50 mb-1">{label}</div>
      <div className="text-2xl font-bold text-slate-900 dark:text-white mb-2">{value.toFixed(4)}</div>
      <div className="text-xs text-slate-600 dark:text-white/60">{description}</div>
    </div>
  );
}

export default function LabResultsPanel(props: {
  error: string;
  result: RunResult | null;
}) {
  return (
    <Card title="🌟 实验结果">
      {props.error ? (
        <ErrorBanner title="执行失败" message={props.error} />
      ) : null}

      {!props.error && !props.result ? (
        <EmptyState
          title="等待执行"
          description="上传 CSV 后点击「开始分析」，系统会自动完成训练并生成可视化结果。"
          className="mt-3"
        />
      ) : null}

      {props.result
        ? (() => {
            const labels = props.result.labels;
            const matrix = props.result.confusionMatrix;
            const actualCounts = matrix.map((row) =>
              row.reduce((a, b) => a + b, 0),
            );
            const predictedCounts = labels.map((_, j) =>
              matrix.reduce((sum, row) => sum + (row[j] ?? 0), 0),
            );

            return (
              <div className="mt-3 space-y-6">
                <div className="bg-gradient-to-br from-indigo-50 via-white to-purple-50 dark:from-indigo-500/10 dark:via-transparent dark:to-purple-500/10 rounded-xl border border-slate-200 dark:border-white/10 p-5">
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                    📊 模型评估
                  </h3>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <MetricCard
                      label="准确率"
                      value={props.result.metrics.accuracy}
                      description="每100颗恒星中有78颗被正确分类"
                    />
                    <MetricCard
                      label="精确率"
                      value={props.result.metrics.precision}
                      description="模型预测某类时，80%是对的"
                    />
                    <MetricCard
                      label="召回率"
                      value={props.result.metrics.recall}
                      description="模型能找到78%的真实该类恒星"
                    />
                    <MetricCard
                      label="F1分数"
                      value={props.result.metrics.f1}
                      description="精确率和召回率的综合评价"
                    />
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                    🔬 恒星分类结果（真实分布）
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {labels.map((l, i) => (
                      <StarClassBadge
                        key={l}
                        englishClass={l}
                        count={actualCounts[i] ?? 0}
                      />
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-4">
                    🎯 预测分布对比
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {labels.map((l, i) => {
                      const diff = (predictedCounts[i] ?? 0) - (actualCounts[i] ?? 0);
                      const info = STAR_CLASS_INFO[l] || {
                        name: l,
                        color: "text-slate-500",
                        bgColor: "bg-slate-100 dark:bg-slate-500/20",
                        description: "",
                        examples: ""
                      };
                      return (
                        <div
                          key={l}
                          className={`rounded-lg p-3 ${info.bgColor} transition-all hover:scale-[1.02] cursor-default`}
                        >
                          <div className="flex items-center justify-between mb-1">
                            <span className={`font-bold text-sm ${info.color}`}>{info.name}</span>
                            <span className={`text-lg font-bold ${info.color}`}>
                              {predictedCounts[i] ?? 0}
                            </span>
                          </div>
                          <div className="text-xs text-slate-600 dark:text-white/60 mb-1">
                            {info.description}
                          </div>
                          <div className={`text-xs font-medium ${
                            diff > 0 ? "text-green-600 dark:text-green-400" :
                            diff < 0 ? "text-red-600 dark:text-red-400" :
                            "text-slate-500"
                          }`}>
                            {diff > 0 ? `↑ 多预测了${diff}颗` :
                             diff < 0 ? `↓ 漏预测了${Math.abs(diff)}颗` :
                             "✓ 预测准确"}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <ConfusionMatrixTable labels={labels} matrix={matrix} />
              </div>
            );
          })()
        : null}
    </Card>
  );
}
