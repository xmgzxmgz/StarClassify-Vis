import Card from "@/components/Card";
import EmptyState from "@/components/EmptyState";
import ErrorBanner from "@/components/ErrorBanner";
import ConfusionMatrixTable from "@/components/ConfusionMatrixTable";
import ResultMetrics from "@/components/ResultMetrics";
import MetricRing from "@/components/MetricRing";
import DistributionBars from "@/components/DistributionBars";
import type { RunResult } from "@/types";

/**
 * 渲染实验结果面板。
 * @param props 组件输入参数。
 * @returns 结果面板组件。
 */
export default function LabResultsPanel(props: {
  error: string;
  result: RunResult | null;
}) {
  return (
    <Card title="结果展示">
      {props.error ? (
        <ErrorBanner title="执行失败" message={props.error} />
      ) : null}

      {!props.error && !props.result ? (
        <EmptyState
          title="等待执行"
          description="上传 CSV 后点击“开始分析”，系统会自动完成训练并生成可视化结果。"
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
            const total = actualCounts.reduce((a, b) => a + b, 0);

            return (
              <div className="mt-3 space-y-4">
                <div className="rounded-xl border border-white/10 bg-gradient-to-br from-blue-500/10 via-transparent to-purple-500/10 p-4 shadow-[0_0_30px_rgba(59,130,246,0.12)]">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    <div className="space-y-2">
                      <div className="text-sm font-semibold text-white">
                        分析总览
                      </div>
                      <div className="text-xs text-white/60">
                        {props.result.request.datasetName} · 目标列{" "}
                        {props.result.request.targetColumn} · 特征{" "}
                        {props.result.request.featureColumns.length} 列
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-xs text-white/70 sm:grid-cols-4">
                        <div className="rounded-lg bg-white/5 p-2">
                          样本 {total}
                        </div>
                        <div className="rounded-lg bg-white/5 p-2">
                          类别 {labels.length}
                        </div>
                        <div className="rounded-lg bg-white/5 p-2">
                          测试占比{" "}
                          {Math.round(props.result.request.testSize * 100)}%
                        </div>
                        <div className="rounded-lg bg-white/5 p-2">
                          模型 GNB
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center justify-center gap-6">
                      <MetricRing
                        value={props.result.metrics.accuracy}
                        label="准确率"
                        tone="blue"
                      />
                      <MetricRing
                        value={props.result.metrics.f1}
                        label="F1"
                        tone="green"
                      />
                    </div>
                  </div>
                </div>

                <ResultMetrics
                  accuracy={props.result.metrics.accuracy}
                  precision={props.result.metrics.precision}
                  recall={props.result.metrics.recall}
                  f1={props.result.metrics.f1}
                />

                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <Card title="类别分布（真实）" className="glow-panel">
                    <DistributionBars
                      title="按真实标签统计"
                      items={labels.map((l, i) => ({
                        label: l,
                        value: actualCounts[i] ?? 0,
                      }))}
                      tone="blue"
                    />
                  </Card>
                  <Card title="类别分布（预测）" className="glow-panel">
                    <DistributionBars
                      title="按预测标签统计"
                      items={labels.map((l, i) => ({
                        label: l,
                        value: predictedCounts[i] ?? 0,
                      }))}
                      tone="purple"
                    />
                  </Card>
                </div>

                <ConfusionMatrixTable labels={labels} matrix={matrix} />
              </div>
            );
          })()
        : null}
    </Card>
  );
}
