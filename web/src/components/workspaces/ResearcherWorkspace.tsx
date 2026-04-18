import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Loader2, FileSpreadsheet, Database, Brain, BarChart3, CheckCircle } from "lucide-react";
import Card from "@/components/Card";
import { Button } from "@/components/Button";
import LabResultsPanel from "@/components/LabResultsPanel";
import { apiFetch } from "@/api/http";
import { parseCsvOverview, type CsvOverview } from "@/utils/csv";
import type { RunCreateRequest, RunResult } from "@/types";
import { useToastStore } from "@/hooks/useToast";

type PrefillState = {
  prefill?: RunCreateRequest;
};

type WorkflowStep = 1 | 2 | 3 | 4;

export default function ResearcherWorkspace() {
  const location = useLocation();
  const prefill = (location.state as PrefillState | null)?.prefill;

  const pushToast = useToastStore((s) => s.push);

  const [currentStep, setCurrentStep] = useState<WorkflowStep>(1);
  const [file, setFile] = useState<File | null>(null);
  const [overview, setOverview] = useState<CsvOverview | null>(null);
  const [datasetName, setDatasetName] = useState<string>("");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [randomState, setRandomState] = useState<string>("42");
  const [varSmoothing, setVarSmoothing] = useState<string>("");

  const [qualityReport, setQualityReport] = useState<string[]>([]);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<RunResult | null>(null);

  useEffect(() => {
    if (!prefill) return;
    setDatasetName(prefill.datasetName ?? "");
    setTargetColumn(prefill.targetColumn ?? "");
    setFeatureColumns(prefill.featureColumns ?? []);
    setTestSize(prefill.testSize ?? 0.2);
    setRandomState(
      prefill.randomState != null ? String(prefill.randomState) : "",
    );
    setVarSmoothing(
      prefill.gnbParams?.varSmoothing != null
        ? String(prefill.gnbParams.varSmoothing)
        : "",
    );
  }, [prefill]);

  const headerOptions = useMemo(() => overview?.headers ?? [], [overview]);
  const featureOptions = useMemo(() => {
    if (!overview) return [];
    return overview.headers.filter((h) => h !== targetColumn);
  }, [overview, targetColumn]);

  useEffect(() => {
    setFeatureColumns((prev) => prev.filter((x) => x !== targetColumn));
  }, [targetColumn]);

  function toggleFeature(col: string) {
    setFeatureColumns((prev) => {
      if (prev.includes(col)) return prev.filter((x) => x !== col);
      return [...prev, col];
    });
  }

  function pickTarget(headers: string[]) {
    const hints = [
      "class",
      "label",
      "target",
      "type",
      "类别",
      "分类",
      "星类",
      "star_type",
    ];
    const lower = headers.map((h) => h.toLowerCase());
    for (const hint of hints) {
      const idx = lower.findIndex((h) => h === hint);
      if (idx >= 0) return headers[idx];
    }
    for (const hint of hints) {
      const idx = lower.findIndex((h) => h.includes(hint));
      if (idx >= 0) return headers[idx];
    }
    return headers[headers.length - 1] ?? "";
  }

  function pickFeatures(ov: CsvOverview, target: string) {
    const numeric = ov.numericColumns.filter((c) => c !== target);
    if (numeric.length > 0) return numeric;
    return ov.headers.filter((c) => c !== target);
  }

  function runQualityCheck(ov: CsvOverview) {
    const suggestions: string[] = [];
    if (ov.rowCount < 50) {
      suggestions.push(
        "⚠️ 数据量过少 (<50行)，可能导致模型过拟合。建议扩充数据。",
      );
    }

    for (const [col, stats] of Object.entries(ov.columnStats)) {
      if (stats.missing > 0) {
        const isNumeric = ov.numericColumns.includes(col);
        const action = isNumeric ? `建议填充均值 (${stats.mean?.toFixed(2)})` : "建议填充众数";
        suggestions.push(`ℹ️ 列 '${col}' 存在 ${stats.missing} 个缺失值，${action}。`);
      }

      if (col.toLowerCase().includes("temp") && stats.min !== undefined && stats.min < 0) {
        suggestions.push(`⚠️ 列 '${col}' 存在负值 (${stats.min})，物理上可能异常，建议检查。`);
      }
    }

    if (suggestions.length === 0) {
      suggestions.push("✅ 数据质量良好，未发现明显缺失或异常。");
    }

    setQualityReport(suggestions);
  }

  async function onPickFile(f: File | null) {
    setError("");
    setResult(null);
    setOverview(null);
    setTargetColumn("");
    setFeatureColumns([]);
    setQualityReport([]);
    setFile(f);
    if (!f) return;

    setDatasetName(f.name);
    try {
      const ov = await parseCsvOverview(f);
      setOverview(ov);
      const nextTarget = pickTarget(ov.headers);
      const nextFeatures = pickFeatures(ov, nextTarget);
      setTargetColumn(nextTarget);
      setFeatureColumns(nextFeatures);

      runQualityCheck(ov);

      pushToast({
        title: "CSV 已载入",
        description: `识别到 ${ov.headers.length} 列，${ov.rowCount} 行`,
      });

      setCurrentStep(2);
    } catch (e) {
      setError(e instanceof Error ? e.message : "CSV 解析失败");
    }
  }

  function proceedToStep3() {
    if (!targetColumn) {
      setError("请选择目标列（MK分类标签）");
      return;
    }
    if (featureColumns.length === 0) {
      setError("请至少选择 1 个特征列");
      return;
    }
    setError("");
    setCurrentStep(3);
  }

  async function runTrainAndSave() {
    setError("");
    setResult(null);
    if (!file) {
      setError("请先上传 CSV 文件");
      return;
    }
    if (!datasetName.trim()) {
      setError("请填写数据集名称");
      return;
    }
    const inferredTarget =
      targetColumn || (overview ? pickTarget(overview.headers) : "");
    const inferredFeatures =
      featureColumns.length > 0
        ? featureColumns
        : overview
          ? pickFeatures(overview, inferredTarget)
          : [];
    if (!inferredTarget) {
      setError("请选择目标列（MK分类标签）");
      return;
    }
    if (inferredFeatures.length === 0) {
      setError("请至少选择 1 个特征列");
      return;
    }
    if (inferredTarget !== targetColumn) {
      setTargetColumn(inferredTarget);
    }
    if (inferredFeatures !== featureColumns) {
      setFeatureColumns(inferredFeatures);
    }

    const rs = randomState.trim() === "" ? undefined : Number(randomState);
    if (randomState.trim() !== "" && Number.isNaN(rs)) {
      setError("random_state 必须为整数");
      return;
    }

    let vsList = varSmoothing
      .split(",")
      .map((v) => v.trim())
      .filter((v) => v !== "");

    if (vsList.length === 0) {
      vsList = [""];
    }

    setBusy(true);
    try {
      const results = [];
      for (const vsStr of vsList) {
        let vs: number | undefined = undefined;
        if (vsStr !== "") {
          vs = Number(vsStr);
          if (Number.isNaN(vs) || vs < 0) {
            setError(`参数错误: ${vsStr} 不是合法的 var_smoothing`);
            setBusy(false);
            return;
          }
        }

        const payload: RunCreateRequest = {
          datasetName:
            datasetName.trim() +
            (vsList.length > 1 && vsStr !== "" ? ` (vs=${vsStr})` : ""),
          targetColumn: inferredTarget,
          featureColumns: inferredFeatures,
          testSize,
          randomState: rs,
          modelType: "gaussian_nb",
          gnbParams: { varSmoothing: vs },
        };

        const fd = new FormData();
        fd.append("file", file);
        fd.append("payload", JSON.stringify(payload));

        const res = await apiFetch<RunResult>("/api/runs", {
          method: "POST",
          body: fd,
        });
        results.push(res);
      }

      setResult(results[results.length - 1]);
      setCurrentStep(4);
      pushToast({
        title: "分类训练完成",
        description: `成功运行 ${results.length} 组实验`,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setBusy(false);
    }
  }

  function resetWorkflow() {
    setCurrentStep(1);
    setFile(null);
    setOverview(null);
    setDatasetName("");
    setTargetColumn("");
    setFeatureColumns([]);
    setQualityReport([]);
    setError("");
    setResult(null);
  }

  const steps = [
    { num: 1, title: "数据获取", icon: FileSpreadsheet, desc: "从SDSS光谱表筛选恒星样本" },
    { num: 2, title: "数据预处理", icon: Database, desc: "剔除异常值与缺失数据" },
    { num: 3, title: "模型训练", icon: Brain, desc: "通过高斯朴素贝叶斯学习分类规则" },
    { num: 4, title: "分类与应用", icon: BarChart3, desc: "完成MK分类预测并可视化" },
  ];

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-6">
        <div className="rounded-lg bg-white dark:bg-white/5 p-5 border border-slate-300 dark:border-white/10">
          <h3 className="mb-2 text-sm font-semibold text-blue-600 dark:text-blue-400">
            恒星MK分类系统
          </h3>
          <p className="text-sm text-slate-600 dark:text-white/60 mb-4">
            从SDSS中获取恒星光学光谱数据与测光星表数据，经清洗、特征提取后，以光谱特征与u/g/r/i/z波段星等为输入、MK分类标签为目标，使用高斯朴素贝叶斯模型完成训练与预测，最终实现恒星MK分类并可视化展示、实验记录回溯。
          </p>

          <div className="flex items-center gap-3 overflow-x-auto pb-2">
            {steps.map((step, idx) => {
              const Icon = step.icon;
              const isActive = currentStep === step.num;
              const isCompleted = currentStep > step.num;
              return (
                <div key={step.num} className="flex items-center">
                  <div className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    isActive
                      ? "bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-300 ring-2 ring-blue-500"
                      : isCompleted
                      ? "bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-300"
                      : "bg-slate-100 dark:bg-white/5 text-slate-500 dark:text-white/40"
                  }`}>
                    <div className={`h-6 w-6 flex items-center justify-center rounded-full ${
                      isCompleted ? "bg-green-500 text-white" : ""
                    }`}>
                      {isCompleted ? <CheckCircle className="h-4 w-4" /> : step.num}
                    </div>
                    <div className="hidden sm:block">
                      <div className="text-xs font-semibold">{step.title}</div>
                      <div className="text-[10px] opacity-70">{step.desc}</div>
                    </div>
                  </div>
                  {idx < steps.length - 1 && (
                    <div className="w-8 h-0.5 bg-slate-200 dark:bg-white/10" />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {currentStep === 1 && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <Card title="步骤 1: 数据获取">
            <div className="space-y-4">
              <p className="text-sm text-slate-600 dark:text-white/60">
                从SDSS光谱表SpecObj筛选恒星样本，提取光谱波长/流量、MK分类标签（如G2V），关联测光表PhotoObj的u/g/r/i/z星等数据。
              </p>
              <div className="space-y-3">
                <div className="text-sm font-medium">上传CSV数据文件</div>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
                  className="block w-full cursor-pointer rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 dark:file:bg-white/10 file:px-3 file:py-2 file:text-sm file:text-slate-900 dark:file:text-white hover:bg-slate-50 dark:hover:bg-white/10"
                />
                <div className="grid grid-cols-1 gap-2 text-xs text-slate-600 dark:text-white/60">
                  <div>推荐使用：apogee_sample_classified.csv (47,994 样本)</div>
                  <div>或完整数据：apogee_dr17_with_classes.csv (645,502 样本)</div>
                </div>
              </div>
              <div className="rounded-lg border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 p-3">
                <div className="text-xs font-medium text-slate-700 dark:text-white/70 mb-2">示例数据</div>
                <div className="text-xs text-slate-600 dark:text-white/60">
                  位置：项目根目录 /datasets
                </div>
              </div>
            </div>
          </Card>

          <Card title="系统说明">
            <div className="space-y-4 text-sm text-slate-600 dark:text-white/60">
              <div className="space-y-2">
                <h4 className="font-medium text-slate-900 dark:text-white">精简操作流程</h4>
                <ol className="list-decimal pl-5 space-y-2">
                  <li><strong>数据获取</strong>：从SDSS光谱表筛选恒星样本</li>
                  <li><strong>数据预处理</strong>：剔除异常值与缺失数据</li>
                  <li><strong>模型训练</strong>：通过高斯朴素贝叶斯学习分类规则</li>
                  <li><strong>分类与应用</strong>：输入待测恒星数据完成MK分类预测</li>
                </ol>
              </div>
            </div>
          </Card>
        </div>
      )}

      {(currentStep === 2 || currentStep === 3) && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[420px_1fr]">
          <div className="space-y-4">
            <div className="rounded-lg bg-white dark:bg-white/5 p-4 border border-slate-300 dark:border-white/10">
              {currentStep === 2 ? (
                <>
                  <h3 className="mb-2 text-sm font-semibold text-blue-600 dark:text-blue-400">
                    步骤 2: 数据预处理
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-white/60 mb-3">
                    剔除异常值与缺失数据，对光谱流量归一化，将MK标签标准化为可训练格式。
                  </p>
                </>
              ) : (
                <>
                  <h3 className="mb-2 text-sm font-semibold text-blue-600 dark:text-blue-400">
                    步骤 3: 模型训练
                  </h3>
                  <p className="text-xs text-slate-600 dark:text-white/60 mb-3">
                    以光谱特征+测光波段为输入特征，通过高斯朴素贝叶斯算法学习MK分类规则。
                  </p>
                </>
              )}
              {qualityReport.length > 0 && (
                <div className="mb-4 rounded bg-yellow-100 dark:bg-yellow-500/10 p-3 text-xs text-yellow-800 dark:text-yellow-200">
                  <p className="font-bold mb-1">数据质控建议：</p>
                  <ul className="list-disc pl-4 space-y-1">
                    {qualityReport.map((q, i) => (
                      <li key={i}>{q}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="space-y-4">
              <Card title="数据配置">
                {overview && (
                  <div className="grid grid-cols-3 gap-3 mb-4">
                    <div className="rounded-lg bg-slate-100 dark:bg-white/5 p-3">
                      <div className="text-xs text-slate-600 dark:text-white/50">列数</div>
                      <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
                        {overview.headers.length}
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-100 dark:bg-white/5 p-3">
                      <div className="text-xs text-slate-600 dark:text-white/50">恒星样本数</div>
                      <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
                        {overview.rowCount}
                      </div>
                    </div>
                    <div className="rounded-lg bg-slate-100 dark:bg-white/5 p-3">
                      <div className="text-xs text-slate-600 dark:text-white/50">缺失(抽样)</div>
                      <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
                        {overview.missingCells}
                      </div>
                    </div>
                  </div>
                )}
                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-medium text-slate-700 dark:text-white/70 mb-1">
                      数据集名称
                    </label>
                    <input
                      value={datasetName}
                      onChange={(e) => setDatasetName(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white outline-none"
                      placeholder="例如：apogee_sample_classified.csv"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-700 dark:text-white/70 mb-1">
                      目标列（MK分类标签）
                    </label>
                    <select
                      value={targetColumn}
                      onChange={(e) => setTargetColumn(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white outline-none"
                    >
                      <option value="">请选择目标列</option>
                      {headerOptions.map((h) => (
                        <option key={h} value={h}>{h}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-700 dark:text-white/70 mb-1">
                      特征列（多选）
                    </label>
                    <div className="max-h-32 space-y-1 overflow-auto rounded-lg border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 p-2">
                      {featureOptions.map((c) => (
                        <label
                          key={c}
                          className="flex cursor-pointer items-center gap-2 text-xs text-slate-700 dark:text-white/80"
                        >
                          <input
                            type="checkbox"
                            checked={featureColumns.includes(c)}
                            onChange={() => toggleFeature(c)}
                            className="h-3 w-3 accent-blue-500"
                          />
                          <span className="truncate">{c}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </Card>

              {currentStep === 2 && (
                <Button onClick={proceedToStep3} className="w-full">
                  进入模型训练
                </Button>
              )}

              {currentStep === 3 && (
                <div className="space-y-3">
                  <Card title="高级参数">
                    <div className="space-y-3">
                      <div>
                        <label className="block text-xs font-medium text-slate-700 dark:text-white/70 mb-1">
                          test_size ({Math.round(testSize * 100)}% 用于测试集)
                        </label>
                        <input
                          type="range"
                          min={0.1}
                          max={0.5}
                          step={0.05}
                          value={testSize}
                          onChange={(e) => setTestSize(Number(e.target.value))}
                          className="w-full"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-slate-700 dark:text-white/70 mb-1">
                          random_state
                        </label>
                        <input
                          value={randomState}
                          onChange={(e) => setRandomState(e.target.value)}
                          className="w-full rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white outline-none"
                          placeholder="例如：42"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-slate-700 dark:text-white/70 mb-1">
                          var_smoothing
                        </label>
                        <input
                          value={varSmoothing}
                          onChange={(e) => setVarSmoothing(e.target.value)}
                          className="w-full rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white outline-none"
                          placeholder="例如：1e-9"
                        />
                      </div>
                    </div>
                  </Card>

                  <div className="flex gap-2">
                    <Button onClick={() => setCurrentStep(2)} variant="secondary" className="flex-1">
                      返回预处理
                    </Button>
                    <Button onClick={runTrainAndSave} disabled={busy} className="flex-1">
                      {busy ? (
                        <span className="inline-flex items-center gap-2">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          训练中
                        </span>
                      ) : "开始训练"}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-5">
            {currentStep === 2 && (
              <Card title="特征预览">
                <div className="text-sm text-slate-600 dark:text-white/60">
                  选择目标列和特征列后，点击"进入模型训练"继续下一步。
                </div>
              </Card>
            )}
            {currentStep === 3 && (
              <Card title="配置确认">
                <div className="text-sm text-slate-600 dark:text-white/60">
                  确认配置无误后，点击"开始训练"执行模型训练。
                </div>
              </Card>
            )}
          </div>
        </div>
      )}

      {currentStep === 4 && (
        <div className="space-y-5">
          <div className="rounded-lg bg-white dark:bg-white/5 p-4 border border-slate-300 dark:border-white/10">
            <h3 className="mb-2 text-sm font-semibold text-green-600 dark:text-green-400">
              步骤 4: 分类与应用
            </h3>
            <p className="text-xs text-slate-600 dark:text-white/60 mb-3">
              输入待测恒星数据完成MK分类预测，在前端展示分类结果，并将实验配置与结果持久化存储。
            </p>
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[420px_1fr]">
            <div className="space-y-4">
              <Card title="操作">
                <div className="space-y-3">
                  <Button onClick={resetWorkflow} className="w-full">
                    开始新实验
                  </Button>
                  {result && (
                    <Link to="/runs" className="block">
                      <Button variant="secondary" className="w-full">
                        查看实验记录
                      </Button>
                    </Link>
                  )}
                </div>
              </Card>
            </div>
            <LabResultsPanel error={error} result={result} />
          </div>
        </div>
      )}
    </div>
  );
}
