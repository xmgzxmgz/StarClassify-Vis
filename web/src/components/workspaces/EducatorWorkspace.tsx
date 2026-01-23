import { useState, useMemo } from "react";
import { BookOpen, AlertTriangle } from "lucide-react";
import { parseCsvOverview, type CsvOverview } from "@/utils/csv";
import LabConfigPanel from "@/components/LabConfigPanel";
import LabResultsPanel from "@/components/LabResultsPanel";
import { apiFetch } from "@/api/http";
import type { RunCreateRequest, RunResult } from "@/types";
import { useToastStore } from "@/hooks/useToast";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function EducatorWorkspace() {
  const pushToast = useToastStore((s) => s.push);

  const [activeTemplate, setActiveTemplate] = useState<string>("");

  // Reusing Lab Logic State (Simplified for brevity, ideally extracted to hook)
  const [file, setFile] = useState<File | null>(null);
  const [overview, setOverview] = useState<CsvOverview | null>(null);
  const [datasetName, setDatasetName] = useState<string>("");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [randomState, setRandomState] = useState<string>("42");
  const [varSmoothing, setVarSmoothing] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<RunResult | null>(null);

  // Demo State
  const [demoSmoothing, setDemoSmoothing] = useState(1e-9);

  const templates = [
    {
      id: "overfit",
      title: "过拟合演示",
      desc: "展示当训练数据过少或模型过于复杂时的表现。",
      datasetUrl: "/datasets/sdss_like_tiny.csv",
      config: { testSize: 0.1, varSmoothing: "1e-11" },
    },
    {
      id: "feature_impact",
      title: "特征选择影响",
      desc: "对比只使用单一特征与使用所有特征的区别。",
      datasetUrl: "/datasets/sdss_like_small.csv",
      config: { testSize: 0.3, varSmoothing: "1e-9" },
    },
  ];

  async function loadTemplate(t: (typeof templates)[0]) {
    setActiveTemplate(t.id);
    setBusy(true);
    try {
      const response = await fetch(t.datasetUrl);
      const blob = await response.blob();
      const loadedFile = new File([blob], t.datasetUrl.split("/").pop()!, {
        type: "text/csv",
      });

      setFile(loadedFile);
      setDatasetName(t.title);

      const ov = await parseCsvOverview(loadedFile);
      setOverview(ov);

      // Auto config based on template
      setTargetColumn("class");
      setFeatureColumns(
        ov.headers.filter(
          (h) => h !== "class" && ov.numericColumns.includes(h),
        ),
      );
      setTestSize(t.config.testSize);
      setVarSmoothing(t.config.varSmoothing);

      pushToast({ title: "模板加载成功", description: t.title });
    } catch (e) {
      setError("模板加载失败: " + e);
    } finally {
      setBusy(false);
    }
  }

  // Gaussian Demo Data Generator
  const gaussianData = (() => {
    const data = [];
    for (let x = -3; x <= 3; x += 0.1) {
      const sigma = Math.sqrt(1 + Math.log10(1 / demoSmoothing + 1)); // Fake visual correlation
      const y =
        (1 / (Math.sqrt(2 * Math.PI) * sigma)) *
        Math.exp(-(x * x) / (2 * sigma * sigma));
      data.push({ x: x.toFixed(1), y });
    }
    return data;
  })();

  // Reused run logic (simplified)
  async function runTrainAndSave() {
    // ... (Same logic as Researcher, but we can call it directly if file is ready)
    // For brevity, just duplicating essential parts or we should refactor.
    // I'll assume we pass the configured state to the panel and let it handle the run button click
    // which calls a function. I'll implement a simple one here.
    if (!file) return;
    const payload: RunCreateRequest = {
      datasetName,
      targetColumn,
      featureColumns,
      testSize,
      randomState: Number(randomState) || 42,
      modelType: "gaussian_nb",
      gnbParams: { varSmoothing: Number(varSmoothing) || 1e-9 },
    };
    const fd = new FormData();
    fd.append("file", file);
    fd.append("payload", JSON.stringify(payload));
    setBusy(true);
    try {
      const res = await apiFetch<RunResult>("/api/runs", {
        method: "POST",
        body: fd,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // Error Analysis Logic
  const errorAnalysis = useMemo(() => {
    if (!result) return null;
    const { confusionMatrix, labels } = result;
    const errors: string[] = [];

    confusionMatrix.forEach((row, i) => {
      row.forEach((count, j) => {
        if (i !== j && count > 0) {
          // Check if this is a significant error (>10% of actual class)
          const actualTotal = row.reduce((a, b) => a + b, 0);
          if (count / actualTotal > 0.1) {
            errors.push(
              `⚠️ 类别 ${labels[i]} 有 ${count} 例被误判为 ${labels[j]}。原因可能是这两类恒星在温度或光度特征上存在重叠。`,
            );
          }
        }
      });
    });

    if (errors.length === 0) return "✅ 模型表现优异，未发现明显混淆模式。";
    return errors;
  }, [result]);

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[300px_1fr]">
        {/* Sidebar: Templates & Principles */}
        <div className="space-y-6">
          <div className="rounded-lg border border-white/10 bg-white/5 p-4">
            <h3 className="mb-3 flex items-center gap-2 text-lg font-semibold text-green-400">
              <BookOpen className="h-5 w-5" /> 教学模板
            </h3>
            <div className="space-y-3">
              {templates.map((t) => (
                <button
                  key={t.id}
                  onClick={() => loadTemplate(t)}
                  className={`w-full text-left rounded p-3 transition border ${activeTemplate === t.id ? "border-green-500 bg-green-500/10" : "border-white/10 hover:bg-white/5"}`}
                >
                  <div className="font-medium text-sm">{t.title}</div>
                  <div className="text-xs text-white/50 mt-1">{t.desc}</div>
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-white/10 bg-white/5 p-4 relative group">
            <h3 className="mb-3 text-sm font-semibold text-blue-300 flex items-center justify-between">
              原理可视化：高斯分布
              <button
                className="text-xs text-white/40 hover:text-white underline"
                onClick={() =>
                  alert(
                    "高斯朴素贝叶斯假设特征服从正态分布。调整平滑参数可以改变分布曲线的'宽窄'，从而影响模型对边缘数据的容忍度。",
                  )
                }
              >
                原理?
              </button>
            </h3>

            {/* Principle Popup (Hover/Click) - Simplified as hover for now or the alert above */}

            <p className="text-xs text-white/60 mb-2">
              拖动滑块观察平滑参数对分布曲线的影响。
            </p>
            <div className="h-32 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={gaussianData}>
                  <Line
                    type="monotone"
                    dataKey="y"
                    stroke="#8884d8"
                    dot={false}
                    strokeWidth={2}
                  />
                  <Tooltip />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <input
              type="range"
              min="1e-11"
              max="1e-5"
              step="1e-11"
              className="w-full mt-2"
              onChange={(e) => setDemoSmoothing(Number(e.target.value))}
            />
            <div className="text-center text-xs text-white/50 mt-1">
              算法精度调节 (var_smoothing): {demoSmoothing.toExponential(1)}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="space-y-5">
          {/* If no file loaded, show guide */}
          {!file && (
            <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-white/20 bg-white/5">
              <div className="text-center text-white/40">
                <BookOpen className="mx-auto mb-2 h-10 w-10 opacity-50" />
                <p>请选择左侧教学模板或上传数据集开始</p>
              </div>
            </div>
          )}

          {/* Reuse Config Panel (Hidden advanced settings?) or just show simple controls */}
          {file && (
            <div className="space-y-4">
              <div className="rounded-lg bg-blue-500/10 border border-blue-500/20 p-4 flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-bold">{datasetName}</h2>
                  <p className="text-sm text-white/60">
                    目标: {targetColumn} | 特征: {featureColumns.length} 个
                  </p>
                </div>
                <button
                  onClick={runTrainAndSave}
                  disabled={busy}
                  className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-lg font-medium transition disabled:opacity-50"
                >
                  {busy ? "运行中..." : "开始实验分析"}
                </button>
              </div>

              <LabResultsPanel error={error} result={result} />

              {/* Error Analysis Section */}
              {result && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-4">
                  <h3 className="mb-2 flex items-center gap-2 text-lg font-semibold text-red-400">
                    <AlertTriangle className="h-5 w-5" /> 错题分析与诊断
                  </h3>
                  {Array.isArray(errorAnalysis) ? (
                    <ul className="list-disc pl-5 space-y-1 text-sm text-white/80">
                      {errorAnalysis.map((err, i) => (
                        <li key={i}>{err}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-sm text-green-400">{errorAnalysis}</p>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
