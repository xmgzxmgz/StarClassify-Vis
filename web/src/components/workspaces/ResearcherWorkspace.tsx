import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import LabConfigPanel from "@/components/LabConfigPanel";
import LabResultsPanel from "@/components/LabResultsPanel";
import { apiFetch } from "@/api/http";
import { parseCsvOverview, type CsvOverview } from "@/utils/csv";
import type { RunCreateRequest, RunResult } from "@/types";
import { useToastStore } from "@/hooks/useToast";

type PrefillState = {
  prefill?: RunCreateRequest;
};

export default function ResearcherWorkspace() {
  const location = useLocation();
  const prefill = (location.state as PrefillState | null)?.prefill;

  const pushToast = useToastStore((s) => s.push);

  const [file, setFile] = useState<File | null>(null);
  const [overview, setOverview] = useState<CsvOverview | null>(null);
  const [datasetName, setDatasetName] = useState<string>("");
  const [targetColumn, setTargetColumn] = useState<string>("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [testSize, setTestSize] = useState<number>(0.2);
  const [randomState, setRandomState] = useState<string>("42");
  const [varSmoothing, setVarSmoothing] = useState<string>("");

  // Data Quality Check State
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

  // Data Quality Check Logic
  function runQualityCheck(ov: CsvOverview) {
    const suggestions: string[] = [];
    if (ov.rowCount < 50) {
      suggestions.push(
        "⚠️ 数据量过少 (<50行)，可能导致模型过拟合。建议扩充数据。",
      );
    }
    
    // Check per-column stats
    for (const [col, stats] of Object.entries(ov.columnStats)) {
        if (stats.missing > 0) {
            const isNumeric = ov.numericColumns.includes(col);
            const action = isNumeric ? `建议填充均值 (${stats.mean?.toFixed(2)})` : "建议填充众数";
            suggestions.push(`ℹ️ 列 '${col}' 存在 ${stats.missing} 个缺失值，${action}。`);
        }
        
        // Simple "abnormal" check for specific known columns (e.g., temperature shouldn't be negative or zero usually, but depends on scale)
        // Here we just check for extreme outliers relative to mean if we had stddev, but we don't. 
        // We can check if min/max are valid numbers.
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
    } catch (e) {
      setError(e instanceof Error ? e.message : "CSV 解析失败");
    }
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
      setError("请选择目标列（label）");
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
    // Batch Processing Logic
    let vsList = varSmoothing
      .split(",")
      .map((v) => v.trim())
      .filter((v) => v !== "");

    // If empty, treat as single run with undefined (backend default)
    if (vsList.length === 0) {
      vsList = [""]; // Placeholder for "default"
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

      // If multiple runs, maybe show the last one or a summary?
      // For now, just showing the last one to keep UI simple, but alerting count.
      setResult(results[results.length - 1]);
      pushToast({
        title: "批量实验完成",
        description: `成功运行 ${results.length} 组实验`,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[420px_1fr]">
        <div className="space-y-4">
          <div className="rounded-lg bg-white dark:bg-white/5 p-4 border border-slate-300 dark:border-white/10">
            <h3 className="mb-2 text-sm font-semibold text-blue-600 dark:text-blue-400">
              科研助手功能
            </h3>
            <p className="text-xs text-slate-600 dark:text-white/60 mb-3">
              支持高级参数调试、批量实验与数据质控。
            </p>
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

          <LabConfigPanel
            overview={overview}
            datasetName={datasetName}
            setDatasetName={setDatasetName}
            targetColumn={targetColumn}
            setTargetColumn={setTargetColumn}
            headerOptions={headerOptions}
            featureOptions={featureOptions}
            featureColumns={featureColumns}
            toggleFeature={toggleFeature}
            testSize={testSize}
            setTestSize={setTestSize}
            randomState={randomState}
            setRandomState={setRandomState}
            varSmoothing={varSmoothing}
            setVarSmoothing={setVarSmoothing}
            busy={busy}
            result={result}
            onPickFile={onPickFile}
            onRun={runTrainAndSave}
          />
        </div>

        <div className="space-y-5">
          <LabResultsPanel error={error} result={result} />
        </div>
      </div>
    </div>
  );
}
