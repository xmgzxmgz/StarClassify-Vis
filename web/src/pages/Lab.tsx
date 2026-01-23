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

/**
 * 实验台页面组件。
 * @returns 实验台页面内容。
 */
export default function Lab() {
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

  /**
   * 切换特征列的选择状态。
   * @param col 特征列名。
   * @returns 无返回值。
   */
  function toggleFeature(col: string) {
    setFeatureColumns((prev) => {
      if (prev.includes(col)) return prev.filter((x) => x !== col);
      return [...prev, col];
    });
  }

  /**
   * 根据表头推断目标列。
   * @param headers CSV 表头数组。
   * @returns 推断的目标列名。
   */
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

  /**
   * 根据数据概览推断特征列。
   * @param ov CSV 概览信息。
   * @param target 目标列名。
   * @returns 特征列数组。
   */
  function pickFeatures(ov: CsvOverview, target: string) {
    const numeric = ov.numericColumns.filter((c) => c !== target);
    if (numeric.length > 0) return numeric;
    return ov.headers.filter((c) => c !== target);
  }

  /**
   * 处理 CSV 文件选择并自动推荐字段。
   * @param f 选择的文件。
   * @returns Promise<void>。
   */
  async function onPickFile(f: File | null) {
    setError("");
    setResult(null);
    setOverview(null);
    setTargetColumn("");
    setFeatureColumns([]);
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
      pushToast({
        title: "CSV 已载入",
        description: `识别到 ${ov.headers.length} 列，${ov.rowCount} 行`,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "CSV 解析失败");
    }
  }

  /**
   * 触发训练、预测并保存结果。
   * @returns Promise<void>。
   */
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
    const vs = varSmoothing.trim() === "" ? undefined : Number(varSmoothing);
    if (varSmoothing.trim() !== "" && (Number.isNaN(vs) || vs! < 0)) {
      setError("var_smoothing 必须为非负数");
      return;
    }

    const payload: RunCreateRequest = {
      datasetName: datasetName.trim(),
      targetColumn: inferredTarget,
      featureColumns: inferredFeatures,
      testSize,
      randomState: rs,
      modelType: "gaussian_nb",
      gnbParams: {
        varSmoothing: vs,
      },
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
      pushToast({
        title: "训练/预测完成",
        description: "结果已保存到数据库",
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-6">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[420px_1fr]">
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

        <div className="space-y-5">
          <LabResultsPanel error={error} result={result} />
        </div>
      </div>
    </div>
  );
}
