import { Link } from "react-router-dom";
import { Loader2 } from "lucide-react";
import Card from "@/components/Card";
import FormRow from "@/components/FormRow";
import { Button } from "@/components/Button";
import type { CsvOverview } from "@/utils/csv";
import type { RunResult } from "@/types";

/**
 * 渲染实验台左侧配置面板。
 * @param props 组件输入参数。
 * @returns 配置面板组件。
 */
export default function LabConfigPanel(props: {
  overview: CsvOverview | null;
  datasetName: string;
  setDatasetName: (v: string) => void;
  targetColumn: string;
  setTargetColumn: (v: string) => void;
  headerOptions: string[];
  featureOptions: string[];
  featureColumns: string[];
  toggleFeature: (col: string) => void;
  testSize: number;
  setTestSize: (v: number) => void;
  randomState: string;
  setRandomState: (v: string) => void;
  varSmoothing: string;
  setVarSmoothing: (v: string) => void;
  busy: boolean;
  result: RunResult | null;
  onPickFile: (f: File | null) => void;
  onRun: () => void;
}) {
  return (
    <div className="space-y-5">
      <Card title="数据导入">
        <div className="space-y-3">
          <input
            type="file"
            accept=".csv,text/csv"
            onChange={(e) => props.onPickFile(e.target.files?.[0] ?? null)}
            className="block w-full cursor-pointer rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white file:mr-3 file:rounded-md file:border-0 file:bg-slate-100 dark:file:bg-white/10 file:px-3 file:py-2 file:text-sm file:text-slate-900 dark:file:text-white hover:bg-slate-50 dark:hover:bg-white/10"
          />
          {props.overview ? (
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-lg bg-slate-100 dark:bg-white/5 p-3">
                <div className="text-xs text-slate-600 dark:text-white/50">列数</div>
                <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
                  {props.overview.headers.length}
                </div>
              </div>
              <div className="rounded-lg bg-slate-100 dark:bg-white/5 p-3">
                <div className="text-xs text-slate-600 dark:text-white/50">行数</div>
                <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
                  {props.overview.rowCount}
                </div>
              </div>
              <div className="rounded-lg bg-slate-100 dark:bg-white/5 p-3">
                <div className="text-xs text-slate-600 dark:text-white/50">缺失值</div>
                <div className="mt-1 text-sm font-semibold text-slate-900 dark:text-white">
                  {props.overview.missingCells}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-600 dark:text-white/50">
              仅支持 CSV；上传后点击"开始分析"，系统会自动推荐目标与特征列
            </div>
          )}
          {props.overview ? (
            <div className="rounded-lg border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 p-3 text-xs text-slate-600 dark:text-white/60">
              已自动推荐：目标列 {props.targetColumn || "未识别"}，特征列{" "}
              {props.featureColumns.length} 项
            </div>
          ) : null}
        </div>
      </Card>

      <Card title="快速开始">
        <div className="space-y-4">
          <div className="text-sm text-slate-700 dark:text-white/70">
            上传 CSV
            后点击开始分析，系统会自动完成字段推荐、训练并生成可视化结果
          </div>
          <Button onClick={props.onRun} disabled={props.busy}>
            {props.busy ? (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                分析中
              </span>
            ) : (
              "开始分析"
            )}
          </Button>
          {props.result ? (
            <Link className="text-sm text-blue-600 dark:text-blue-300 hover:underline" to="/runs">
              查看结果记录
            </Link>
          ) : null}
        </div>
      </Card>

      <Card title="示例数据">
        <div className="space-y-2 text-sm text-slate-700 dark:text-white/70">
          <div>位置：项目根目录 /DB</div>
          <div className="grid grid-cols-1 gap-2 text-xs text-slate-600 dark:text-white/60">
            <div>star_data_small.csv (500 样本)</div>
            <div>star_data_medium.csv (1000 样本)</div>
            <div>star_data_large.csv (2000 样本)</div>
          </div>
        </div>
      </Card>

      <Card title="高级设置">
        <div className="space-y-4">
          <FormRow
            label="测试集比例"
            hint={`${Math.round(props.testSize * 100)}%`}
          >
            <input
              type="range"
              min={0.1}
              max={0.5}
              step={0.05}
              value={props.testSize}
              onChange={(e) => props.setTestSize(Number(e.target.value))}
              className="w-full"
            />
          </FormRow>

          <FormRow label="随机种子" hint="可选">
            <input
              value={props.randomState}
              onChange={(e) => props.setRandomState(e.target.value)}
              className="w-full rounded-lg border border-slate-300 dark:border-white/15 bg-white dark:bg-white/5 px-3 py-2 text-sm text-slate-900 dark:text-white outline-none"
              placeholder="例如：42"
            />
          </FormRow>
        </div>
      </Card>
    </div>
  );
}
