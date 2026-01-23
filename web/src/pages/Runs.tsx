import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import Card from "@/components/Card";
import EmptyState from "@/components/EmptyState";
import ErrorBanner from "@/components/ErrorBanner";
import { Button } from "@/components/Button";
import { apiFetch } from "@/api/http";
import type { RunListResponse, RunResult } from "@/types";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { X } from "lucide-react";

function fmtTime(iso: string) {
  const d = new Date(iso);
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(d);
}

function fmt(v: number) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 }).format(
    v,
  );
}

export default function Runs() {
  const navigate = useNavigate();

  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [data, setData] = useState<RunListResponse | null>(null);
  const [selected, setSelected] = useState<RunResult | null>(null);
  
  // Multi-select for comparison
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [showCompare, setShowCompare] = useState(false);

  const totalPages = useMemo(() => {
    if (!data) return 1;
    return Math.max(1, Math.ceil(data.total / pageSize));
  }, [data, pageSize]);

  const load = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const res = await apiFetch<RunListResponse>(
        `/api/runs?query=${encodeURIComponent(query)}&page=${page}&pageSize=${pageSize}`,
      );
      setData(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setBusy(false);
    }
  }, [query, page, pageSize]);

  useEffect(() => {
    load();
  }, [load]);

  const toggleCompare = (id: string) => {
      setCompareIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
  };

  const compareData = useMemo(() => {
      if (!data) return [];
      return data.items.filter(r => compareIds.includes(r.id)).map(r => ({
          name: r.request.datasetName.substring(0, 15),
          accuracy: r.metrics.accuracy,
          f1: r.metrics.f1,
          varSmoothing: r.request.gnbParams?.varSmoothing
      }));
  }, [data, compareIds]);

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-6">
      <div className="space-y-5">
        <Card title="筛选">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-sm text-white outline-none sm:flex-1"
              placeholder="按数据集名关键字筛选"
            />
            <div className="flex items-center gap-3">
              <Button
                onClick={() => {
                  setPage(1);
                  load();
                }}
                disabled={busy}
              >
                查询
              </Button>
              <Button
                variant="secondary"
                onClick={() => {
                  setQuery("");
                  setPage(1);
                  setSelected(null);
                  setCompareIds([]);
                  setTimeout(load, 0);
                }}
                disabled={busy}
              >
                重置
              </Button>
            </div>
          </div>
        </Card>

        <Card title="记录列表">
            <div className="mb-4 flex items-center justify-between">
                 <div className="text-sm text-white/50">已选择 {compareIds.length} 项进行对比</div>
                 {compareIds.length > 1 && (
                     <Button onClick={() => setShowCompare(true)} className="bg-green-600 hover:bg-green-500">
                         开始对比
                     </Button>
                 )}
            </div>

          {error ? <ErrorBanner title="加载失败" message={error} /> : null}
          {!error && data && data.items.length === 0 ? (
            <EmptyState
              title="暂无记录"
              description="回到实验台运行一次训练/预测即可生成记录。"
              className="mt-3"
            />
          ) : null}

          {data && data.items.length > 0 ? (
            <div className="mt-3 overflow-auto rounded-lg border border-white/10">
              <table className="min-w-full border-separate border-spacing-0 text-sm">
                <thead>
                  <tr className="bg-white/5">
                    <th className="px-3 py-2 text-left font-medium text-white/80 w-10">
                       <input type="checkbox" 
                            checked={compareIds.length === data.items.length}
                            onChange={() => {
                                if (compareIds.length === data.items.length) setCompareIds([]);
                                else setCompareIds(data.items.map(r => r.id));
                            }}
                       />
                    </th>
                    <th className="px-3 py-2 text-left font-medium text-white/80">
                      时间
                    </th>
                    <th className="px-3 py-2 text-left font-medium text-white/80">
                      数据集
                    </th>
                    <th className="px-3 py-2 text-left font-medium text-white/80">
                      目标列
                    </th>
                    <th className="px-3 py-2 text-left font-medium text-white/80">
                      Accuracy
                    </th>
                    <th className="px-3 py-2 text-left font-medium text-white/80">
                      F1
                    </th>
                    <th className="px-3 py-2 text-left font-medium text-white/80">
                      操作
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((r) => (
                    <tr
                      key={r.id}
                      className="cursor-pointer hover:bg-white/5"
                      onClick={() => setSelected(r)}
                    >
                      <td className="border-b border-white/5 px-3 py-2" onClick={e => e.stopPropagation()}>
                          <input type="checkbox" 
                            checked={compareIds.includes(r.id)}
                            onChange={() => toggleCompare(r.id)}
                          />
                      </td>
                      <td className="border-b border-white/5 px-3 py-2 text-white/70">
                        {fmtTime(r.createdAt)}
                      </td>
                      <td className="border-b border-white/5 px-3 py-2 text-white/70">
                        {r.request.datasetName}
                      </td>
                      <td className="border-b border-white/5 px-3 py-2 text-white/70">
                        {r.request.targetColumn}
                      </td>
                      <td className="border-b border-white/5 px-3 py-2 text-white/70">
                        {fmt(r.metrics.accuracy)}
                      </td>
                      <td className="border-b border-white/5 px-3 py-2 text-white/70">
                        {fmt(r.metrics.f1)}
                      </td>
                      <td className="border-b border-white/5 px-3 py-2">
                        <Button
                          variant="secondary"
                          className="h-8 px-2 py-1 text-xs"
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelected(r);
                          }}
                        >
                          查看
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          <div className="mt-4 flex items-center justify-between">
            <div className="text-sm text-white/50">
              共 {data?.total ?? 0} 条
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                disabled={page <= 1 || busy}
                onClick={() => setPage((p) => p - 1)}
              >
                上一页
              </Button>
              <div className="text-sm text-white/70">
                {page} / {totalPages}
              </div>
              <Button
                variant="secondary"
                disabled={page >= totalPages || busy}
                onClick={() => setPage((p) => p + 1)}
              >
                下一页
              </Button>
            </div>
          </div>
        </Card>
      </div>

      {/* Comparison Modal */}
      {showCompare && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-6">
              <div className="w-full max-w-4xl rounded-xl border border-white/10 bg-[#0B1220] p-6 shadow-2xl">
                  <div className="mb-6 flex items-center justify-between">
                      <h2 className="text-xl font-bold">实验对比分析</h2>
                      <button onClick={() => setShowCompare(false)} className="rounded-full p-1 hover:bg-white/10">
                          <X className="h-6 w-6" />
                      </button>
                  </div>
                  
                  <div className="h-[400px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={compareData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                              <XAxis dataKey="name" stroke="#ffffff80" />
                              <YAxis stroke="#ffffff80" />
                              <Tooltip contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151' }} />
                              <Legend />
                              <Bar dataKey="accuracy" name="准确率" fill="#3b82f6" />
                              <Bar dataKey="f1" name="F1 分数" fill="#10b981" />
                          </BarChart>
                      </ResponsiveContainer>
                  </div>

                  <div className="mt-6 overflow-auto">
                      <table className="w-full text-sm text-left">
                          <thead>
                              <tr className="border-b border-white/10 text-white/60">
                                  <th className="py-2">实验名称</th>
                                  <th className="py-2">Var Smoothing</th>
                                  <th className="py-2">Accuracy</th>
                                  <th className="py-2">F1</th>
                              </tr>
                          </thead>
                          <tbody>
                              {compareData.map((d, i) => (
                                  <tr key={i} className="border-b border-white/5">
                                      <td className="py-2 font-medium">{d.name}</td>
                                      <td className="py-2 font-mono text-xs">{d.varSmoothing?.toExponential(2) ?? "Default"}</td>
                                      <td className="py-2 text-blue-400">{fmt(d.accuracy)}</td>
                                      <td className="py-2 text-green-400">{fmt(d.f1)}</td>
                                  </tr>
                              ))}
                          </tbody>
                      </table>
                  </div>
              </div>
          </div>
      )}

      {selected ? (
        <div className="fixed inset-0 z-40 flex items-end justify-center bg-black/50 p-4 sm:items-center">
          <div className="w-full max-w-3xl rounded-xl border border-white/10 bg-[#111A2E] p-4 shadow-xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-white">记录详情</div>
                <div className="mt-1 text-sm text-white/60">
                  {selected.request.datasetName}
                </div>
              </div>
              <Button
                variant="secondary"
                onClick={() => setSelected(null)}
                className="h-8 px-2 py-1 text-xs"
              >
                关闭
              </Button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-xs text-white/50">Accuracy</div>
                <div className="mt-1 text-sm font-semibold text-white">
                  {fmt(selected.metrics.accuracy)}
                </div>
              </div>
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-xs text-white/50">Precision</div>
                <div className="mt-1 text-sm font-semibold text-white">
                  {fmt(selected.metrics.precision)}
                </div>
              </div>
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-xs text-white/50">Recall</div>
                <div className="mt-1 text-sm font-semibold text-white">
                  {fmt(selected.metrics.recall)}
                </div>
              </div>
              <div className="rounded-lg bg-white/5 p-3">
                <div className="text-xs text-white/50">F1</div>
                <div className="mt-1 text-sm font-semibold text-white">
                  {fmt(selected.metrics.f1)}
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm text-white/70">
              目标列：{selected.request.targetColumn}；特征列：
              {selected.request.featureColumns.join(", ")}
            </div>

            <div className="mt-4 overflow-auto rounded-lg border border-white/10">
              <table className="min-w-full border-separate border-spacing-0 text-sm">
                <thead>
                  <tr className="bg-white/5">
                    <th className="sticky left-0 z-10 w-32 border-b border-white/10 bg-white/5 px-3 py-2 text-left font-medium text-white/80">
                      混淆矩阵
                    </th>
                    {selected.labels.map((l) => (
                      <th
                        key={l}
                        className="border-b border-white/10 px-3 py-2 text-left font-medium text-white/80"
                      >
                        预测:{l}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {selected.confusionMatrix.map((row, i) => (
                    <tr
                      key={selected.labels[i] ?? String(i)}
                      className="hover:bg-white/5"
                    >
                      <td className="sticky left-0 z-10 border-b border-white/5 bg-[#111A2E] px-3 py-2 font-medium text-white/80">
                        实际:{selected.labels[i] ?? "?"}
                      </td>
                      {row.map((v, j) => (
                        <td
                          key={j}
                          className="border-b border-white/5 px-3 py-2 text-white/70"
                        >
                          {v}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-5 flex items-center justify-end gap-3">
              <Button
                onClick={() => {
                  navigate("/", { state: { prefill: selected.request } });
                  setSelected(null);
                }}
              >
                加载配置并重试
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}