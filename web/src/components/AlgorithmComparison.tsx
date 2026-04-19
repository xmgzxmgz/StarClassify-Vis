/**
 * 算法对比模块
 * 展示高斯朴素贝叶斯相比其他算法的优势
 */

import { useState } from "react";
import { Brain, Shield, Zap, BarChart3, Info } from "lucide-react";

type Algorithm = "gnb" | "svm" | "rf" | "nn";

interface ComparisonResult {
  accuracy: number;
  interpretability: number;
  stability: number;
  speed: number;
}

const ALGORITHM_INFO: Record<
  Algorithm,
  {
    name: string;
    icon: React.ReactNode;
    color: string;
    strengths: string[];
    weakness: string;
    scores: ComparisonResult;
  }
> = {
  gnb: {
    name: "高斯朴素贝叶斯 (GNB)",
    icon: <BarChart3 className="h-5 w-5" />,
    color: "#22C55E",
    strengths: [
      "与高斯分布天然契合",
      "输出直观概率",
      "完全可解释",
      "无需调参",
      "训练极快",
    ],
    weakness: "特征条件独立假设在现实中不一定成立",
    scores: {
      accuracy: 0.82,
      interpretability: 0.98,
      stability: 0.95,
      speed: 0.99,
    },
  },
  svm: {
    name: "支持向量机 (SVM)",
    icon: <Shield className="h-5 w-5" />,
    color: "#3B82F6",
    strengths: ["处理高维数据能力强", "泛化能力强", "核函数灵活"],
    weakness: "黑盒模型，不易解释；大数据集训练慢；不易输出概率",
    scores: {
      accuracy: 0.88,
      interpretability: 0.25,
      stability: 0.75,
      speed: 0.45,
    },
  },
  rf: {
    name: "随机森林 (RF)",
    icon: <Zap className="h-5 w-5" />,
    color: "#F59E0B",
    strengths: ["处理复杂非线性关系", "抗噪声能力强", "特征重要性可评估"],
    weakness: "黑盒模型，单棵树可解释但整体复杂；计算开销大",
    scores: {
      accuracy: 0.91,
      interpretability: 0.20,
      stability: 0.85,
      speed: 0.55,
    },
  },
  nn: {
    name: "神经网络 (NN)",
    icon: <Brain className="h-5 w-5" />,
    color: "#8B5CF6",
    strengths: ["拟合任意复杂函数", "自动特征学习", "精度高"],
    weakness: "完全黑盒；需要大量数据；训练慢；调参困难",
    scores: {
      accuracy: 0.93,
      interpretability: 0.05,
      stability: 0.60,
      speed: 0.30,
    },
  },
};

export default function AlgorithmComparison() {
  const [selected, setSelected] = useState<Algorithm>("gnb");

  const algo = ALGORITHM_INFO[selected];

  return (
    <div className="space-y-4">
      <div className="text-sm font-medium text-slate-700 dark:text-white/80">
        算法对比：为什么科普场景选择 GNB？
      </div>

      <div className="grid grid-cols-4 gap-2">
        {(Object.keys(ALGORITHM_INFO) as Algorithm[]).map((key) => {
          const info = ALGORITHM_INFO[key];
          const isSelected = key === selected;
          return (
            <button
              key={key}
              onClick={() => setSelected(key)}
              className={`rounded-lg border p-3 text-left transition-all ${
                isSelected
                  ? "border-current bg-opacity-10"
                  : "border-slate-200 dark:border-white/10 hover:border-slate-300 dark:hover:border-white/20"
              }`}
              style={{
                borderColor: isSelected ? info.color : undefined,
                backgroundColor: isSelected ? `${info.color}15` : undefined,
              }}
            >
              <div
                className="mb-1 flex items-center gap-1"
                style={{ color: info.color }}
              >
                {info.icon}
                <span className="text-xs font-medium">{info.name.split(" ")[0]}</span>
              </div>
              <div className="text-xs text-slate-500 dark:text-white/40">
                准确率: {(info.scores.accuracy * 100).toFixed(0)}%
              </div>
            </button>
          );
        })}
      </div>

      <div className="rounded-lg border border-slate-200 dark:border-white/10 bg-white dark:bg-white/5 p-4">
        <div
          className="mb-3 flex items-center gap-2 text-sm font-medium"
          style={{ color: algo.color }}
        >
          {algo.icon}
          {algo.name}
        </div>

        <div className="mb-4">
          <div className="mb-2 text-xs font-medium text-slate-600 dark:text-white/60">
            四维评分对比（0-100%）
          </div>
          <div className="grid grid-cols-2 gap-3">
            {[
              { key: "interpretability", label: "可解释性", emoji: "🧠" },
              { key: "accuracy", label: "准确率", emoji: "🎯" },
              { key: "stability", label: "稳定性", emoji: "⚖️" },
              { key: "speed", label: "速度", emoji: "⚡" },
            ].map(({ key, label }) => {
              const value = algo.scores[key as keyof ComparisonResult] * 100;
              return (
                <div key={key} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-600 dark:text-white/60">
                      {label}
                    </span>
                    <span className="font-mono text-slate-800 dark:text-white">
                      {value.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${value}%`,
                        backgroundColor: algo.color,
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-start gap-2 text-xs">
            <span className="text-green-500">✓</span>
            <div className="flex flex-wrap gap-1">
              {algo.strengths.map((s, i) => (
                <span
                  key={i}
                  className="rounded bg-green-100 px-1.5 py-0.5 text-green-700 dark:bg-green-500/20 dark:text-green-300"
                >
                  {s}
                </span>
              ))}
            </div>
          </div>
          <div className="flex items-start gap-2 text-xs">
            <span className="text-red-500">✗</span>
            <span className="text-slate-600 dark:text-white/60">
              {algo.weakness}
            </span>
          </div>
        </div>

        {selected === "gnb" && (
          <div className="mt-4 rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-500/20 dark:bg-green-500/10">
            <div className="flex items-center gap-2 text-xs font-medium text-green-700 dark:text-green-300">
              <Info className="h-4 w-4" />
              科普优势总结
            </div>
            <p className="mt-1 text-xs text-green-600 dark:text-green-400">
              高斯朴素贝叶斯在科普场景下具有<span className="font-bold">绝对优势</span>：
              它的决策过程可以用"钟形曲线 + 概率"完整解释，
              而 SVM、随机森林、神经网络都是黑盒模型，
              无法向公众解释"机器为什么这么判断"。
              <br />
              <br />
              <span className="font-medium">
                科普的核心不是最高精度，而是让人理解"机器如何思考"。
              </span>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
