/**
 * 分类概率柱状图
 * 展示输入恒星属于各类别的概率
 */

import type { StarClass } from "@/lib/starSimulator";
import { STAR_COLORS, STAR_CLASSES } from "@/lib/starSimulator";

interface ProbabilityBarsProps {
  probabilities: Record<StarClass, number>;
  predictedClass?: StarClass;
}

export default function ProbabilityBars({
  probabilities,
  predictedClass,
}: ProbabilityBarsProps) {
  const totalProb = Object.values(probabilities).reduce((sum, p) => sum + p, 0);

  return (
    <div className="space-y-4">
      <div className="text-sm font-medium text-slate-700 dark:text-white/80">
        分类概率
      </div>
      <div className="space-y-3">
        {STAR_CLASSES.map((starClass) => {
          const prob = probabilities[starClass] ?? 0;
          const isPredicted = starClass === predictedClass;
          const barWidth = totalProb > 0 ? (prob / totalProb) * 100 : 0;

          return (
            <div key={starClass} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span
                  className={`font-medium ${
                    isPredicted
                      ? "text-slate-900 dark:text-white"
                      : "text-slate-600 dark:text-white/60"
                  }`}
                >
                  {starClass}
                  {isPredicted && (
                    <span className="ml-2 text-xs text-green-600 dark:text-green-400">
                      ← 判定结果
                    </span>
                  )}
                </span>
                <span
                  className={`font-mono ${
                    isPredicted
                      ? "text-slate-900 dark:text-white"
                      : "text-slate-600 dark:text-white/60"
                  }`}
                >
                  {(prob * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
                <div
                  className="h-full rounded-full transition-all duration-500 ease-out"
                  style={{
                    width: `${Math.min(Math.max(barWidth, 2), 100)}%`,
                    backgroundColor: STAR_COLORS[starClass],
                    boxShadow: isPredicted
                      ? `0 0 10px ${STAR_COLORS[starClass]}80`
                      : "none",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      <div className="text-xs text-slate-500 dark:text-white/40 text-right">
        概率总和: {(totalProb * 100).toFixed(1)}%
      </div>
    </div>
  );
}
