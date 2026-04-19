/**
 * 先验概率调节器
 * 拖动滑块调整宇宙中恒星比例，观察分类结果变化
 */

import type { StarClass } from "@/lib/starSimulator";
import { STAR_CLASSES, STAR_COLORS } from "@/lib/starSimulator";

interface PriorSliderProps {
  priors: Record<StarClass, number>;
  onChange: (adjustments: Partial<Record<StarClass, number>>) => void;
}

export default function PriorSlider({ priors, onChange }: PriorSliderProps) {
  const multipliers = { 主序星: 1, 红巨星: 1, 白矮星: 1 };

  return (
    <div className="space-y-4">
      <div className="text-sm font-medium text-slate-700 dark:text-white/80">
        调整先验概率（宇宙中恒星比例）
      </div>
      <p className="text-xs text-slate-500 dark:text-white/40">
        拖动滑块改变某类恒星在宇宙中的比例，观察分类结果如何变化
      </p>
      <div className="space-y-3">
        {STAR_CLASSES.map((starClass) => (
          <div key={starClass} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span
                className="flex items-center gap-2"
                style={{ color: STAR_COLORS[starClass] }}
              >
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: STAR_COLORS[starClass] }}
                />
                {starClass}
              </span>
              <span className="font-mono text-slate-600 dark:text-white/60">
                {(priors[starClass] * 100).toFixed(1)}%
              </span>
            </div>
            <input
              type="range"
              min="0.01"
              max="0.99"
              step="0.01"
              value={priors[starClass]}
              onChange={(e) => {
                const newValue = parseFloat(e.target.value);
                const otherClasses = STAR_CLASSES.filter((c) => c !== starClass);
                const otherTotal = otherClasses.reduce((sum, c) => sum + priors[c], 0);
                const remaining = 1 - newValue;
                const scale = otherTotal > 0 ? remaining / otherTotal : 1;

                const newPriors = {
                  [starClass]: newValue,
                } as Partial<Record<StarClass, number>>;

                otherClasses.forEach((c) => {
                  (newPriors as Record<StarClass, number>)[c] = priors[c] * scale;
                });

                onChange(newPriors);
              }}
              className="w-full"
              style={{
                accentColor: STAR_COLORS[starClass],
              }}
            />
          </div>
        ))}
      </div>
      <div className="rounded-lg bg-blue-50 dark:bg-blue-500/10 p-3 text-xs text-blue-700 dark:text-blue-300">
        <strong>贝叶斯原理：</strong>先验概率影响后验概率。当某类恒星在宇宙中更常见时，模型会稍微偏向将其作为分类结果。
      </div>
    </div>
  );
}
