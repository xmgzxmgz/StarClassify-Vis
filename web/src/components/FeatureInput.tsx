/**
 * 恒星特征输入面板
 * 手动输入恒星物理参数，或随机生成
 */

import { useState } from "react";
import { RefreshCw } from "lucide-react";
import type { StarParams, StarClass } from "@/lib/starSimulator";
import { FEATURE_LABELS } from "@/lib/starSimulator";
import { generateStarSample } from "@/lib/starSimulator";

interface FeatureInputProps {
  features: StarParams;
  onChange: (features: StarParams) => void;
  onRandomize: () => void;
}

const FEATURE_RANGES: Record<keyof StarParams, { min: number; max: number; step: number }> = {
  temperature: { min: 2000, max: 50000, step: 100 },
  luminosity: { min: 0.001, max: 1000, step: 0.1 },
  radius: { min: 0.005, max: 100, step: 0.1 },
  mass: { min: 0.1, max: 20, step: 0.1 },
  colorIndex: { min: -1, max: 2.5, step: 0.05 },
};

const STAR_CLASS_EXAMPLES: Record<StarClass, Partial<StarParams>> = {
  "主序星": { temperature: 5800, luminosity: 1.0, radius: 1.0, mass: 1.0, colorIndex: 0.65 },
  "红巨星": { temperature: 4500, luminosity: 50, radius: 20, mass: 1.2, colorIndex: 1.5 },
  "白矮星": { temperature: 15000, luminosity: 0.01, radius: 0.01, mass: 0.7, colorIndex: -0.3 },
};

export default function FeatureInput({
  features,
  onChange,
  onRandomize,
}: FeatureInputProps) {
  const [activeExample, setActiveExample] = useState<StarClass | null>(null);

  const handleFeatureChange = (key: keyof StarParams, value: number) => {
    onChange({ ...features, [key]: value });
    setActiveExample(null);
  };

  const handleExampleClick = (starClass: StarClass) => {
    const example = STAR_CLASS_EXAMPLES[starClass];
    onChange({
      temperature: example.temperature ?? features.temperature,
      luminosity: example.luminosity ?? features.luminosity,
      radius: example.radius ?? features.radius,
      mass: example.mass ?? features.mass,
      colorIndex: example.colorIndex ?? features.colorIndex,
    });
    setActiveExample(starClass);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700 dark:text-white/80">
          恒星物理参数
        </span>
        <button
          onClick={onRandomize}
          className="flex items-center gap-1 rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 transition-colors hover:bg-slate-100 dark:border-white/10 dark:text-white/60 dark:hover:bg-white/5"
        >
          <RefreshCw className="h-3 w-3" />
          随机生成
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3">
        {(Object.keys(FEATURE_RANGES) as (keyof StarParams)[]).map((key) => {
          const range = FEATURE_RANGES[key];
          const displayValue =
            key === "luminosity" || key === "radius"
              ? features[key].toFixed(3)
              : key === "colorIndex"
                ? features[key].toFixed(2)
                : Math.round(features[key]).toString();

          return (
            <div key={key} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-600 dark:text-white/50">
                  {FEATURE_LABELS[key]}
                </span>
                <span className="font-mono text-slate-800 dark:text-white">
                  {displayValue}
                </span>
              </div>
              <input
                type="range"
                min={range.min}
                max={range.max}
                step={range.step}
                value={features[key]}
                onChange={(e) =>
                  handleFeatureChange(key, parseFloat(e.target.value))
                }
                className="w-full"
                style={{ accentColor: "#3B82F6" }}
              />
            </div>
          );
        })}
      </div>

      <div className="space-y-2">
        <div className="text-xs text-slate-500 dark:text-white/40">
          快速示例（点击填充典型值）：
        </div>
        <div className="flex gap-2">
          {(["主序星", "红巨星", "白矮星"] as StarClass[]).map((starClass) => (
            <button
              key={starClass}
              onClick={() => handleExampleClick(starClass)}
              className={`flex-1 rounded-lg border px-2 py-1.5 text-xs font-medium transition-all ${
                activeExample === starClass
                  ? "border-blue-500 bg-blue-500/10 text-blue-600 dark:text-blue-400"
                  : "border-slate-200 text-slate-600 hover:border-slate-300 dark:border-white/10 dark:text-white/60 dark:hover:border-white/20"
              }`}
            >
              {starClass}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
