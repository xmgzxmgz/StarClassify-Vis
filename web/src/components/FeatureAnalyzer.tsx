/**
 * 特征分析器 - 交互式恒星分类特征科普
 * 探索每个特征如何影响恒星分类
 */

import { useState, useMemo } from "react";
import {
  Thermometer,
  Sun,
  CircleDot,
  Scale,
  Palette,
  TrendingUp,
  Info,
  ChevronRight,
} from "lucide-react";
import Card from "@/components/Card";
import {
  generateStarSample,
  type StarParams,
  STAR_CLASSES,
  STAR_COLORS,
  FEATURE_LABELS,
  gaussianPDF,
} from "@/lib/starSimulator";
import {
  trainGNB,
  classifyGNB,
} from "@/lib/gnbClassifier";

const FEATURE_RANGES: Record<keyof StarParams, { min: number; max: number }> = {
  temperature: { min: 2000, max: 30000 },
  luminosity: { min: 0.001, max: 100 },
  radius: { min: 0.01, max: 30 },
  mass: { min: 0.1, max: 5 },
  colorIndex: { min: -1, max: 2.5 },
};

const FEATURE_INFO: Record<keyof StarParams, {
  icon: React.ReactNode;
  title: string;
  description: string;
  tip: string;
  unit: string;
  format: (v: number) => string;
}> = {
  temperature: {
    icon: <Thermometer className="h-5 w-5" />,
    title: "表面温度",
    description: "恒星的表面温度决定了它的颜色和光谱类型。温度越高，恒星越蓝；温度越低，恒星越红。",
    tip: "💡 主序星的温度范围最广，从3000K的红矮星到40000K的蓝巨星。白矮星温度虽高，但光度很低。",
    unit: "K",
    format: (v) => `${v.toFixed(0)} K`,
  },
  luminosity: {
    icon: <Sun className="h-5 w-5" />,
    title: "光度",
    description: "恒星的光度表示它辐射的总能量。太阳的光度定为1，其他恒星用相对于太阳的倍数表示。",
    tip: "💡 红巨星虽然表面温度低，但因为体积巨大，光度反而很高。白矮星体积小，光度极低。",
    unit: "L☉",
    format: (v) => `${v.toFixed(3)} L☉`,
  },
  radius: {
    icon: <CircleDot className="h-5 w-5" />,
    title: "半径",
    description: "恒星的半径决定了它的大小。红巨星半径可达太阳的100倍以上，而白矮星只有地球大小。",
    tip: "💡 半径是区分红巨星和白矮星的关键特征。红巨星半径大但密度低，白矮星半径小但密度极高。",
    unit: "R☉",
    format: (v) => `${v.toFixed(3)} R☉`,
  },
  mass: {
    icon: <Scale className="h-5 w-5" />,
    title: "质量",
    description: "恒星的质量决定了它的演化路径和寿命。质量越大，燃烧越剧烈，寿命越短。",
    tip: "💡 太阳质量定为1。红矮星质量小但寿命可达万亿年；蓝巨星质量大但寿命只有几百万年。",
    unit: "M☉",
    format: (v) => `${v.toFixed(2)} M☉`,
  },
  colorIndex: {
    icon: <Palette className="h-5 w-5" />,
    title: "颜色指数",
    description: "颜色指数是恒星蓝色和黄色波段亮度之差。数值越小，恒星越蓝；数值越大，恒星越红。",
    tip: "💡 蓝巨星颜色指数约为-0.3，黄矮星（如太阳）约为0.65，红矮星约为1.5。",
    unit: "B-V",
    format: (v) => v.toFixed(2),
  },
};

const STAR_TYPE_DESCRIPTIONS: Record<string, Record<string, string>> = {
  temperature: {
    主序星: "3000-40000K，范围最广",
    红巨星: "3000-5000K，表面较冷",
    白矮星: "10000-30000K，表面灼热",
  },
  luminosity: {
    主序星: "0.01-1000 L☉，范围适中",
    红巨星: "10-10000 L☉，极其明亮",
    白矮星: "0.001-0.1 L☉，相对暗淡",
  },
  radius: {
    主序星: "0.1-10 R☉，大小适中",
    红巨星: "10-100 R☉，体积巨大",
    白矮星: "0.005-0.02 R☉，体积微小",
  },
  mass: {
    主序星: "0.08-50 M☉，质量各异",
    红巨星: "0.3-10 M☉，中等质量",
    白矮星: "0.1-1.4 M☉，质量受限",
  },
  colorIndex: {
    主序星: "-0.3到1.5，涵盖各色",
    红巨星: "1.0到1.8，偏红",
    白矮星: "-0.5到0.5，偏蓝白",
  },
};

export default function FeatureAnalyzer() {
  const [selectedFeature, setSelectedFeature] = useState<keyof StarParams>("temperature");
  const [currentValue, setCurrentValue] = useState(5800);
  const [model] = useState(() => {
    const samples = [];
    for (let i = 0; i < 500; i++) {
      samples.push(generateStarSample());
    }
    return trainGNB(samples, Object.keys(FEATURE_RANGES) as (keyof StarParams)[]);
  });

  const info = FEATURE_INFO[selectedFeature];

  const classificationResult = useMemo(() => {
    if (!model) return null;
    const features: StarParams = {
      temperature: selectedFeature === "temperature" ? currentValue : 5800,
      luminosity: selectedFeature === "luminosity" ? currentValue : 1.0,
      radius: selectedFeature === "radius" ? currentValue : 1.0,
      mass: selectedFeature === "mass" ? currentValue : 1.0,
      colorIndex: selectedFeature === "colorIndex" ? currentValue : 0.65,
    };
    return classifyGNB(model, features);
  }, [model, selectedFeature, currentValue]);

  const distributions = {
    主序星: {
      temperature: { mean: 5800, std: 2500 },
      luminosity: { mean: 1.0, std: 1.2 },
      radius: { mean: 1.0, std: 0.6 },
      mass: { mean: 1.0, std: 0.6 },
      colorIndex: { mean: 0.65, std: 0.4 },
    },
    红巨星: {
      temperature: { mean: 4500, std: 1500 },
      luminosity: { mean: 30, std: 25 },
      radius: { mean: 15, std: 12 },
      mass: { mean: 1.1, std: 0.5 },
      colorIndex: { mean: 1.2, std: 0.4 },
    },
    白矮星: {
      temperature: { mean: 12000, std: 6000 },
      luminosity: { mean: 0.02, std: 0.015 },
      radius: { mean: 0.015, std: 0.008 },
      mass: { mean: 0.8, std: 0.3 },
      colorIndex: { mean: 0.0, std: 0.35 },
    },
  };

  const currentProbabilityDensities = useMemo(() => {
    const densities: Record<string, number> = {};

    for (const starClass of STAR_CLASSES) {
      const classDist = distributions[starClass as keyof typeof distributions];
      const { mean, std } = classDist[selectedFeature as keyof typeof classDist];
      densities[starClass] = gaussianPDF(currentValue, mean, std);
    }

    return densities;
  }, [selectedFeature, currentValue]);

  const features = Object.keys(FEATURE_RANGES) as (keyof StarParams)[];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {features.map((feature) => {
          const featureInfo = FEATURE_INFO[feature];
          const isSelected = feature === selectedFeature;
          return (
            <button
              key={feature}
              onClick={() => {
                setSelectedFeature(feature);
                const defaults: Record<string, number> = { temperature: 5800, luminosity: 1.0, radius: 1.0, mass: 1.0, colorIndex: 0.65 };
                setCurrentValue(defaults[feature]);
              }}
              className={`flex items-center gap-3 rounded-xl border p-4 transition-all ${
                isSelected
                  ? "border-blue-500 bg-blue-50 dark:bg-blue-500/10 shadow-lg"
                  : "border-slate-200 dark:border-white/10 hover:bg-slate-50 dark:hover:bg-white/5"
              }`}
            >
              <div
                className={`rounded-lg p-2 ${
                  isSelected
                    ? "bg-blue-500 text-white"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-white/60"
                }`}
              >
                {featureInfo.icon}
              </div>
              <div className="text-left">
                <div className={`text-sm font-semibold ${isSelected ? "text-blue-600 dark:text-blue-400" : "text-slate-900 dark:text-white"}`}>
                  {featureInfo.title}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_350px]">
        <div className="space-y-4">
          <Card title={`探索 ${info.title} 对恒星分类的影响`}>
            <div className="mb-6 text-center">
              <div className="inline-flex items-center gap-2 rounded-full bg-blue-100 px-6 py-3 dark:bg-blue-500/20">
                <span className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {info.format(currentValue)}
                </span>
                <span className="text-sm text-blue-600 dark:text-blue-400">{info.unit}</span>
              </div>
            </div>

            <div className="mb-6">
              <input
                type="range"
                min={FEATURE_RANGES[selectedFeature].min}
                max={FEATURE_RANGES[selectedFeature].max}
                step={(FEATURE_RANGES[selectedFeature].max - FEATURE_RANGES[selectedFeature].min) / 100}
                value={currentValue}
                onChange={(e) => setCurrentValue(parseFloat(e.target.value))}
                className="h-3 w-full cursor-pointer appearance-none rounded-full bg-gradient-to-r from-blue-400 via-green-400 to-red-400"
                style={{ accentColor: "#3B82F6" }}
              />
              <div className="mt-2 flex justify-between text-xs text-slate-500">
                <span>{info.format(FEATURE_RANGES[selectedFeature].min)}</span>
                <span>{info.format(FEATURE_RANGES[selectedFeature].max)}</span>
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 dark:border-white/10 dark:bg-slate-900">
              <div className="mb-3 flex items-center justify-between">
                <h4 className="text-sm font-medium text-slate-700 dark:text-white/80">
                  {info.title}分布曲线
                </h4>
                <div className="flex items-center gap-4 text-xs">
                  {STAR_CLASSES.map((sc) => (
                    <div key={sc} className="flex items-center gap-1">
                      <div
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: STAR_COLORS[sc] }}
                      />
                      <span className="text-slate-600 dark:text-white/60">{sc}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-3">
                {STAR_CLASSES.map((starClass) => {
                  const maxDensity = Math.max(...Object.values(currentProbabilityDensities));
                  const density = currentProbabilityDensities[starClass];
                  const barWidth = maxDensity > 0 ? (density / maxDensity) * 100 : 0;

                  return (
                    <div key={starClass} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span
                          className="font-medium"
                          style={{ color: STAR_COLORS[starClass] }}
                        >
                          {starClass}
                        </span>
                        <span className="text-slate-600 dark:text-white/60">
                          概率密度: {density.toFixed(6)}
                        </span>
                      </div>
                      <div className="h-4 w-full overflow-hidden rounded bg-slate-200 dark:bg-slate-700">
                        <div
                          className="h-full rounded transition-all duration-300"
                          style={{
                            width: `${barWidth}%`,
                            backgroundColor: STAR_COLORS[starClass],
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="relative mt-4 h-8 rounded bg-slate-200 dark:bg-slate-700">
                <div
                  className="absolute top-0 h-full w-1 bg-blue-500 shadow-lg"
                  style={{
                    left: `${((currentValue - FEATURE_RANGES[selectedFeature].min) / (FEATURE_RANGES[selectedFeature].max - FEATURE_RANGES[selectedFeature].min)) * 100}%`,
                  }}
                />
              </div>
              <div className="mt-1 text-center text-xs text-slate-500">
                滑块位置标记
              </div>
            </div>

            <div className="mt-4 rounded-lg bg-blue-50 p-4 dark:bg-blue-500/10">
              <div className="mb-2 flex items-center gap-2 text-sm font-medium text-blue-700 dark:text-blue-300">
                <Info className="h-4 w-4" />
                {info.title}科普
              </div>
              <p className="text-sm text-blue-600 dark:text-blue-400">
                {info.description}
              </p>
            </div>
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="分类预测结果">
            {classificationResult && (
              <div className="space-y-4">
                <div className="flex flex-col items-center justify-center p-4">
                  <div
                    className="mb-3 h-16 w-16 rounded-full shadow-lg transition-all"
                    style={{
                      backgroundColor: STAR_COLORS[classificationResult.class],
                      boxShadow: `0 0 30px ${STAR_COLORS[classificationResult.class]}80`,
                    }}
                  />
                  <h3
                    className="text-xl font-bold"
                    style={{ color: STAR_COLORS[classificationResult.class] }}
                  >
                    {classificationResult.class}
                  </h3>
                  <p className="mt-1 text-sm text-slate-600 dark:text-white/60">
                    当前 {info.title}: {info.format(currentValue)}
                  </p>
                </div>

                <div className="space-y-2">
                  {STAR_CLASSES.map((starClass) => {
                    const prob = classificationResult.probabilities[starClass] ?? 0;
                    const isPredicted = starClass === classificationResult.class;
                    return (
                      <div key={starClass} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span
                            className={`font-medium ${
                              isPredicted
                                ? "text-slate-900 dark:text-white"
                                : "text-slate-600 dark:text-white/60"
                            }`}
                          >
                            {starClass}
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
                        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-white/10">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${prob * 100}%`,
                              backgroundColor: STAR_COLORS[starClass],
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </Card>

          <Card title={`${info.title}参考范围`}>
            <div className="space-y-3">
              {STAR_CLASSES.map((starClass) => (
                <div
                  key={starClass}
                  className="flex items-center gap-3 rounded-lg border border-slate-200 p-3 dark:border-white/10"
                >
                  <div
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: STAR_COLORS[starClass] }}
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-slate-900 dark:text-white">
                      {starClass}
                    </div>
                    <div className="text-xs text-slate-600 dark:text-white/60">
                      {STAR_TYPE_DESCRIPTIONS[selectedFeature]?.[starClass] || "典型范围"}
                    </div>
                  </div>
                  <ChevronRight className="h-4 w-4 text-slate-400" />
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <div className="flex items-start gap-3">
              <div className="rounded-full bg-amber-100 p-2 text-amber-600 dark:bg-amber-500/20 dark:text-amber-400">
                <TrendingUp className="h-4 w-4" />
              </div>
              <p className="text-sm text-slate-600 dark:text-white/60">
                {info.tip}
              </p>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
