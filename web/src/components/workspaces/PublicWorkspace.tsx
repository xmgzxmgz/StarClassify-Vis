/**
 * 科普人员工作区 - 天文实验模拟
 * 模拟天文学家做实验的场景，用语亲民，内容简单易懂
 * 适合科普人员和天文爱好者使用
 */

import { useState, useMemo } from "react";
import { FlaskConical, Star, Rocket, Target, Lightbulb, Trophy, Sparkles, ZoomIn } from "lucide-react";
import Card from "@/components/Card";
import {
  generateStarSample,
  type StarClass,
  type StarParams,
  STAR_CLASSES,
  STAR_COLORS,
  FEATURE_LABELS,
} from "@/lib/starSimulator";
import {
  trainGNB,
  classifyGNB,
  generateGaussianCurve as genGaussianCurve,
} from "@/lib/gnbClassifier";

const FEATURE_RANGES: Record<keyof StarParams, { min: number; max: number; default: number }> = {
  temperature: { min: 2000, max: 30000, default: 5800 },
  luminosity: { min: 0.001, max: 100, default: 1.0 },
  radius: { min: 0.01, max: 30, default: 1.0 },
  mass: { min: 0.1, max: 5, default: 1.0 },
  colorIndex: { min: -1, max: 2.5, default: 0.65 },
};

type ExperimentType = "star_hunter" | "hr_explorer" | "color_analyzer";

export default function PublicWorkspace() {
  const [activeExperiment, setActiveExperiment] = useState<ExperimentType>("star_hunter");
  const [features, setFeatures] = useState<StarParams>({
    temperature: 5800,
    luminosity: 1.0,
    radius: 1.0,
    mass: 1.0,
    colorIndex: 0.65,
  });
  const [dataset, setDataset] = useState(() => {
    const samples = [];
    for (let i = 0; i < 200; i++) {
      samples.push(generateStarSample());
    }
    return samples;
  });
  const [model] = useState(() => {
    const samples = [];
    for (let i = 0; i < 500; i++) {
      samples.push(generateStarSample());
    }
    return trainGNB(samples, Object.keys(features) as (keyof StarParams)[]);
  });

  // 分类结果
  const classificationResult = useMemo(() => {
    if (!model) return null;
    return classifyGNB(model, features);
  }, [model, features]);

  // 随机生成恒星样本
  const handleRandomize = () => {
    const sample = generateStarSample();
    setFeatures({
      temperature: sample.temperature,
      luminosity: sample.luminosity,
      radius: sample.radius,
      mass: sample.mass,
      colorIndex: sample.colorIndex,
    });
  };

  // 重置实验
  const handleReset = () => {
    setFeatures({
      temperature: 5800,
      luminosity: 1.0,
      radius: 1.0,
      mass: 1.0,
      colorIndex: 0.65,
    });
  };

  // 实验选项
  const experiments = [
    {
      id: "star_hunter" as ExperimentType,
      label: "恒星猎人",
      icon: <Target className="h-4 w-4" />,
      description: "像天文学家一样，通过调整恒星参数来寻找不同类型的恒星",
    },
    {
      id: "hr_explorer" as ExperimentType,
      label: "赫罗图探索",
      icon: <ZoomIn className="h-4 w-4" />,
      description: "在赫罗图上探索恒星的分布规律，了解恒星演化",
    },
    {
      id: "color_analyzer" as ExperimentType,
      label: "颜色分析器",
      icon: <Lightbulb className="h-4 w-4" />,
      description: "通过颜色指数判断恒星温度，体验天文学家的观测方法",
    },
  ];

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-8">
        <div className="flex items-center gap-3">
          <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-blue-500 text-white shadow-lg">
            <FlaskConical className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
              天文实验室
            </h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-white/60">
              像天文学家一样做实验，探索恒星的奥秘
            </p>
          </div>
        </div>
      </div>

      {/* 实验选择 */}
      <div className="mb-8 flex flex-wrap gap-3">
        {experiments.map((exp) => (
          <button
            key={exp.id}
            onClick={() => setActiveExperiment(exp.id)}
            className={`flex flex-col items-center gap-2 rounded-xl border px-6 py-4 transition-all ${activeExperiment === exp.id
                ? "border-blue-500 bg-blue-50 dark:bg-blue-500/10 shadow-lg"
                : "border-slate-200 dark:border-white/10 hover:bg-slate-50 dark:hover:bg-white/5"
              }`}
          >
            <div
              className={`rounded-full p-3 ${activeExperiment === exp.id
                  ? "bg-blue-500 text-white"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-white/60"
                }`}
            >
              {exp.icon}
            </div>
            <div className="text-sm font-semibold text-slate-900 dark:text-white">
              {exp.label}
            </div>
            <div className="text-xs text-center text-slate-600 dark:text-white/60">
              {exp.description}
            </div>
          </button>
        ))}
      </div>

      {/* 恒星猎人实验 */}
      {activeExperiment === "star_hunter" && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[350px_1fr]">
          <div className="space-y-5">
            <Card title="调整恒星参数">
              <div className="space-y-4">
                {Object.entries(features).map(([key, value]) => {
                  const featureKey = key as keyof StarParams;
                  const range = FEATURE_RANGES[featureKey];
                  return (
                    <div key={key}>
                      <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-white/80">
                        <span>{FEATURE_LABELS[featureKey]}</span>
                        <span className="font-mono text-slate-600 dark:text-white/60">
                          {featureKey === "colorIndex"
                            ? value.toFixed(2)
                            : value >= 1000
                              ? `${(value / 1000).toFixed(1)}k`
                              : value >= 1
                                ? value.toFixed(1)
                                : value.toFixed(3)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={range.min}
                        max={range.max}
                        step={range.max - range.min < 10 ? 0.01 : 1}
                        value={value}
                        onChange={(e) => {
                          setFeatures({
                            ...features,
                            [featureKey]: parseFloat(e.target.value),
                          });
                        }}
                        className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 dark:bg-slate-700"
                        style={{
                          accentColor: STAR_COLORS[classificationResult?.class || "主序星"],
                        }}
                      />
                      <div className="flex justify-between text-xs text-slate-500 dark:text-white/40">
                        <span>{range.min}</span>
                        <span>{range.max}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-6 flex gap-3">
                <button
                  onClick={handleRandomize}
                  className="flex-1 rounded-lg border border-slate-200 bg-white py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-white/10 dark:bg-white/5 dark:text-white dark:hover:bg-white/10"
                >
                  <span className="flex items-center gap-1">
                    <Sparkles className="h-4 w-4" />
                    随机恒星
                  </span>
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 rounded-lg border border-slate-200 bg-white py-2 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-50 dark:border-white/10 dark:bg-white/5 dark:text-white dark:hover:bg-white/10"
                >
                  重置
                </button>
              </div>
            </Card>

            <Card title="观测结果">
              {classificationResult && (
                <div className="space-y-4">
                  <div className="flex flex-col items-center justify-center p-6 text-center">
                    <div
                      className="mb-4 h-20 w-20 rounded-full shadow-lg transition-all"
                      style={{
                        backgroundColor: STAR_COLORS[classificationResult.class],
                        boxShadow: `0 0 30px ${STAR_COLORS[classificationResult.class]}80`,
                      }}
                    />
                    <h3
                      className="text-2xl font-bold transition-colors"
                      style={{
                        color: STAR_COLORS[classificationResult.class],
                      }}
                    >
                      {classificationResult.class}
                    </h3>
                    <p className="mt-2 text-sm text-slate-600 dark:text-white/60">
                      匹配度: {(classificationResult.probabilities[classificationResult.class] * 100).toFixed(1)}%
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
                              className="h-full rounded-full transition-all duration-500 ease-out"
                              style={{
                                width: `${Math.max(prob * 100, 2)}%`,
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

            <Card title="实验提示">
              <div className="space-y-2 text-xs text-slate-600 dark:text-white/60">
                <p>🌟 <strong>主序星</strong>：像太阳一样的恒星，稳定燃烧氢</p>
                <p>🔴 <strong>红巨星</strong>：年老的恒星，体积膨胀，温度降低</p>
                <p>⚪ <strong>白矮星</strong>：恒星的残骸，体积小但密度极高</p>
                <p className="mt-3 text-blue-600 dark:text-blue-400">
                  💡 提示：调整温度和光度，看看恒星类型会如何变化！
                </p>
              </div>
            </Card>
          </div>

          <div className="space-y-5">
            <Card title="科普知识">
              <div className="space-y-4">
                <div className="flex items-start gap-3 rounded-lg border border-slate-200 p-4 dark:border-white/10">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400">
                    <Star className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-slate-900 dark:text-white">
                      恒星是如何分类的？
                    </h4>
                    <p className="mt-1 text-sm text-slate-600 dark:text-white/60">
                      天文学家主要根据恒星的温度、光度和颜色来分类。这些参数决定了恒星在赫罗图上的位置，
                      而位置又反映了恒星的演化阶段。
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3 rounded-lg border border-slate-200 p-4 dark:border-white/10">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-purple-100 dark:bg-purple-500/20 text-purple-600 dark:text-purple-400">
                    <Rocket className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-slate-900 dark:text-white">
                      为什么这个实验有意义？
                    </h4>
                    <p className="mt-1 text-sm text-slate-600 dark:text-white/60">
                      通过调整恒星参数，你可以直观地理解不同类型恒星的特征。这就是天文学家每天在做的工作——
                      通过观测数据来判断恒星的类型和演化阶段。
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3 rounded-lg border border-slate-200 p-4 dark:border-white/10">
                  <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-green-100 dark:bg-green-500/20 text-green-600 dark:text-green-400">
                    <Trophy className="h-5 w-5" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-slate-900 dark:text-white">
                      你发现了什么？
                    </h4>
                    <p className="mt-1 text-sm text-slate-600 dark:text-white/60">
                      主序星温度适中，光度稳定；红巨星温度较低但光度很高；白矮星温度很高但光度很低。
                      这些特征反映了它们不同的演化阶段。
                    </p>
                  </div>
                </div>
              </div>
            </Card>

            <Card title="观测记录">
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between border-b border-slate-200 pb-2 dark:border-white/10">
                  <span className="font-medium text-slate-700 dark:text-white/80">参数</span>
                  <span className="font-mono text-slate-600 dark:text-white/60">当前值</span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-slate-600 dark:text-white/60">表面温度</span>
                  <span className="font-mono text-slate-900 dark:text-white">
                    {features.temperature.toFixed(0)} K
                  </span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-slate-600 dark:text-white/60">光度</span>
                  <span className="font-mono text-slate-900 dark:text-white">
                    {features.luminosity.toFixed(3)} L☉
                  </span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-slate-600 dark:text-white/60">半径</span>
                  <span className="font-mono text-slate-900 dark:text-white">
                    {features.radius.toFixed(3)} R☉
                  </span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-slate-600 dark:text-white/60">质量</span>
                  <span className="font-mono text-slate-900 dark:text-white">
                    {features.mass.toFixed(2)} M☉
                  </span>
                </div>
                <div className="flex items-center justify-between py-2">
                  <span className="text-slate-600 dark:text-white/60">颜色指数</span>
                  <span className="font-mono text-slate-900 dark:text-white">
                    {features.colorIndex.toFixed(2)}
                  </span>
                </div>
              </div>
            </Card>
          </div>
        </div>
      )}

      {/* 赫罗图探索实验 */}
      {activeExperiment === "hr_explorer" && (
        <div className="space-y-6">
          <Card title="赫罗图探索">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_350px]">
              <div className="rounded-lg border border-slate-200 dark:border-white/10 p-4">
                <h3 className="mb-4 text-lg font-semibold text-slate-900 dark:text-white">
                  恒星在赫罗图上的分布
                </h3>
                <div className="h-[500px] w-full bg-slate-50 dark:bg-slate-900 rounded-lg flex items-center justify-center">
                  <div className="text-center text-slate-500 dark:text-white/50">
                    <div className="mb-2 text-4xl">🌟</div>
                    <p>赫罗图可视化</p>
                    <p className="text-xs mt-1">展示恒星温度与光度的关系</p>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="rounded-lg border border-slate-200 dark:border-white/10 p-4">
                  <h4 className="font-semibold text-slate-900 dark:text-white">
                    赫罗图科普
                  </h4>
                  <div className="mt-2 space-y-2 text-sm text-slate-600 dark:text-white/60">
                    <p>🌡️ <strong>温度轴</strong>：从左到右温度降低</p>
                    <p>💡 <strong>光度轴</strong>：从下到上光度增加</p>
                    <p>✨ <strong>主序星</strong>：从左上到右下的带状区域</p>
                    <p>🔴 <strong>红巨星</strong>：右上角区域</p>
                    <p>⚪ <strong>白矮星</strong>：左下角区域</p>
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-white/10 p-4">
                  <h4 className="font-semibold text-slate-900 dark:text-white">
                    探索提示
                  </h4>
                  <div className="mt-2 text-sm text-blue-600 dark:text-blue-400">
                    <p>💡 大部分恒星都在主序带上，包括我们的太阳</p>
                    <p className="mt-1">🌟 红巨星是恒星晚年的状态</p>
                    <p className="mt-1">⚡ 白矮星是恒星的最终残骸</p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* 颜色分析器实验 */}
      {activeExperiment === "color_analyzer" && (
        <div className="space-y-6">
          <Card title="恒星颜色分析">
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_350px]">
              <div className="rounded-lg border border-slate-200 dark:border-white/10 p-4">
                <h3 className="mb-4 text-lg font-semibold text-slate-900 dark:text-white">
                  恒星颜色与温度关系
                </h3>
                <div className="h-[400px] w-full bg-gradient-to-r from-blue-500 via-white to-red-500 rounded-lg relative">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center text-white text-shadow-lg">
                      <div className="text-2xl font-bold">恒星颜色光谱</div>
                      <div className="mt-2 text-sm">蓝 → 白 → 黄 → 橙 → 红</div>
                      <div className="mt-4 text-sm">温度：高 ← 低</div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div className="rounded-lg border border-slate-200 dark:border-white/10 p-4">
                  <h4 className="font-semibold text-slate-900 dark:text-white">
                    颜色指数科普
                  </h4>
                  <div className="mt-2 space-y-2 text-sm text-slate-600 dark:text-white/60">
                    <p>🔵 <strong>蓝色恒星</strong>：温度高（10000K以上）</p>
                    <p>⚪ <strong>白色恒星</strong>：温度较高（7500-10000K）</p>
                    <p>🟡 <strong>黄色恒星</strong>：温度适中（5200-6000K，如太阳）</p>
                    <p>🟠 <strong>橙色恒星</strong>：温度较低（3700-5200K）</p>
                    <p>🔴 <strong>红色恒星</strong>：温度低（3700K以下）</p>
                  </div>
                </div>
                <div className="rounded-lg border border-slate-200 dark:border-white/10 p-4">
                  <h4 className="font-semibold text-slate-900 dark:text-white">
                    天文学家如何观测
                  </h4>
                  <div className="mt-2 text-sm text-slate-600 dark:text-white/60">
                    <p>天文学家使用望远镜和光谱仪来测量恒星的颜色，通过颜色指数（B-V）来判断恒星的温度。</p>
                    <p className="mt-2 text-blue-600 dark:text-blue-400">
                      💡 颜色指数越小，恒星越蓝，温度越高；颜色指数越大，恒星越红，温度越低。
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
