/**
 * 科普人员工作区 - 恒星分类科普系统
 *
 * 核心功能：
 * 1. 恒星物理参数模拟数据生成（主序星/红巨星/白矮星）
 * 2. 高斯朴素贝叶斯分类器（纯前端实现，无需API）
 * 3. 可视化：高斯分布曲线、赫罗图、概率柱状图
 * 4. 科普交互：特征调节、先验调节、算法对比
 */

import { useState, useMemo, useCallback } from "react";
import { Sparkles, BookOpen, Shuffle, Info, Telescope } from "lucide-react";
import Card from "@/components/Card";
import GaussianCurveChart from "@/components/GaussianCurveChart";
import HRDiagram from "@/components/HRDiagram";
import ProbabilityBars from "@/components/ProbabilityBars";
import FeatureInput from "@/components/FeatureInput";
import FeatureAnalyzer from "@/components/FeatureAnalyzer";
import {
  generateStarDataset,
  generateStarDatasetSync,
  generateStarSample,
  getFeatureDistribution,
  STAR_CLASSES,
  FEATURE_LABELS,
  STAR_COLORS,
  type StarClass,
  type StarParams,
  type StarSample,
} from "@/lib/starSimulator";
import {
  trainGNB,
  classifyGNB,
  generateGaussianCurve as genGaussianCurve,
} from "@/lib/gnbClassifier";

type Tab = "classify" | "visualize" | "compare";

const FEATURE_KEYS: (keyof StarParams)[] = ["temperature", "luminosity", "radius", "mass", "colorIndex"];

const FEATURE_RANGES: Record<keyof StarParams, { min: number; max: number; default: number }> = {
  temperature: { min: 2000, max: 50000, default: 5800 },
  luminosity: { min: 0.001, max: 1000, default: 1.0 },
  radius: { min: 0.005, max: 100, default: 1.0 },
  mass: { min: 0.1, max: 20, default: 1.0 },
  colorIndex: { min: -1, max: 2.5, default: 0.65 },
};

// 典型恒星参数（用于高质量随机）
const TYPICAL_STAR_PARAMS: Record<StarClass, StarParams> = {
  "主序星": { temperature: 5800, luminosity: 1.0, radius: 1.0, mass: 1.0, colorIndex: 0.65 },
  "红巨星": { temperature: 4500, luminosity: 50, radius: 20, mass: 1.2, colorIndex: 1.5 },
  "白矮星": { temperature: 15000, luminosity: 0.01, radius: 0.01, mass: 0.7, colorIndex: -0.3 },
};

export default function EducatorWorkspace() {
  const [activeTab, setActiveTab] = useState<Tab>("classify");
  const [dataset, setDataset] = useState<StarSample[]>([]);
  const [model, setModel] = useState<ReturnType<typeof trainGNB> | null>(null);

  const [features, setFeatures] = useState<StarParams>({
    temperature: 5800,
    luminosity: 1.0,
    radius: 1.0,
    mass: 1.0,
    colorIndex: 0.65,
  });

  const [selectedFeature, setSelectedFeature] = useState<keyof StarParams>("temperature");

  const [loading, setLoading] = useState<boolean>(true);
  const [dataSource, setDataSource] = useState<string>("加载中...");

  const initializeSystem = useCallback(async () => {
    setLoading(true);
    try {
      // 使用增强的 SDSS 数据（含真实主序星 + 模拟红巨星和白矮星）
      const samples = await generateStarDataset(500, { useSdss: true });
      setDataset(samples);
      const trainedModel = trainGNB(samples, FEATURE_KEYS);
      setModel(trainedModel);
      setDataSource('增强 SDSS 数据（真实主序星 + 模拟红巨星/白矮星）');
    } catch (error) {
      console.error('Failed to initialize system:', error);
      // 失败时使用模拟数据
      const samples = generateStarDatasetSync(500);
      setDataset(samples);
      const trainedModel = trainGNB(samples, FEATURE_KEYS);
      setModel(trainedModel);
      setDataSource('模拟数据');
    } finally {
      setLoading(false);
    }
  }, []);

  useState(() => {
    if (!model) {
      initializeSystem();
    }
  });

  const classificationResult = useMemo(() => {
    if (!model) return null;
    return classifyGNB(model, features);
  }, [model, features]);

  const gaussianCurveData = useMemo(() => {
    const result: Record<StarClass, Array<{ x: number; y: number }>> = {
      主序星: [],
      红巨星: [],
      白矮星: [],
    };

    const { min, max } = FEATURE_RANGES[selectedFeature];

    for (const starClass of STAR_CLASSES) {
      const { mean, std } = getFeatureDistribution(selectedFeature, starClass);
      result[starClass] = genGaussianCurve(mean, std, min, max, 100);
    }

    return result;
  }, [selectedFeature]);

  const hrDiagramData = useMemo(() => {
    return dataset.slice(0, 200);
  }, [dataset]);

  const handleRandomize = () => {
    // 随机选择一个星类型（按真实比例）
    const starClasses: StarClass[] = ["主序星", "红巨星", "白矮星"];
    const priors = { "主序星": 0.9, "红巨星": 0.08, "白矮星": 0.02 };
    const r = Math.random();
    let cumProb = 0;
    let selectedClass: StarClass = "主序星";
    for (let i = 0; i < starClasses.length; i++) {
      cumProb += priors[starClasses[i]];
      if (r < cumProb) {
        selectedClass = starClasses[i];
        break;
      }
    }

    // 获取该类型的典型参数
    const typical = TYPICAL_STAR_PARAMS[selectedClass];

    // 生成一个在该类型典型值附近的高质量样本
    const sample = generateStarSample(selectedClass);

    setFeatures({
      temperature: sample.temperature,
      luminosity: sample.luminosity,
      radius: sample.radius,
      mass: sample.mass,
      colorIndex: sample.colorIndex,
    });
  };

  const handleRegenerate = () => {
    initializeSystem();
  };



  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: "classify", label: "交互分类", icon: <Sparkles className="h-4 w-4" /> },
    { id: "visualize", label: "可视化原理", icon: <BookOpen className="h-4 w-4" /> },
  ];

  return (
    <div className="mx-auto max-w-[1400px] px-6 py-6">
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
              恒星分类科普系统
            </h1>
            <p className="mt-1 text-sm text-slate-600 dark:text-white/60">
              基于高斯朴素贝叶斯 · {dataSource} · 恒星物理参数分析
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-800 dark:bg-green-900/30 dark:text-green-400">
              {dataSource}
            </span>
            <button
              onClick={handleRegenerate}
              disabled={loading}
              className="flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm text-slate-600 transition-colors hover:bg-slate-100 dark:border-white/10 dark:text-white/60 dark:hover:bg-white/5 disabled:opacity-50 disabled:pointer-events-none"
            >
              <Shuffle className="h-4 w-4" />
              {loading ? "加载中..." : "重置数据"}
            </button>
          </div>
        </div>
      </div>
      
      {loading ? (
        <div className="flex min-h-[400px] items-center justify-center rounded-xl border border-slate-200 bg-white p-8 dark:border-white/10 dark:bg-slate-800">
          <div className="text-center">
            <div className="inline-block animate-spin rounded-full border-2 border-slate-200 border-t-blue-600 p-4 dark:border-slate-700 dark:border-t-blue-400"></div>
            <p className="mt-4 text-sm text-slate-600 dark:text-slate-400">
              正在加载真实 SDSS 数据...
            </p>
          </div>
        </div>
      ) : model ? (
        <>
          <section className="mb-6 p-4 rounded-lg bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-100 dark:border-blue-800/50">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                <Telescope className="w-4 h-4" />
              </div>
              <h2 className="text-lg font-semibold text-blue-800 dark:text-blue-300">恒星分类科普</h2>
            </div>
            <p className="text-sm text-blue-700 dark:text-blue-400">
              高斯朴素贝叶斯是一种基于概率的分类算法，特别适合描述恒星物理参数的正态分布。
              通过调整恒星的温度、光度、半径等参数，观察分类结果的变化，了解恒星演化的基本规律。
            </p>
          </section>
          <div className="mb-6 flex gap-1 rounded-lg border border-slate-200 p-1 dark:border-white/10 dark:bg-white/5 w-fit">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                  activeTab === tab.id
                    ? "bg-blue-500 text-white shadow"
                    : "text-slate-600 hover:bg-slate-100 dark:text-white/60 dark:hover:bg-white/5"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>

      {activeTab === "classify" && (
        <div className="space-y-6">
          <Card title="特征分析器 - 探索每个特征如何影响恒星分类">
            <FeatureAnalyzer />
          </Card>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[350px_1fr]">
            <div className="space-y-5">
              <Card title="输入恒星参数">
                <FeatureInput
                  features={features}
                  onChange={setFeatures}
                  onRandomize={handleRandomize}
                />
              </Card>
            </div>

            <div className="space-y-5">
              <Card title="分类结果">
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    {STAR_CLASSES.map((starClass) => {
                      const prob = classificationResult?.probabilities[starClass] ?? 0;
                      const isPredicted = starClass === classificationResult?.class;
                      return (
                        <div
                          key={starClass}
                          className={`rounded-xl border-2 p-4 text-center transition-all ${
                            isPredicted
                              ? "border-current shadow-lg"
                              : "border-slate-200 dark:border-white/10"
                          }`}
                          style={{
                            borderColor: isPredicted ? STAR_COLORS[starClass] : undefined,
                            backgroundColor: isPredicted
                              ? `${STAR_COLORS[starClass]}15`
                              : undefined,
                          }}
                        >
                          <div
                            className="mb-2 text-2xl font-bold"
                            style={{ color: STAR_COLORS[starClass] }}
                          >
                            {(prob * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm font-medium text-slate-700 dark:text-white">
                            {starClass}
                          </div>
                          {isPredicted && (
                            <div className="mt-2 text-xs text-green-600 dark:text-green-400">
                              ← 判定结果
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {classificationResult && (
                    <div className="rounded-lg bg-slate-50 p-4 dark:bg-white/5">
                      <div className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-white">
                        <Info className="h-4 w-4 text-blue-500" />
                        分类依据
                      </div>
                      <p className="text-sm text-slate-600 dark:text-white/60">
                        输入恒星被判定为
                        <span
                          className="mx-1 font-bold"
                          style={{ color: STAR_COLORS[classificationResult.class] }}
                        >
                          {classificationResult.class}
                        </span>
                        ，概率为{" "}
                        <span className="font-bold">
                          {(classificationResult.probabilities[classificationResult.class] * 100).toFixed(1)}%
                        </span>
                        。这是因为该恒星的光度
                        <span className="font-mono">
                          {features.luminosity.toFixed(3)} L☉
                        </span>
                        和半径
                        <span className="font-mono">
                          {features.radius.toFixed(3)} R☉
                        </span>
                        与
                        <span
                          className="mx-1 font-bold"
                          style={{ color: STAR_COLORS[classificationResult.class] }}
                        >
                          {classificationResult.class}
                        </span>
                        的典型分布高度吻合。
                      </p>
                    </div>
                  )}
                </div>
              </Card>

              <Card title="概率分布">
                {classificationResult && (
                  <ProbabilityBars
                    probabilities={classificationResult.probabilities}
                    predictedClass={classificationResult.class}
                  />
                )}
              </Card>
            </div>
          </div>
        </div>
      )}

      {activeTab === "visualize" && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="赫罗图 (H-R Diagram)">
              <HRDiagram samples={hrDiagramData} />
              <div className="mt-4 space-y-2 rounded-lg bg-purple-50 p-3 dark:bg-purple-500/10">
                <div className="text-xs font-medium text-purple-700 dark:text-purple-300">
                  🌟 赫罗图科普
                </div>
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  <strong>主序星</strong>：一条从左上（高温高光度）到右下（低温低光度）的带状区域。
                  太阳就是这个带上的一个点。
                </p>
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  <strong>红巨星</strong>：在右上角，低温但高光度（因为半径大）。
                  已经离开主序带，演化到后期阶段。
                </p>
                <p className="text-xs text-purple-600 dark:text-purple-400">
                  <strong>白矮星</strong>：在左下角，高温但低光度（因为体积小）。
                  是恒星演化的最终产物之一。
                </p>
              </div>
            </Card>

            <Card title="高斯分布曲线">
              <div className="mb-4">
                <div className="text-xs text-slate-600 dark:text-white/50 mb-2">
                  选择特征：
                </div>
                <div className="flex flex-wrap gap-2">
                  {FEATURE_KEYS.map((key) => (
                    <button
                      key={key}
                      onClick={() => setSelectedFeature(key)}
                      className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-all ${
                        selectedFeature === key
                          ? "border-blue-500 bg-blue-500/10 text-blue-600 dark:text-blue-400"
                          : "border-slate-200 text-slate-600 dark:border-white/10 dark:text-white/60 dark:hover:bg-white/5"
                      }`}
                    >
                      {FEATURE_LABELS[key]}
                    </button>
                  ))}
                </div>
              </div>
              <GaussianCurveChart
                data={gaussianCurveData}
                featureLabel={FEATURE_LABELS[selectedFeature]}
                selectedFeature={selectedFeature}
              />
              <div className="mt-4 rounded-lg bg-blue-50 p-3 dark:bg-blue-500/10">
                <div className="text-xs font-medium text-blue-700 dark:text-blue-300">
                  📊 科普要点
                </div>
                <p className="mt-1 text-xs text-blue-600 dark:text-blue-400">
                  同类恒星的光度集中在平均值附近（钟形曲线中间），极端值很少（曲线两端）。
                  这就是为什么可以用高斯分布来描述恒星物理参数。
                  {selectedFeature === "luminosity" &&
                    " 红巨星的光度远高于主序星和白矮星，在右尾分布。"}
                  {selectedFeature === "temperature" &&
                    " 白矮星温度最高（表面灼热但光度低），主序星次之，红巨星最低。"}
                </p>
              </div>
            </Card>
          </div>
        </div>
      )}
        </>
      ) : null}
    </div>
  );
}
