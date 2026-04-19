/**
 * 高斯朴素贝叶斯分类器 - 纯 TypeScript 实现
 * 用于恒星分类科普场景 - 修复概率极端化问题
 */

import type { StarClass, StarParams } from "./starSimulator";

export interface GNBParams {
  priors?: Record<StarClass, number>;
}

export interface ClassStats {
  mean: Record<keyof StarParams, number>;
  std: Record<keyof StarParams, number>;
  prior: number;
}

export interface GNBModel {
  classes: StarClass[];
  stats: Record<StarClass, ClassStats>;
  featureNames: (keyof StarParams)[];
  featureScales: Record<keyof StarParams, { mean: number; std: number }>;
}

/**
 * 计算高斯分布概率密度
 */
function gaussianPDF(x: number, mean: number, std: number): number {
  if (std === 0) return x === mean ? Infinity : 0;
  const coefficient = 1 / (std * Math.sqrt(2 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mean) / std, 2);
  return coefficient * Math.exp(exponent);
}

/**
 * 训练高斯朴素贝叶斯模型
 */
export function trainGNB(
  samples: Array<StarParams & { class: StarClass }>,
  featureNames: (keyof StarParams)[],
  priors?: Record<StarClass, number>
): GNBModel {
  const classes: StarClass[] = [...new Set(samples.map(s => s.class))];
  const stats: Record<StarClass, ClassStats> = {} as Record<StarClass, ClassStats>;

  // 特征标准化 - 基于所有数据
  const featureScales: Record<keyof StarParams, { mean: number; std: number }> = {} as any;
  for (const f of featureNames) {
    const vals = samples.map(s => s[f] as number);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length);
    featureScales[f] = { mean, std: std || 1 };
  }

  // 标准化样本
  const normalizedSamples = samples.map(sample => {
    const normalized: any = { class: sample.class };
    for (const f of featureNames) {
      const { mean, std } = featureScales[f];
      normalized[f] = ((sample[f] as number) - mean) / std;
    }
    return normalized as StarParams & { class: StarClass };
  });

  const classCounts: Record<StarClass, number> = {} as Record<StarClass, number>;
  const classSums: Record<StarClass, Record<string, number>> = {} as Record<StarClass, Record<string, number>>;
  const classSquaredSums: Record<StarClass, Record<string, number>> = {} as Record<StarClass, Record<string, number>>;

  for (const c of classes) {
    classCounts[c] = 0;
    classSums[c] = {};
    classSquaredSums[c] = {};
    for (const f of featureNames) {
      classSums[c][f] = 0;
      classSquaredSums[c][f] = 0;
    }
  }

  for (const sample of normalizedSamples) {
    const c = sample.class;
    classCounts[c]++;
    for (const f of featureNames) {
      const val = sample[f] as number;
      classSums[c][f] += val;
      classSquaredSums[c][f] += val * val;
    }
  }

  const total = normalizedSamples.length;

  for (const c of classes) {
    const count = classCounts[c];
    const mean: Record<string, number> = {};
    const std: Record<string, number> = {};

    for (const f of featureNames) {
      mean[f] = classSums[c][f] / count;
      const variance = (classSquaredSums[c][f] / count) - mean[f] * mean[f];
      std[f] = Math.sqrt(Math.max(variance, 0.01)); // 更大的最小方差，避免过于自信
    }

    stats[c] = {
      mean: mean as Record<keyof StarParams, number>,
      std: std as Record<keyof StarParams, number>,
      prior: priors?.[c] ?? count / total,
    };
  }

  return { classes, stats, featureNames, featureScales };
}

/**
 * 预测单颗恒星的分类概率
 */
export function predictGNB(
  model: GNBModel,
  star: StarParams,
  priorAdjustments?: Partial<Record<StarClass, number>>
): Record<StarClass, { probability: number; likelihood: number }> {
  const results: Record<StarClass, { probability: number; likelihood: number }> = {} as Record<StarClass, { probability: number; likelihood: number }>;

  // 标准化输入
  const normalizedStar: StarParams = {} as StarParams;
  for (const f of model.featureNames) {
    const { mean, std } = model.featureScales[f];
    normalizedStar[f] = ((star[f] as number) - mean) / std;
  }

  const logLikelihoods: Record<StarClass, number> = {} as Record<StarClass, number>;
  let maxLogL = -Infinity;

  for (const c of model.classes) {
    const stats = model.stats[c];
    let logL = Math.log(stats.prior * (priorAdjustments?.[c] ?? 1));

    // 使用部分特征（只选温度、光度、颜色指数），避免 5 个特征相乘导致指数级差异
    const selectedFeatures = model.featureNames.filter(f => ['temperature', 'luminosity', 'colorIndex'].includes(f));
    for (const f of selectedFeatures) {
      const val = normalizedStar[f] as number;
      const mean = stats.mean[f];
      const std = stats.std[f];
      const pdf = gaussianPDF(val, mean, std);
      logL += Math.log(Math.max(pdf, 1e-10));
    }

    logLikelihoods[c] = logL;
    if (logL > maxLogL) maxLogL = logL;
  }

  // 先计算未校准的归一化，保证概率和为1
  let unnormalizedSum = 0;
  for (const c of model.classes) {
    unnormalizedSum += Math.exp(logLikelihoods[c] - maxLogL);
  }

  // 概率校准 - 温度调整，避免过于极端
  const temperature = 1.5;
  
  for (const c of model.classes) {
    const prob = Math.exp(logLikelihoods[c] - maxLogL) / unnormalizedSum;
    // 温度校准
    const calibratedProb = Math.pow(prob, 1 / temperature);
    results[c] = {
      probability: calibratedProb,
      likelihood: Math.exp(logLikelihoods[c] - maxLogL),
    };
  }

  // 归一化校准后的概率
  let total = 0;
  for (const c of model.classes) {
    total += results[c].probability;
  }
  for (const c of model.classes) {
    results[c].probability /= total;
  }

  return results;
}

/**
 * 预测单颗恒星的最终分类
 */
export function classifyGNB(
  model: GNBModel,
  star: StarParams,
  priorAdjustments?: Partial<Record<StarClass, number>>
): { class: StarClass; probabilities: Record<StarClass, number> } {
  const probs = predictGNB(model, star, priorAdjustments);

  let maxProb = 0;
  let predictedClass: StarClass = model.classes[0];

  for (const c of model.classes) {
    if (probs[c].probability > maxProb) {
      maxProb = probs[c].probability;
      predictedClass = c;
    }
  }

  const probabilities: Record<StarClass, number> = {} as Record<StarClass, number>;
  for (const c of model.classes) {
    probabilities[c] = probs[c].probability;
  }

  return { class: predictedClass, probabilities };
}

/**
 * 计算模型准确率（用于对比演示）
 */
export function evaluateModel(
  model: GNBModel,
  samples: Array<StarParams & { class: StarClass }>,
  priorAdjustments?: Partial<Record<StarClass, number>>
): { accuracy: number; confusionMatrix: number[][]; labels: StarClass[] } {
  const labels = model.classes;
  const confusionMatrix: number[][] = labels.map(() => labels.map(() => 0));

  let correct = 0;

  for (const sample of samples) {
    const { class: predicted } = classifyGNB(model, sample, priorAdjustments);
    const actual = sample.class;
    const actualIdx = labels.indexOf(actual);
    const predIdx = labels.indexOf(predicted);
    if (actualIdx >= 0 && predIdx >= 0) {
      confusionMatrix[actualIdx][predIdx]++;
    }
    if (predicted === actual) correct++;
  }

  return {
    accuracy: correct / samples.length,
    confusionMatrix,
    labels,
  };
}

/**
 * 生成用于可视化的高斯曲线数据点
 */
export function generateGaussianCurve(
  mean: number,
  std: number,
  rangeMin: number,
  rangeMax: number,
  steps: number = 100
): Array<{ x: number; y: number }> {
  const points: Array<{ x: number; y: number }> = [];
  const step = (rangeMax - rangeMin) / steps;

  for (let i = 0; i <= steps; i++) {
    const x = rangeMin + i * step;
    const y = gaussianPDF(x, mean, std);
    points.push({ x, y });
  }

  return points;
}
