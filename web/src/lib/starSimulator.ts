/**
 * 恒星模拟数据生成器
 * 基于真实天文物理参数，模拟三类恒星的高斯分布特征
 * 主序星、红巨星、白矮星 - 调整参数增加分布重叠
 */

export type StarClass = "主序星" | "红巨星" | "白矮星";

export interface StarParams {
  temperature: number;    // 表面温度 (K)
  luminosity: number;     // 光度 (L☉)
  radius: number;         // 半径 (R☉)
  mass: number;           // 质量 (M☉)
  colorIndex: number;     // 颜色指数 (B-V)
}

export interface StarSample extends StarParams {
  class: StarClass;
}

/**
 * 三类恒星的物理参数分布（高斯分布的均值和标准差）
 * 基于真实天文观测数据的近似值 - 调整标准差增加重叠，避免极端概率
 */
const STAR_DISTRIBUTIONS: Record<StarClass, {
  temperature: [number, number];  // [mean, std] in K
  luminosity: [number, number];   // [mean, std] in L☉
  radius: [number, number];       // [mean, std] in R☉
  mass: [number, number];         // [mean, std] in M☉
  colorIndex: [number, number];  // [mean, std] in B-V
  prior: number;                  // 先验概率（宇宙中的比例）
}> = {
  主序星: {
    temperature: [5800, 2500],      // 更大标准差，增加重叠
    luminosity: [1.0, 1.2],          // 更大标准差
    radius: [1.0, 0.6],              // 更大标准差
    mass: [1.0, 0.6],                // 更大标准差
    colorIndex: [0.65, 0.4],         // 更大标准差
    prior: 0.90,                      // 90% 的恒星是主序星
  },
  红巨星: {
    temperature: [4800, 1500],       // 提高温度均值，增大标准差
    luminosity: [30, 25],            // 降低光度均值，增大标准差
    radius: [15, 12],                // 降低半径均值，增大标准差
    mass: [1.1, 0.5],                // 更接近主序星
    colorIndex: [1.2, 0.4],          // 降低颜色指数，增大标准差
    prior: 0.08,                      // 约 8% 是红巨星
  },
  白矮星: {
    temperature: [12000, 6000],      // 降低温度均值，增大标准差
    luminosity: [0.02, 0.015],       // 提高光度均值，增大标准差
    radius: [0.015, 0.008],          // 增大半径，增大标准差
    mass: [0.8, 0.3],                // 更接近主序星
    colorIndex: [0.0, 0.35],         // 提高颜色指数，增大标准差
    prior: 0.02,                      // 约 2% 是白矮星
  },
};

/**
 * Box-Muller 变换生成标准正态分布随机数
 */
function gaussianRandom(mean: number, std: number): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * std + mean;
}

/**
 * 生成单颗恒星样本
 */
export function generateStarSample(starClass?: StarClass): StarSample {
  const classes: StarClass[] = ["主序星", "红巨星", "白矮星"];

  if (!starClass) {
    const r = Math.random();
    let cumProb = 0;
    for (const c of classes) {
      cumProb += STAR_DISTRIBUTIONS[c].prior;
      if (r < cumProb) {
        starClass = c;
        break;
      }
    }
    starClass = starClass || "主序星";
  }

  const dist = STAR_DISTRIBUTIONS[starClass];

  return {
    class: starClass,
    temperature: gaussianRandom(dist.temperature[0], dist.temperature[1]),
    luminosity: Math.max(0.001, gaussianRandom(dist.luminosity[0], dist.luminosity[1])),
    radius: Math.max(0.001, gaussianRandom(dist.radius[0], dist.radius[1])),
    mass: Math.max(0.1, gaussianRandom(dist.mass[0], dist.mass[1])),
    colorIndex: gaussianRandom(dist.colorIndex[0], dist.colorIndex[1]),
  };
}

/**
 * 生成指定数量的恒星数据集
 */
export function generateStarDataset(count: number, options?: {
  classCounts?: Partial<Record<StarClass, number>>;
  priorAdjustments?: Partial<Record<StarClass, number>>;
}): StarSample[] {
  const { classCounts, priorAdjustments = {} } = options || {};

  const samples: StarSample[] = [];

  if (classCounts) {
    for (const [starClass, n] of Object.entries(classCounts)) {
      for (let i = 0; i < (n as number); i++) {
        samples.push(generateStarSample(starClass as StarClass));
      }
    }
  } else {
    const classes: StarClass[] = ["主序星", "红巨星", "白矮星"];
    const priors = classes.map(c => STAR_DISTRIBUTIONS[c].prior * (priorAdjustments[c] || 1));
    const total = priors.reduce((a, b) => a + b, 0);
    const normalizedPriors = priors.map(p => p / total);

    for (let i = 0; i < count; i++) {
      const r = Math.random();
      let cumProb = 0;
      let starClass: StarClass = "主序星";
      for (let j = 0; j < classes.length; j++) {
        cumProb += normalizedPriors[j];
        if (r < cumProb) {
          starClass = classes[j];
          break;
        }
      }
      samples.push(generateStarSample(starClass));
    }
  }

  return samples;
}

/**
 * 获取某类恒星在特定特征值上的高斯分布参数
 */
export function getFeatureDistribution(
  feature: keyof StarParams,
  starClass: StarClass
): { mean: number; std: number } {
  const dist = STAR_DISTRIBUTIONS[starClass];
  const [mean, std] = dist[feature];
  return { mean, std };
}

/**
 * 计算某特征值在某类恒星分布下的概率密度
 */
export function gaussianPDF(x: number, mean: number, std: number): number {
  const coefficient = 1 / (std * Math.sqrt(2 * Math.PI));
  const exponent = -0.5 * Math.pow((x - mean) / std, 2);
  return coefficient * Math.exp(exponent);
}

/**
 * 获取所有恒星类别的先验概率
 */
export function getPriors(adjustments?: Partial<Record<StarClass, number>>): Record<StarClass, number> {
  const classes: StarClass[] = ["主序星", "红巨星", "白矮星"];
  const priors = classes.map(c => STAR_DISTRIBUTIONS[c].prior * (adjustments?.[c] || 1));
  const total = priors.reduce((a, b) => a + b, 0);
  return classes.reduce((acc, c, i) => {
    acc[c] = priors[i] / total;
    return acc;
  }, {} as Record<StarClass, number>);
}

export const STAR_CLASSES: StarClass[] = ["主序星", "红巨星", "白矮星"];

export const FEATURE_LABELS: Record<keyof StarParams, string> = {
  temperature: "表面温度 (K)",
  luminosity: "光度 (L☉)",
  radius: "半径 (R☉)",
  mass: "质量 (M☉)",
  colorIndex: "颜色指数 (B-V)",
};

export const STAR_COLORS: Record<StarClass, string> = {
  "主序星": "#FCD34D",   // 黄色（太阳色）
  "红巨星": "#EF4444",   // 红色
  "白矮星": "#60A5FA",   // 蓝白色
};
