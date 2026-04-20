/**
 * 恒星模拟数据生成器
 * 支持两种模式：
 * 1. 基于真实 SDSS 数据（优先）
 * 2. 基于高斯分布的模拟数据（备用）
 */

import { loadSdssData } from './sdssDataLoader';

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

// 缓存的 SDSS 数据
let cachedSdssData: StarSample[] | null = null;

// 转换英文分类到中文
function englishToChineseClass(englishClass: string): StarClass {
  const mapping: Record<string, StarClass> = {
    'mainSequence': '主序星',
    'redGiant': '红巨星',
    'whiteDwarf': '白矮星'
  };
  return mapping[englishClass] || '主序星';
}

// 加载并增强 SDSS 数据（添加红巨星和白矮星）
async function loadAndEnhanceSdssData(): Promise<StarSample[]> {
  if (cachedSdssData) {
    return cachedSdssData;
  }
  
  try {
    // 加载原始 SDSS 数据
    const sdssData = await loadSdssData();
    const processedData = sdssData.map(item => ({
      ...item,
      class: englishToChineseClass(item.class)
    }));
    
    // 增强数据：添加红巨星和白矮星
    const enhancedData = enhanceSdssData(processedData);
    
    cachedSdssData = enhancedData;
    return cachedSdssData;
  } catch (error) {
    console.error('Failed to load SDSS data:', error);
    // 失败时返回增强的模拟数据
    return enhanceSdssData([]);
  }
}

// 增强 SDSS 数据，添加红巨星和白矮星
function enhanceSdssData(sdssData: StarSample[]): StarSample[] {
  const enhanced: StarSample[] = [...sdssData];
  
  // 统计现有分类
  const counts = {
    '主序星': 0,
    '红巨星': 0,
    '白矮星': 0
  };
  
  for (const star of sdssData) {
    counts[star.class]++;
  }
  
  // 添加红巨星（如果不足）
  const redGiantCount = Math.max(30, Math.floor(sdssData.length * 0.1));
  while (counts['红巨星'] < redGiantCount) {
    const redGiant = generateStarSample('红巨星');
    enhanced.push(redGiant);
    counts['红巨星']++;
  }
  
  // 添加白矮星（如果不足）
  const whiteDwarfCount = Math.max(20, Math.floor(sdssData.length * 0.05));
  while (counts['白矮星'] < whiteDwarfCount) {
    const whiteDwarf = generateStarSample('白矮星');
    enhanced.push(whiteDwarf);
    counts['白矮星']++;
  }
  
  return enhanced;
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
 * 从增强的 SDSS 数据中随机采样
 */
export async function generateStarDatasetFromSdss(count: number): Promise<StarSample[]> {
  const enhancedData = await loadAndEnhanceSdssData();
  
  if (enhancedData.length === 0) {
    // 如果 SDSS 数据加载失败，使用模拟数据
    return generateStarDataset(count);
  }
  
  // 随机采样
  const samples: StarSample[] = [];
  for (let i = 0; i < count; i++) {
    const randomIndex = Math.floor(Math.random() * enhancedData.length);
    samples.push({ ...enhancedData[randomIndex] });
  }
  
  return samples;
}

/**
 * 生成指定数量的恒星数据集
 * 优先使用真实 SDSS 数据，失败时使用模拟数据
 */
export async function generateStarDataset(count: number, options?: {
  classCounts?: Partial<Record<StarClass, number>>;
  priorAdjustments?: Partial<Record<StarClass, number>>;
  useSdss?: boolean; // 是否使用 SDSS 数据
}): Promise<StarSample[]> {
  const { classCounts, priorAdjustments = {}, useSdss = true } = options || {};

  // 如果指定使用 SDSS 数据且没有指定具体类别数量
  if (useSdss && !classCounts) {
    return generateStarDatasetFromSdss(count);
  }

  // 否则使用模拟数据
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
 * 同步版本的生成函数（始终使用模拟数据）
 */
export function generateStarDatasetSync(count: number, options?: {
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
