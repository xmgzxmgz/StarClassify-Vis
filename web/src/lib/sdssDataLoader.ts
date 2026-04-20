/**
 * SDSS 数据集加载器
 * 从真实的 SDSS 数据集中加载恒星数据
 */

import { StarClass, StarParams } from './starSimulator';

// 从 CSV 字符串解析恒星数据
function parseStarData(csvContent: string): StarParams[] {
  const lines = csvContent.split('\n');
  const headers = lines[0].split(',');
  
  const stars: StarParams[] = [];
  
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    const values = line.split(',');
    if (values[values.length - 1] === 'STAR') {
      const temperature = parseFloat(values[6]);
      const logg = parseFloat(values[7]);
      const luminosity = calculateLuminosity(temperature, logg);
      const star: StarParams = {
        temperature,
        luminosity,
        radius: calculateRadius(temperature, luminosity),
        mass: calculateMass(temperature),
        colorIndex: calculateColorIndex(
          parseFloat(values[2]), // r
          parseFloat(values[3])  // i
        )
      };
      
      stars.push(star);
    }
  }
  
  return stars;
}

// 根据温度和表面重力计算光度
function calculateLuminosity(temperature: number, logg: number): number {
  // 基于温度的简化光度计算（更适合赫罗图展示）
  // 使用主序星的温度-光度关系
  if (temperature < 3500) return 0.01; // 红矮星
  if (temperature < 5000) return 0.1;  // K型星
  if (temperature < 6000) return 1.0;  // G型星（太阳）
  if (temperature < 7500) return 3.0;  // F型星
  if (temperature < 10000) return 10.0; // A型星
  if (temperature < 15000) return 100.0; // B型星
  return 1000.0; // O型星
}

// 根据温度和光度计算半径
function calculateRadius(temperature: number, luminosity: number): number {
  // 基于温度和光度的简化半径计算
  // 使用 Stefan-Boltzmann 定律：L ∝ R² T⁴
  const solarTemperature = 5778; // 太阳温度 (K)
  const solarRadius = 1.0; // 太阳半径
  
  // 计算半径比例
  const radiusRatio = Math.sqrt(luminosity) * Math.pow(solarTemperature / temperature, 2);
  
  return Math.max(0.001, radiusRatio); // 确保半径为正数
}

// 根据温度估算质量
function calculateMass(temperature: number): number {
  // 主序星质量-温度关系的简化公式
  if (temperature < 3500) return 0.1; // 红矮星
  if (temperature < 5000) return 0.5; // K型星
  if (temperature < 6000) return 1.0; // G型星
  if (temperature < 7500) return 1.5; // F型星
  if (temperature < 10000) return 2.0; // A型星
  return 3.0; // B型星
}

// 计算颜色指数 (r-i)
function calculateColorIndex(r: number, i: number): number {
  return r - i;
}

// 根据恒星参数分类（基于温度和光度）
function classifyStar(star: StarParams): StarClass {
  const { temperature, luminosity } = star;
  
  // 白矮星：高温，低光度
  if (temperature > 10000 && luminosity < 0.1) {
    return '白矮星';
  }
  
  // 红巨星：低温，高光度
  if (temperature < 4500 && luminosity > 10) {
    return '红巨星';
  }
  
  // 主序星：其他情况
  return '主序星';
}

// 加载和处理 SDSS 数据
export async function loadSdssData(): Promise<Array<StarParams & { class: StarClass }>> {
  try {
    // 加载 SDSS 数据文件
    const response = await fetch('/datasets/sdss_like_small.csv');
    if (!response.ok) {
      throw new Error('Failed to load SDSS data');
    }
    
    const csvContent = await response.text();
    const stars = parseStarData(csvContent);
    
    // 为每个恒星添加分类
    return stars.map(star => ({
      ...star,
      class: classifyStar(star)
    }));
  } catch (error) {
    console.error('Error loading SDSS data:', error);
    // 失败时返回模拟数据
    return generateFallbackStars(100);
  }
}

// 生成备用恒星数据（当 SDSS 数据加载失败时）
function generateFallbackStars(count: number): Array<StarParams & { class: StarClass }> {
  const stars: Array<StarParams & { class: StarClass }> = [];
  
  for (let i = 0; i < count; i++) {
    const type: StarClass = Math.random() < 0.8 ? '主序星' : (Math.random() < 0.5 ? '红巨星' : '白矮星');
    
    let temperature: number;
    let luminosity: number;
    let radius: number;
    let mass: number;
    let colorIndex: number;
    
    switch (type) {
      case '主序星':
        temperature = 3000 + Math.random() * 5000;
        luminosity = 0.1 + Math.random() * 10;
        radius = 0.5 + Math.random() * 2;
        mass = 0.5 + Math.random() * 2;
        colorIndex = 0.3 + Math.random() * 0.8;
        break;
      case '红巨星':
        temperature = 2500 + Math.random() * 1000;
        luminosity = 10 + Math.random() * 90;
        radius = 10 + Math.random() * 50;
        mass = 0.8 + Math.random() * 2;
        colorIndex = 1.0 + Math.random() * 1.5;
        break;
      case '白矮星':
        temperature = 7000 + Math.random() * 10000;
        luminosity = 0.001 + Math.random() * 0.01;
        radius = 0.01 + Math.random() * 0.09;
        mass = 0.4 + Math.random() * 0.6;
        colorIndex = 0.0 + Math.random() * 0.3;
        break;
    }
    
    stars.push({
      class: type,
      temperature,
      luminosity,
      radius,
      mass,
      colorIndex
    });
  }
  
  return stars;
}
