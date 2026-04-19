/**
 * 赫罗图（H-R Diagram）可视化
 * 二维散点图，展示恒星的温度-光度分布
 * 支持交互调参：显示/隐藏分类、透明度调节、高斯等高线
 */

import { useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from "recharts";
import type { StarClass } from "@/lib/starSimulator";
import { STAR_COLORS, STAR_CLASSES } from "@/lib/starSimulator";

interface HRDiagramProps {
  samples: Array<{
    temperature: number;
    luminosity: number;
    class: StarClass;
  }>;
}

interface VisibilityState {
  主序星: boolean;
  红巨星: boolean;
  白矮星: boolean;
  等高线: boolean;
}

export default function HRDiagram({ samples }: HRDiagramProps) {
  const [visibility, setVisibility] = useState<VisibilityState>({
    主序星: true,
    红巨星: true,
    白矮星: true,
    等高线: false,
  });
  const [opacity, setOpacity] = useState(0.6);

  const toggleVisibility = (key: keyof VisibilityState) => {
    setVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const starClasses: StarClass[] = ["主序星", "红巨星", "白矮星"];

  // 为每类恒星计算高斯分布等高线
  const generateContourData = (starClass: StarClass) => {
    const classSamples = samples.filter((s) => s.class === starClass);
    if (classSamples.length === 0) return [];

    const temps = classSamples.map((s) => s.temperature);
    const lums = classSamples.map((s) => s.luminosity);

    const tempMean = temps.reduce((a, b) => a + b, 0) / temps.length;
    const lumMean = lums.reduce((a, b) => a + b, 0) / lums.length;

    const points: { temperature: number; luminosity: number }[] = [];
    for (let t = 3000; t <= 25000; t += 500) {
      for (let l = -4; l <= 3; l += 0.2) {
        const lumLinear = Math.pow(10, l);
        const dist = Math.sqrt(
          Math.pow((t - tempMean) / 5000, 2) + Math.pow((l - Math.log10(lumMean)) / 1, 2)
        );
        if (dist < 1.5) {
          points.push({ temperature: t, luminosity: l });
        }
      }
    }
    return points;
  };

  const scatterData = starClasses
    .filter((sc) => visibility[sc])
    .map((starClass) => ({
      name: starClass,
      data: samples
        .filter((s) => s.class === starClass)
        .map((s) => ({
          temperature: s.temperature,
          luminosity: Math.log10(Math.max(s.luminosity, 0.0001)),
        })),
    }));

  const contourData = visibility.等高线
    ? starClasses
        .filter((sc) => visibility[sc])
        .map((starClass) => ({
          name: starClass,
          data: generateContourData(starClass),
        }))
    : [];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-xs text-slate-500 dark:text-white/50">显示：</span>
        {starClasses.map((starClass) => (
          <label
            key={starClass}
            className="flex items-center gap-1 cursor-pointer"
          >
            <input
              type="checkbox"
              checked={visibility[starClass]}
              onChange={() => toggleVisibility(starClass)}
              className="h-3 w-3"
              style={{ accentColor: STAR_COLORS[starClass] }}
            />
            <span
              className="text-xs"
              style={{ color: STAR_COLORS[starClass] }}
            >
              {starClass}
            </span>
          </label>
        ))}
        <label className="flex items-center gap-1 cursor-pointer ml-2">
          <input
            type="checkbox"
            checked={visibility.等高线}
            onChange={() => toggleVisibility("等高线")}
            className="h-3 w-3"
            style={{ accentColor: "#888" }}
          />
          <span className="text-xs text-slate-600 dark:text-white/60">
            等高线
          </span>
        </label>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-xs text-slate-500 dark:text-white/50">透明度：</span>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.1"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="w-20"
            style={{ accentColor: "#3B82F6" }}
          />
          <span className="text-xs text-slate-600 dark:text-white/60 w-8">
            {(opacity * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis
              type="number"
              dataKey="temperature"
              name="温度"
              stroke="#888"
              tick={{ fill: "#888", fontSize: 12 }}
              label={{ value: "温度 (K)", position: "bottom", offset: 0, fill: "#888" }}
              domain={[0, 30000]}
              reversed
              tickFormatter={(value) => {
                if (value >= 1000) return `${(value / 1000).toFixed(0)}k`;
                return value.toString();
              }}
            />
            <YAxis
              type="number"
              dataKey="luminosity"
              name="光度"
              stroke="#888"
              tick={{ fill: "#888", fontSize: 12 }}
              label={{ value: "log₁₀(光度/L☉)", angle: -90, position: "insideLeft", fill: "#888" }}
              domain={[-4, 3]}
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{
                backgroundColor: "rgba(0,0,0,0.8)",
                border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: "8px",
                color: "#fff",
              }}
              formatter={(value: number, name: string) => {
                if (name === "temperature") return [`${value.toFixed(0)} K`, "温度"];
                if (name === "luminosity")
                  return [`${Math.pow(10, value).toFixed(4)} L☉`, "光度"];
                return [value, name];
              }}
            />
            {contourData.map(({ name, data }) => (
              <Scatter
                key={`contour-${name}`}
                name={`${name}-等高线`}
                data={data}
                fill={STAR_COLORS[name as StarClass]}
                fillOpacity={0.1}
                shape="circle"
              />
            ))}
            {scatterData.map(({ name, data }) => (
              <Scatter
                key={name}
                name={name}
                data={data}
                fill={STAR_COLORS[name as StarClass]}
                fillOpacity={opacity}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
