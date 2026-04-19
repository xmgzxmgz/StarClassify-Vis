/**
 * 高斯分布曲线图
 * 展示单特征在各类恒星中的高斯分布
 * 支持对数坐标（用于光度和半径）
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { StarClass } from "@/lib/starSimulator";
import { STAR_COLORS } from "@/lib/starSimulator";

interface GaussianCurveChartProps {
  data: Record<StarClass, Array<{ x: number; y: number }>>;
  featureLabel: string;
  selectedFeature?: string;
}

// 为不同特征设置合适的X轴范围
const FEATURE_RANGES: Record<string, { min: number; max: number }> = {
  temperature: { min: 2000, max: 20000 },
  luminosity: { min: 0.01, max: 100 },
  radius: { min: 0.01, max: 30 },
  mass: { min: 0.1, max: 5 },
  colorIndex: { min: -1, max: 2.5 },
};

export default function GaussianCurveChart({
  data,
  featureLabel,
  selectedFeature,
}: GaussianCurveChartProps) {
  const starClasses: StarClass[] = ["主序星", "红巨星", "白矮星"];

  const chartData = data["主序星"].map((point, i) => ({
    x: point.x,
    xDisplay: point.x,
    主序星: point.y,
    红巨星: data["红巨星"][i]?.y ?? 0,
    白矮星: data["白矮星"][i]?.y ?? 0,
  }));

  // 对数和线性切换
  const isLogFeature = selectedFeature === 'luminosity' || selectedFeature === 'radius';
  const range = FEATURE_RANGES[selectedFeature || 'temperature'] || FEATURE_RANGES.temperature;

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis
            dataKey="x"
            stroke="#888"
            tick={{ fill: "#888", fontSize: 12 }}
            domain={[range.min, range.max]}
            tickFormatter={(value) => {
              if (value >= 1000) return `${(value/1000).toFixed(1)}k`;
              if (value >= 1) return value.toFixed(1);
              if (value >= 0.01) return value.toFixed(2);
              return value.toExponential(0);
            }}
          />
          <YAxis
            stroke="#888"
            tick={{ fill: "#888", fontSize: 12 }}
            label={{ value: "概率密度", angle: -90, position: "insideLeft", fill: "#888" }}
            domain={[0, "auto"]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "rgba(0,0,0,0.8)",
              border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: "8px",
              color: "#fff",
            }}
            formatter={(value: number) => value.toFixed(6)}
            labelFormatter={(value) => {
              if (selectedFeature === 'luminosity') return `光度: ${value.toFixed(4)} L☉`;
              if (selectedFeature === 'radius') return `半径: ${value.toFixed(4)} R☉`;
              if (selectedFeature === 'temperature') return `温度: ${value.toFixed(0)} K`;
              if (selectedFeature === 'mass') return `质量: ${value.toFixed(2)} M☉`;
              return `颜色指数: ${value.toFixed(2)}`;
            }}
          />
          <Legend
            wrapperStyle={{ color: "#888" }}
            formatter={(value) => <span style={{ color: "#888" }}>{value}</span>}
          />
          {starClasses.map((starClass) => (
            <Line
              key={starClass}
              type="monotone"
              dataKey={starClass}
              stroke={STAR_COLORS[starClass]}
              strokeWidth={2}
              dot={false}
              name={starClass}
              connectNulls={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
