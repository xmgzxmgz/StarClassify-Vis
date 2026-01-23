import { useState, useMemo } from "react";
import { Telescope, Star } from "lucide-react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function PublicWorkspace() {
  const [temperature, setTemperature] = useState(5000);
  const [luminosity, setLuminosity] = useState(1);
  const [radius, setRadius] = useState(1);

  // Mock Prediction Logic
  const predictedType = useMemo(() => {
    // Simplified HR Diagram logic
    // O: > 30000K
    // B: 10000 - 30000K
    // A: 7500 - 10000K
    // F: 6000 - 7500K
    // G: 5200 - 6000K (Sun is 5778K)
    // K: 3700 - 5200K
    // M: < 3700K

    if (temperature > 30000)
      return { type: "O", color: "#9bb0ff", desc: "蓝巨星 - 极热、极亮" };
    if (temperature > 10000)
      return { type: "B", color: "#aabfff", desc: "蓝白星 - 高温" };
    if (temperature > 7500)
      return { type: "A", color: "#cad7ff", desc: "白星 - 常见" };
    if (temperature > 6000)
      return { type: "F", color: "#f8f7ff", desc: "黄白星" };
    if (temperature > 5200)
      return { type: "G", color: "#fff4ea", desc: "黄矮星 - 太阳属于此类" };
    if (temperature > 3700)
      return { type: "K", color: "#ffd2a1", desc: "橙矮星 - 较凉" };
    return { type: "M", color: "#ffcc6f", desc: "红矮星 - 数量最多，低温" };
  }, [temperature, luminosity]);

  // Mock Background Stars for HR Diagram
  const bgStars = useMemo(() => {
    const stars = [];
    for (let i = 0; i < 100; i++) {
      stars.push({
        temp: Math.random() * 30000 + 2000,
        lum: Math.pow(10, Math.random() * 6 - 2),
        r: Math.random() * 2 + 1,
      });
    }
    return stars;
  }, []);

  const [showReport, setShowReport] = useState(false);

  const reportText = useMemo(() => {
    return `【星际探索报告】
      
您发现了一颗${predictedType.desc.split(" - ")[0]}！
      
这颗恒星表面温度高达 ${temperature} K，亮度是太阳的 ${luminosity.toFixed(2)} 倍。
根据赫罗图分析，它属于 ${predictedType.type} 型恒星。
${
  predictedType.type === "G"
    ? "太棒了！太阳也是一颗 G 型恒星，这颗星可能孕育着生命！"
    : predictedType.type === "M"
      ? "这是宇宙中最常见的红矮星，虽然光芒微弱，但寿命极长。"
      : predictedType.type === "O" || predictedType.type === "B"
        ? "这是一颗年轻而狂野的巨星，燃烧速度极快，未来可能会爆发成超新星！"
        : "这是一颗处于演化中的恒星，正散发着独特的光芒。"
}
      
建议后续观测方向：使用光谱仪进一步分析其化学成分。`;
  }, [predictedType, temperature, luminosity]);

  return (
    <div className="mx-auto max-w-[1200px] px-6 py-8">
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[350px_1fr]">
        {/* Interactive Controls */}
        <div className="rounded-xl border border-white/10 bg-white/5 p-6 backdrop-blur">
          <h2 className="mb-6 flex items-center gap-3 text-2xl font-bold text-purple-300">
            <Telescope className="h-6 w-6" /> 恒星探索仪
          </h2>

          <div className="space-y-6">
            <div>
              <label className="mb-2 block text-sm font-medium text-white/80">
                表面温度 (K)
              </label>
              <input
                type="range"
                min="2000"
                max="40000"
                step="100"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-gradient-to-r from-red-500 via-yellow-200 to-blue-500"
              />
              <div className="mt-1 flex justify-between text-xs text-white/50">
                <span>2000K (冷)</span>
                <span className="font-mono text-white">{temperature} K</span>
                <span>40000K (热)</span>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium text-white/80">
                光度 (相对于太阳)
              </label>
              <input
                type="range"
                min="-2"
                max="6"
                step="0.1"
                value={Math.log10(luminosity)}
                onChange={(e) =>
                  setLuminosity(Math.pow(10, Number(e.target.value)))
                }
                className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-white/20"
              />
              <div className="mt-1 flex justify-between text-xs text-white/50">
                <span>暗淡</span>
                <span className="font-mono text-white">
                  {luminosity.toFixed(2)} L☉
                </span>
                <span>明亮</span>
              </div>
            </div>
          </div>

          <div className="mt-8 rounded-xl bg-black/40 p-6 text-center border border-white/10">
            <div className="text-sm text-white/60">预测类型</div>
            <div
              className="my-2 text-6xl font-black"
              style={{ color: predictedType.color }}
            >
              {predictedType.type}
            </div>
            <div className="text-lg font-medium text-white">
              {predictedType.desc}
            </div>
            <div className="mt-4 flex justify-center">
              <div
                className="rounded-full shadow-[0_0_30px_currentColor]"
                style={{
                  width: Math.min(100, Math.max(20, luminosity * 5)) + "px",
                  height: Math.min(100, Math.max(20, luminosity * 5)) + "px",
                  backgroundColor: predictedType.color,
                  color: predictedType.color,
                }}
              />
            </div>
          </div>

          <button
            onClick={() => setShowReport(true)}
            className="mt-6 w-full rounded-lg bg-purple-600 py-3 font-semibold text-white hover:bg-purple-500 transition shadow-lg shadow-purple-500/20"
          >
            生成科普报告
          </button>
        </div>

        {/* Visualization: Simplified HR Diagram */}
        <div className="relative overflow-hidden rounded-xl border border-white/10 bg-[#050810] p-6">
          <h3 className="mb-4 text-xl font-bold text-white/80">
            赫罗图 (H-R Diagram)
          </h3>
          <p className="mb-6 text-sm text-white/50">
            赫罗图展示了恒星温度与光度的关系。大部分恒星位于对角线的"主序带"上。
          </p>

          <div className="h-[500px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
              >
                <XAxis
                  type="number"
                  dataKey="temp"
                  name="温度"
                  unit="K"
                  reversed={true}
                  stroke="#ffffff50"
                />
                <YAxis
                  type="number"
                  dataKey="lum"
                  name="光度"
                  unit="L☉"
                  scale="log"
                  domain={["auto", "auto"]}
                  stroke="#ffffff50"
                />
                <ZAxis type="number" dataKey="r" range={[10, 100]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    borderColor: "#374151",
                  }}
                />

                {/* Background Stars */}
                <Scatter
                  name="Background Stars"
                  data={bgStars}
                  fill="#ffffff30"
                  shape="circle"
                />

                {/* User Star */}
                <Scatter
                  name="Your Star"
                  data={[{ temp: temperature, lum: luminosity }]}
                  fill={predictedType.color}
                >
                  <Cell
                    key="user-star"
                    fill={predictedType.color}
                    stroke="#fff"
                    strokeWidth={2}
                  />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Overlay Info */}
          <div className="absolute top-6 right-6 text-right">
            <div className="flex items-center justify-end gap-2 text-yellow-400">
              <Star className="h-4 w-4 fill-yellow-400" />
              <span className="font-bold">当前位置</span>
            </div>
          </div>

          {/* Report Modal Overlay */}
          {showReport && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8">
              <div className="w-full max-w-md rounded-xl bg-[#1a1f2e] p-6 shadow-2xl border border-purple-500/30">
                <h4 className="mb-4 text-lg font-bold text-purple-300">
                  科普观测报告
                </h4>
                <div className="mb-6 whitespace-pre-wrap text-sm leading-relaxed text-white/90">
                  {reportText}
                </div>
                <button
                  onClick={() => setShowReport(false)}
                  className="w-full rounded-lg bg-white/10 py-2 text-sm hover:bg-white/20"
                >
                  关闭
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
