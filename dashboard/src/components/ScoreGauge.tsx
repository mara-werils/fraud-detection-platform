"use client";

interface ScoreGaugeProps {
  score: number; // 0-1
  size?: number;
  strokeWidth?: number;
}

export default function ScoreGauge({
  score,
  size = 120,
  strokeWidth = 8,
}: ScoreGaugeProps) {
  const percentage = Math.round(score * 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score * circumference);
  const center = size / 2;

  const getColor = () => {
    if (score >= 0.7) return { stroke: "#ef4444", text: "text-red-400" };
    if (score >= 0.3) return { stroke: "#f97316", text: "text-yellow-400" };
    return { stroke: "#22c55e", text: "text-green-400" };
  };

  const { stroke, text } = getColor();

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#1f2937"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={stroke}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-700 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-2xl font-bold ${text}`}>{percentage}</span>
        <span className="text-[10px] font-medium text-gray-500 uppercase tracking-wider">
          Risk
        </span>
      </div>
    </div>
  );
}
