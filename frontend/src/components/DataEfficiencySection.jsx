import React, { useState } from 'react';
import {
  Database,
  TrendingUp,
  BarChart3,
  Award,
  Layers,
  Sparkles,
  Info,
  CheckCircle2,
  Scale,
  ArrowRight,
  ExternalLink,
  ShieldCheck,
  ChevronRight
} from 'lucide-react';
import { ANIMALS10_DATA_EFFICIENCY_BENCHMARK } from '../data/animals10MockData';

export default function DataEfficiencySection({ dataEfficiencyData }) {
  const [activeMetricTab, setActiveMetricTab] = useState('accuracy'); // 'accuracy' | 'macroF1' | 'delta'
  const [hoveredFraction, setHoveredFraction] = useState(null);
  const [showPlotModal, setShowPlotModal] = useState(false);

  const benchmark =
    dataEfficiencyData && dataEfficiencyData.available
      ? dataEfficiencyData
      : ANIMALS10_DATA_EFFICIENCY_BENCHMARK;

  const comparison = benchmark.comparison_by_fraction || [];

  // Active chart data
  const chartPoints = comparison.map((item) => ({
    fraction: item.fraction,
    fractionPct: item.fraction_pct,
    trainSamples: item.train_samples,
    resnet: activeMetricTab === 'macroF1' ? item.resnet_macro_f1 : item.resnet_test_acc,
    swin: activeMetricTab === 'macroF1' ? item.swin_macro_f1 : item.swin_test_acc,
    delta: activeMetricTab === 'macroF1' ? item.f1_delta : item.acc_delta,
  }));

  // SVG Chart bounds
  const yMin = activeMetricTab === 'delta' ? 0 : 92;
  const yMax = activeMetricTab === 'delta' ? 5 : 100;
  const svgWidth = 640;
  const svgHeight = 240;
  const padLeft = 45;
  const padRight = 30;
  const padTop = 25;
  const padBottom = 35;
  const plotW = svgWidth - padLeft - padRight;
  const plotH = svgHeight - padTop - padBottom;

  const getX = (idx) => padLeft + (idx / (chartPoints.length - 1)) * plotW;
  const getY = (val) => padTop + plotH - ((val - yMin) / (yMax - yMin)) * plotH;

  const swinPointsStr = chartPoints.map((p, i) => `${getX(i)},${getY(p.swin)}`).join(' ');
  const resnetPointsStr = chartPoints.map((p, i) => `${getX(i)},${getY(p.resnet)}`).join(' ');

  return (
    <div className="space-y-6 sm:space-y-8 animate-fadeIn">
      {/* 1. Header Banner */}
      <div className="p-6 sm:p-8 rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-3 transition-colors duration-150">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-950/60 border border-cyan-100 dark:border-cyan-800/40 text-cyan-700 dark:text-cyan-400 text-xs font-semibold">
            <Database className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
            <span>Phase A Empirical Benchmark</span>
          </div>

          <button
            onClick={() => setShowPlotModal(true)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-slate-600 dark:text-dark-muted hover:text-slate-900 dark:hover:text-dark-text bg-slate-50 dark:bg-dark-secondary hover:bg-slate-100 dark:hover:bg-dark-border border border-slate-200/70 dark:border-dark-border transition"
          >
            <ExternalLink className="w-3.5 h-3.5 text-indigo-500" />
            <span>Inspect Generated Plot</span>
          </button>
        </div>

        <div className="space-y-1.5">
          <h2 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-dark-text tracking-tight flex items-center gap-2">
            <span>Data Efficiency &amp; Sample Scaling Study</span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-dark-resnet border border-indigo-100 dark:border-indigo-800/40 font-mono font-medium">
              10% – 100%
            </span>
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-dark-muted max-w-3xl leading-relaxed">
            Investigating how Vision Transformer (<strong className="text-violet-600 dark:text-dark-swin font-semibold">Swin-Tiny</strong>) and Residual CNN (<strong className="text-indigo-600 dark:text-dark-resnet font-semibold">ResNet-18</strong>) adapt under constrained data budgets across identical, nested stratified training subsets on the complete <strong className="text-slate-800 dark:text-slate-200">3,534 held-out Animals-10 test set</strong>.
          </p>
        </div>

        {/* Protocol Highlights Ribbon */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-3 border-t border-slate-100 dark:border-dark-border">
          <div className="flex items-start gap-2 text-xs">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-bold text-slate-800 dark:text-slate-200">Frozen-Backbone Adaptation</span>
              <p className="text-[11px] text-slate-400 dark:text-dark-muted leading-tight">
                ImageNet-1K backbones were completely frozen; only linear classification heads were trained.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-2 text-xs">
            <CheckCircle2 className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-bold text-slate-800 dark:text-slate-200">Nested Stratified Subsets</span>
              <p className="text-[11px] text-slate-400 dark:text-dark-muted leading-tight">
                10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100% (Seed 42); both models trained on identical image indices.
              </p>
            </div>
          </div>

          <div className="flex items-start gap-2 text-xs">
            <CheckCircle2 className="w-4 h-4 text-cyan-500 shrink-0 mt-0.5" />
            <div className="space-y-0.5">
              <span className="font-bold text-slate-800 dark:text-slate-200">Untouched Held-Out Test Split</span>
              <p className="text-[11px] text-slate-400 dark:text-dark-muted leading-tight">
                All 3,534 test images remained strictly excluded from training and checkpoint selection.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Interactive Scaling Curves Chart */}
      <div className="p-6 sm:p-8 rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-6 transition-colors duration-150">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-dark-border">
          <div>
            <h3 className="text-base font-bold text-slate-900 dark:text-dark-text flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
              <span>Comparative Sample Scaling Trajectory</span>
            </h3>
            <p className="text-xs text-slate-500 dark:text-dark-muted">
              Hover over data points to inspect exact test performance and delta
            </p>
          </div>

          {/* Metric Switch Buttons */}
          <div className="flex items-center p-1 rounded-xl bg-slate-100 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border self-start sm:self-auto">
            <button
              onClick={() => setActiveMetricTab('accuracy')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                activeMetricTab === 'accuracy'
                  ? 'bg-white dark:bg-dark-card text-indigo-700 dark:text-dark-resnet shadow-xs'
                  : 'text-slate-500 dark:text-dark-muted hover:text-slate-800 dark:hover:text-dark-text'
              }`}
            >
              Test Accuracy (%)
            </button>
            <button
              onClick={() => setActiveMetricTab('macroF1')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                activeMetricTab === 'macroF1'
                  ? 'bg-white dark:bg-dark-card text-indigo-700 dark:text-dark-resnet shadow-xs'
                  : 'text-slate-500 dark:text-dark-muted hover:text-slate-800 dark:hover:text-dark-text'
              }`}
            >
              Macro F1-Score (%)
            </button>
            <button
              onClick={() => setActiveMetricTab('delta')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                activeMetricTab === 'delta'
                  ? 'bg-white dark:bg-dark-card text-cyan-700 dark:text-cyan-400 shadow-xs'
                  : 'text-slate-500 dark:text-dark-muted hover:text-slate-800 dark:hover:text-dark-text'
              }`}
            >
              Swin Advantage (+Δ%)
            </button>
          </div>
        </div>

        {/* SVG Chart */}
        <div className="space-y-3">
          <div className="w-full overflow-x-auto">
            <svg
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full h-auto min-w-[540px] select-none"
            >
              {/* Grid Lines */}
              {[0, 0.25, 0.5, 0.75, 1.0].map((t) => {
                const yVal = yMin + t * (yMax - yMin);
                const yPos = getY(yVal);
                return (
                  <g key={t}>
                    <line
                      x1={padLeft}
                      y1={yPos}
                      x2={svgWidth - padRight}
                      y2={yPos}
                      className="stroke-slate-100 dark:stroke-dark-border"
                      strokeWidth="1"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={padLeft - 8}
                      y={yPos + 3}
                      textAnchor="end"
                      className="fill-slate-400 dark:fill-dark-muted text-[10px] font-mono"
                    >
                      {activeMetricTab === 'delta' ? `+${yVal.toFixed(1)}%` : `${yVal.toFixed(0)}%`}
                    </text>
                  </g>
                );
              })}

              {/* X-axis labels */}
              {chartPoints.map((p, i) => {
                const xPos = getX(i);
                return (
                  <g key={p.fraction}>
                    <line
                      x1={xPos}
                      y1={padTop}
                      x2={xPos}
                      y2={padTop + plotH}
                      className="stroke-slate-100 dark:stroke-dark-border"
                      strokeWidth="1"
                    />
                    <text
                      x={xPos}
                      y={svgHeight - 12}
                      textAnchor="middle"
                      className="fill-slate-600 dark:fill-slate-300 text-[11px] font-bold"
                    >
                      {p.fraction}
                    </text>
                    <text
                      x={xPos}
                      y={svgHeight - 2}
                      textAnchor="middle"
                      className="fill-slate-400 dark:fill-dark-muted text-[9px] font-mono"
                    >
                      {p.trainSamples.toLocaleString()}
                    </text>
                  </g>
                );
              })}

              {activeMetricTab !== 'delta' ? (
                <>
                  {/* Swin-Tiny Line */}
                  <polyline
                    fill="none"
                    stroke="#7c3aed"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={swinPointsStr}
                  />

                  {/* ResNet-18 Line */}
                  <polyline
                    fill="none"
                    stroke="#4f46e5"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={resnetPointsStr}
                  />

                  {/* Data Points */}
                  {chartPoints.map((p, i) => {
                    const xPos = getX(i);
                    const ySwin = getY(p.swin);
                    const yResnet = getY(p.resnet);
                    const isHovered = hoveredFraction === p.fraction;

                    return (
                      <g key={p.fraction}>
                        {/* Swin point */}
                        <circle
                          cx={xPos}
                          cy={ySwin}
                          r={isHovered ? 6 : 4.5}
                          className="fill-violet-600 stroke-white dark:stroke-dark-card transition-all cursor-pointer"
                          strokeWidth="2"
                          onMouseEnter={() => setHoveredFraction(p.fraction)}
                          onMouseLeave={() => setHoveredFraction(null)}
                        />
                        <text
                          x={xPos}
                          y={ySwin - 8}
                          textAnchor="middle"
                          className="fill-violet-700 dark:fill-dark-swin text-[10px] font-bold font-mono"
                        >
                          {p.swin.toFixed(1)}%
                        </text>

                        {/* ResNet point */}
                        <circle
                          cx={xPos}
                          cy={yResnet}
                          r={isHovered ? 6 : 4.5}
                          className="fill-indigo-600 stroke-white dark:stroke-dark-card transition-all cursor-pointer"
                          strokeWidth="2"
                          onMouseEnter={() => setHoveredFraction(p.fraction)}
                          onMouseLeave={() => setHoveredFraction(null)}
                        />
                        <text
                          x={xPos}
                          y={yResnet + 15}
                          textAnchor="middle"
                          className="fill-indigo-700 dark:fill-dark-resnet text-[10px] font-bold font-mono"
                        >
                          {p.resnet.toFixed(1)}%
                        </text>
                      </g>
                    );
                  })}
                </>
              ) : (
                /* Delta Bar Chart Mode */
                chartPoints.map((p, i) => {
                  const xPos = getX(i);
                  const barW = 28;
                  const yZero = getY(0);
                  const yPos = getY(p.delta);
                  const barH = Math.max(2, yZero - yPos);

                  return (
                    <g key={p.fraction}>
                      <rect
                        x={xPos - barW / 2}
                        y={yPos}
                        width={barW}
                        height={barH}
                        rx="4"
                        className="fill-cyan-500 dark:fill-cyan-400 hover:fill-cyan-600 transition-colors"
                      />
                      <text
                        x={xPos}
                        y={yPos - 6}
                        textAnchor="middle"
                        className="fill-cyan-700 dark:fill-cyan-300 text-[11px] font-bold font-mono"
                      >
                        +{p.delta.toFixed(2)}%
                      </text>
                    </g>
                  );
                })
              )}
            </svg>
          </div>

          {/* Legend and Active Status */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-100 dark:border-dark-border text-xs">
            <div className="flex items-center gap-5">
              {activeMetricTab !== 'delta' ? (
                <>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-violet-600 shrink-0" />
                    <span className="font-bold text-slate-800 dark:text-dark-text">Swin-Tiny (Vision Transformer)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-indigo-600 shrink-0" />
                    <span className="font-bold text-slate-800 dark:text-dark-text">ResNet-18 (Residual CNN)</span>
                  </div>
                </>
              ) : (
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-md bg-cyan-500 shrink-0" />
                  <span className="font-bold text-slate-800 dark:text-dark-text">Swin-Tiny Performance Lead (+Δ % Accuracy Points)</span>
                </div>
              )}
            </div>

            <div className="text-[11px] text-slate-400 dark:text-dark-muted font-mono">
              Evaluated on 3,534 held-out test images
            </div>
          </div>
        </div>
      </div>

      {/* 3. Detailed Data Efficiency Quantitative Table */}
      <div className="bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border rounded-3xl shadow-xs overflow-hidden transition-colors duration-150">
        <div className="p-5 sm:p-6 border-b border-slate-100 dark:border-dark-border flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm sm:text-base font-bold text-slate-900 dark:text-dark-text">
              Master Data Efficiency Quantitative Breakdown
            </h3>
            <p className="text-xs text-slate-500 dark:text-dark-muted">
              Side-by-side performance comparison across all 5 nested training budgets
            </p>
          </div>

          <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-cyan-50 dark:bg-cyan-950/60 text-cyan-700 dark:text-cyan-400 border border-cyan-100 dark:border-cyan-800/40">
            5 Training Fractions
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50/80 dark:bg-dark-secondary border-b border-slate-200/70 dark:border-dark-border text-slate-600 dark:text-dark-muted font-bold uppercase text-[10px] tracking-wider">
              <tr>
                <th className="py-3 px-4">Dataset Fraction</th>
                <th className="py-3 px-4">Train Samples</th>
                <th className="py-3 px-4 text-indigo-700 dark:text-dark-resnet">ResNet-18 Test Acc</th>
                <th className="py-3 px-4 text-violet-700 dark:text-dark-swin">Swin-Tiny Test Acc</th>
                <th className="py-3 px-4 text-center">Accuracy Delta</th>
                <th className="py-3 px-4 text-indigo-700 dark:text-dark-resnet">ResNet-18 Macro F1</th>
                <th className="py-3 px-4 text-violet-700 dark:text-dark-swin">Swin-Tiny Macro F1</th>
                <th className="py-3 px-4 text-center">F1 Delta</th>
                <th className="py-3 px-4 text-slate-400">Head Train Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-dark-border font-medium">
              {comparison.map((row) => {
                const isHighlight = row.fraction === '10%' || row.fraction === '100%';
                return (
                  <tr
                    key={row.fraction}
                    className={`hover:bg-slate-50/60 dark:hover:bg-dark-secondary/60 transition-colors ${
                      isHighlight ? 'bg-slate-50/40 dark:bg-dark-secondary/30' : ''
                    }`}
                  >
                    <td className="py-3.5 px-4 font-bold text-slate-900 dark:text-dark-text flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-500 shrink-0" />
                      <span>{row.fraction}</span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-600 dark:text-dark-muted">
                      {row.train_samples.toLocaleString()} images
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-800 dark:text-slate-200 font-bold">
                      {row.resnet_test_acc.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-800 dark:text-slate-200 font-bold">
                      {row.swin_test_acc.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-4 text-center">
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-cyan-50 dark:bg-cyan-950/60 text-cyan-700 dark:text-cyan-400 border border-cyan-100 dark:border-cyan-800/40">
                        +{row.acc_delta.toFixed(2)}% Swin
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-800 dark:text-slate-200">
                      {row.resnet_macro_f1.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-800 dark:text-slate-200">
                      {row.swin_macro_f1.toFixed(2)}%
                    </td>
                    <td className="py-3.5 px-4 text-center">
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-dark-swin border border-violet-100 dark:border-violet-800/40">
                        +{row.f1_delta.toFixed(2)}%
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono text-slate-400 text-[11px]">
                      {row.resnet_train_time}s / {row.swin_train_time}s
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Empirical Research Findings Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
        <div className="p-5 sm:p-6 rounded-3xl bg-violet-50/50 dark:bg-violet-950/20 border border-violet-100 dark:border-violet-900/30 space-y-2">
          <div className="flex items-center gap-2 text-violet-900 dark:text-dark-swin font-bold text-sm">
            <Award className="w-4 h-4 text-violet-600 dark:text-dark-swin shrink-0" />
            <span>Finding 1: Transformer Dominance in Low-Data Budgets</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-dark-muted leading-relaxed">
            At only <strong className="text-violet-900 dark:text-violet-300">10% training data (1,650 images)</strong>, Swin-Tiny achieved <strong className="text-slate-900 dark:text-slate-100">98.59% test accuracy</strong>, establishing its widest margin over ResNet-18 (<strong className="text-violet-700 dark:text-dark-swin">+3.88% accuracy, +4.20% Macro F1</strong>). Shifted-window self-attention features demonstrate remarkable transferability with sparse supervision.
          </p>
        </div>

        <div className="p-5 sm:p-6 rounded-3xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100 dark:border-indigo-900/30 space-y-2">
          <div className="flex items-center gap-2 text-indigo-900 dark:text-dark-resnet font-bold text-sm">
            <Scale className="w-4 h-4 text-indigo-600 dark:text-dark-resnet shrink-0" />
            <span>Finding 2: CNN Scaling Efficiency &amp; Data Saturation</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-dark-muted leading-relaxed">
            ResNet-18 scales steadily from <strong className="text-indigo-900 dark:text-indigo-300">94.71% (10%)</strong> to <strong className="text-indigo-900 dark:text-indigo-300">96.21% (75%)</strong>, progressively closing the accuracy gap against Swin. By <strong className="text-slate-900 dark:text-slate-100">50% data (~8,245 images)</strong>, both models achieve &gt;95% accuracy, indicating strong sample efficiency for both paradigms.
          </p>
        </div>
      </div>

      {/* Plot Modal */}
      {showPlotModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs animate-fadeIn">
          <div className="bg-white dark:bg-dark-card border border-slate-200 dark:border-dark-border rounded-3xl p-5 max-w-4xl w-full max-h-[90vh] overflow-y-auto space-y-4 shadow-xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-dark-border">
              <div>
                <h4 className="text-sm font-bold text-slate-900 dark:text-dark-text">
                  Generated Comparative Visualization (Matplotlib PNG)
                </h4>
                <p className="text-xs text-slate-400 font-mono">results/animals10/data_efficiency_comparison.png</p>
              </div>
              <button
                onClick={() => setShowPlotModal(false)}
                className="px-3 py-1 text-xs font-semibold rounded-lg bg-slate-100 dark:bg-dark-secondary hover:bg-slate-200 dark:hover:bg-dark-border text-slate-700 dark:text-dark-text transition"
              >
                Close
              </button>
            </div>

            <div className="rounded-2xl overflow-hidden bg-slate-50 dark:bg-dark-secondary border border-slate-200/80 dark:border-dark-border">
              <img
                src="/api/animals10/data-efficiency-plot"
                alt="Data Efficiency Benchmark Comparison"
                className="w-full h-auto object-contain"
                onError={(e) => {
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              <div style={{ display: 'none' }} className="p-8 text-center text-xs text-slate-500">
                Plot image available at <code>results/animals10/data_efficiency_comparison.png</code>.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
