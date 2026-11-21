import React from 'react';
import {
  Award,
  TrendingUp,
  CheckCircle2,
  Info,
  Sparkles,
  BarChart3,
} from 'lucide-react';
import Badge from './ui/Badge';

export default function OverallPerformanceSection() {
  // Verified held-out test benchmark data (3,534 test images)
  const BENCHMARK_DATA = {
    datasetSize: '3,534 held-out test images',
    accuracyAdvantage: '+2.15 percentage points',
    resnet: {
      name: 'ResNet-18',
      architecture: 'Residual CNN',
      testAccuracy: 96.49,
      macroF1: 95.92,
      correctCount: '3,410 / 3,534',
    },
    swin: {
      name: 'Swin-Tiny',
      architecture: 'Vision Transformer',
      testAccuracy: 98.64,
      macroF1: 98.38,
      correctCount: '3,486 / 3,534',
      badge: 'Better Overall Performance',
      advantage: '+2.15 percentage points',
    },
  };

  return (
    <section
      aria-label="Overall Model Performance"
      className="rounded-3xl p-5 sm:p-6 bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-5 transition-all"
    >
      {/* 1. Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-100 dark:border-dark-border/70">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200/80 dark:border-indigo-800/60 flex items-center justify-center text-indigo-600 dark:text-dark-resnet shrink-0 shadow-2xs">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-lg sm:text-xl font-bold tracking-tight text-slate-900 dark:text-dark-text">
              Overall Model Performance
            </h3>
            <p className="text-xs text-slate-500 dark:text-dark-muted">
              Evaluated on {BENCHMARK_DATA.datasetSize} across all 10 animal classes
            </p>
          </div>
        </div>

        {/* Advantage Highlight Pill */}
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200/80 dark:border-emerald-800/60 text-emerald-800 dark:text-emerald-300 text-xs font-bold self-start sm:self-auto shadow-2xs">
          <TrendingUp className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
          <span>Swin-Tiny Advantage: {BENCHMARK_DATA.accuracyAdvantage}</span>
        </div>
      </div>

      {/* 2. Side-by-Side Model Comparison Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-5">
        {/* ResNet-18 Card */}
        <div className="p-4 sm:p-5 rounded-2xl bg-slate-50/70 dark:bg-dark-secondary/50 border border-slate-200/70 dark:border-dark-border/80 flex flex-col justify-between space-y-3.5">
          <div>
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 dark:bg-dark-resnet" />
                <h4 className="font-extrabold text-slate-900 dark:text-dark-text text-base sm:text-lg">
                  ResNet-18
                </h4>
              </div>
              <span className="text-[11px] font-semibold px-2.5 py-0.5 rounded-full bg-indigo-100/70 dark:bg-indigo-950/70 text-indigo-700 dark:text-dark-resnet border border-indigo-200/70 dark:border-indigo-800/50">
                {BENCHMARK_DATA.resnet.architecture}
              </span>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div className="p-3 rounded-xl bg-white dark:bg-dark-card border border-slate-200/60 dark:border-dark-border">
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 dark:text-slate-500 block">
                  Test Accuracy
                </span>
                <span className="text-xl sm:text-2xl font-black font-mono text-slate-900 dark:text-dark-text mt-0.5 block">
                  {BENCHMARK_DATA.resnet.testAccuracy.toFixed(2)}%
                </span>
                <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                  {BENCHMARK_DATA.resnet.correctCount} correct
                </span>
              </div>

              <div className="p-3 rounded-xl bg-white dark:bg-dark-card border border-slate-200/60 dark:border-dark-border">
                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 dark:text-slate-500 block">
                  Macro F1-Score
                </span>
                <span className="text-xl sm:text-2xl font-black font-mono text-slate-700 dark:text-slate-300 mt-0.5 block">
                  {BENCHMARK_DATA.resnet.macroF1.toFixed(2)}%
                </span>
                <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                  Balanced 10-class mean
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Swin-Tiny Card (Highlighted Leader) */}
        <div className="p-4 sm:p-5 rounded-2xl bg-gradient-to-br from-violet-50/80 via-purple-50/40 to-white dark:from-violet-950/30 dark:via-dark-card dark:to-dark-secondary border-2 border-violet-300 dark:border-violet-700/70 shadow-xs flex flex-col justify-between space-y-3.5 relative overflow-hidden">
          {/* Subtle decorative glow */}
          <div className="absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl from-violet-400/10 via-transparent to-transparent pointer-events-none" />

          <div>
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-violet-600 dark:bg-dark-swin" />
                <h4 className="font-extrabold text-slate-900 dark:text-dark-text text-base sm:text-lg">
                  Swin-Tiny
                </h4>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="inline-flex items-center gap-1 text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-violet-100 dark:bg-violet-900/60 text-violet-800 dark:text-violet-300 border border-violet-200 dark:border-violet-700">
                  <Sparkles className="w-3 h-3 text-violet-600 dark:text-violet-400" />
                  <span>{BENCHMARK_DATA.swin.badge}</span>
                </span>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div className="p-3 rounded-xl bg-white/90 dark:bg-dark-card/90 border border-violet-200/80 dark:border-violet-800/60 shadow-2xs">
                <span className="text-[10px] uppercase font-bold tracking-wider text-violet-600 dark:text-violet-400 block">
                  Test Accuracy
                </span>
                <span className="text-xl sm:text-2xl font-black font-mono text-violet-700 dark:text-dark-swin mt-0.5 block">
                  {BENCHMARK_DATA.swin.testAccuracy.toFixed(2)}%
                </span>
                <span className="text-[10px] text-violet-500/80 dark:text-violet-400/80 font-mono">
                  {BENCHMARK_DATA.swin.correctCount} correct
                </span>
              </div>

              <div className="p-3 rounded-xl bg-white/90 dark:bg-dark-card/90 border border-violet-200/80 dark:border-violet-800/60 shadow-2xs">
                <span className="text-[10px] uppercase font-bold tracking-wider text-violet-600 dark:text-violet-400 block">
                  Macro F1-Score
                </span>
                <span className="text-xl sm:text-2xl font-black font-mono text-violet-700 dark:text-dark-swin mt-0.5 block">
                  {BENCHMARK_DATA.swin.macroF1.toFixed(2)}%
                </span>
                <span className="text-[10px] text-violet-500/80 dark:text-violet-400/80 font-mono">
                  Balanced 10-class mean
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Accuracy Comparison Visualization (Two Horizontal Bars on the Same Scale) */}
      <div className="p-4 sm:p-5 rounded-2xl bg-slate-50/70 dark:bg-dark-secondary/40 border border-slate-200/70 dark:border-dark-border/80 space-y-3">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 font-bold text-slate-800 dark:text-dark-text">
            <BarChart3 className="w-4 h-4 text-indigo-500 dark:text-dark-resnet" />
            <span>Accuracy Comparison (Same Scale: 0% – 100%)</span>
          </div>
          <span className="text-[11px] font-mono text-slate-500 dark:text-dark-muted hidden sm:inline">
            Difference: {BENCHMARK_DATA.accuracyAdvantage}
          </span>
        </div>

        {/* ResNet-18 Bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-semibold text-slate-700 dark:text-slate-300">
              ResNet-18 (CNN)
            </span>
            <span className="font-extrabold text-slate-900 dark:text-slate-100">
              {BENCHMARK_DATA.resnet.testAccuracy.toFixed(2)}%
            </span>
          </div>
          <div className="h-3 w-full rounded-full bg-slate-200/70 dark:bg-dark-elevated overflow-hidden">
            <div
              role="progressbar"
              aria-valuenow={BENCHMARK_DATA.resnet.testAccuracy}
              aria-valuemin="0"
              aria-valuemax="100"
              className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-blue-600 transition-all duration-700"
              style={{ width: `${BENCHMARK_DATA.resnet.testAccuracy}%` }}
            />
          </div>
        </div>

        {/* Swin-Tiny Bar */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-xs font-mono">
            <span className="font-semibold text-violet-700 dark:text-dark-swin flex items-center gap-1.5">
              <span>Swin-Tiny (ViT)</span>
              <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-violet-100 dark:bg-violet-950/80 text-violet-800 dark:text-violet-300 border border-violet-200/60 dark:border-violet-800/60">
                Better Overall
              </span>
            </span>
            <span className="font-extrabold text-violet-700 dark:text-dark-swin">
              {BENCHMARK_DATA.swin.testAccuracy.toFixed(2)}% ({BENCHMARK_DATA.accuracyAdvantage})
            </span>
          </div>
          <div className="h-3 w-full rounded-full bg-slate-200/70 dark:bg-dark-elevated overflow-hidden">
            <div
              role="progressbar"
              aria-valuenow={BENCHMARK_DATA.swin.testAccuracy}
              aria-valuemin="0"
              aria-valuemax="100"
              className="h-full rounded-full bg-gradient-to-r from-violet-500 to-purple-600 transition-all duration-700"
              style={{ width: `${BENCHMARK_DATA.swin.testAccuracy}%` }}
            />
          </div>
        </div>
      </div>

      {/* 4. Highlight & Evaluated Explanation Callout */}
      <div className="p-3.5 sm:p-4 rounded-2xl bg-emerald-50/70 dark:bg-emerald-950/30 border border-emerald-200/80 dark:border-emerald-800/60 flex items-start gap-3">
        <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm text-emerald-950 dark:text-emerald-200 leading-relaxed">
          <span className="font-extrabold">Better Overall Performance: </span>
          Swin-Tiny achieved higher test accuracy than ResNet-18 on the evaluated held-out dataset.
        </div>
      </div>

      {/* 5. Educational Note */}
      <div className="p-3.5 sm:p-4 rounded-2xl bg-slate-50/90 dark:bg-dark-secondary/60 border border-slate-200/70 dark:border-dark-border text-xs text-slate-600 dark:text-dark-muted flex items-start gap-3">
        <Info className="w-4 h-4 text-indigo-500 dark:text-dark-resnet shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          Overall accuracy measures how correctly a model classifies images across the complete held-out test dataset. Prediction confidence represents the model's confidence for an individual uploaded image. A higher confidence score on one image does not necessarily mean that the model performs better overall.
        </p>
      </div>
    </section>
  );
}
