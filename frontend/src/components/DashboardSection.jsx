import React from 'react';
import {
  Sparkles,
  Zap,
  Award,
  Layers,
  ArrowRight,
  Scale,
  ShieldCheck,
  TrendingUp,
  Database,
  Cpu,
  ChevronRight,
} from 'lucide-react';

export default function DashboardSection({
  setActiveTab,
  modelsInfo,
  metricsData,
}) {
  // Verified Phase 3B Held-Out Test Benchmarks (3,534 images)
  const resnetAcc = 96.49;
  const swinAcc = 98.64;
  const accAdvantage = +(swinAcc - resnetAcc).toFixed(2); // +2.15%
  const totalTestSamples = 3534;

  const resnetMacroF1 = 95.92;
  const swinMacroF1 = 98.38;

  // Animals-10 Target Categories
  const animalClasses = [
    { name: 'Butterfly', emoji: '🦋', count: 241 },
    { name: 'Cat', emoji: '🐱', count: 180 },
    { name: 'Chicken', emoji: '🐔', count: 449 },
    { name: 'Cow', emoji: '🐮', count: 270 },
    { name: 'Dog', emoji: '🐶', count: 722 },
    { name: 'Elephant', emoji: '🐘', count: 155 },
    { name: 'Horse', emoji: '🐴', count: 390 },
    { name: 'Sheep', emoji: '🐑', count: 206 },
    { name: 'Spider', emoji: '🕷️', count: 651 },
    { name: 'Squirrel', emoji: '🐿️', count: 270 },
  ];

  return (
    <div className="space-y-8 animate-fadeIn max-w-7xl mx-auto pb-12">
      {/* =========================================================================
          1. DASHBOARD HEADER
          ========================================================================= */}
      <div className="space-y-3">
        {/* Breadcrumb Navigation */}
        <nav aria-label="Breadcrumb" className="flex items-center gap-2 text-xs text-slate-500 dark:text-dark-muted font-medium">
          <span
            className="hover:text-slate-800 dark:hover:text-dark-text transition-colors cursor-pointer"
            onClick={() => setActiveTab('dashboard')}
          >
            Home
          </span>
          <ChevronRight className="w-3.5 h-3.5 text-slate-400 dark:text-dark-muted" />
          <span
            className="hover:text-slate-800 dark:hover:text-dark-text transition-colors cursor-pointer"
            onClick={() => setActiveTab('dashboard')}
          >
            Overview
          </span>
          <ChevronRight className="w-3.5 h-3.5 text-slate-400 dark:text-dark-muted" />
          <span className="text-indigo-600 dark:text-dark-resnet font-semibold">
            Dashboard
          </span>
        </nav>

        {/* Dashboard Title & Meta Badge */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pt-1">
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-slate-900 dark:text-dark-text">
                AnimalVision{' '}
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 via-indigo-500 to-violet-600 dark:from-indigo-400 dark:via-purple-300 dark:to-violet-400">
                  AI
                </span>
              </h1>
              <span className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-dark-resnet border border-indigo-200/70 dark:border-indigo-800/50 shadow-2xs">
                <span className="w-2 h-2 rounded-full bg-indigo-500 dark:bg-dark-resnet animate-pulse" />
                Phase 3B Benchmark
              </span>
            </div>
            <p className="text-sm sm:text-base text-slate-600 dark:text-dark-muted mt-1.5 max-w-2xl leading-relaxed">
              Explore model performance, insights and research results.
            </p>
          </div>

          {/* Minimal Quick Action Pill */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setActiveTab('predict')}
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-500 text-white font-semibold text-xs sm:text-sm shadow-sm transition-all flex items-center gap-2 active:scale-[0.98]"
            >
              <Sparkles className="w-4 h-4 text-indigo-200" />
              <span>Launch Prediction</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTab('compare')}
              className="px-4 py-2 rounded-xl bg-white dark:bg-dark-card hover:bg-slate-50 dark:hover:bg-dark-elevated text-slate-700 dark:text-dark-text border border-slate-200/80 dark:border-dark-border font-semibold text-xs sm:text-sm transition-all flex items-center gap-2 active:scale-[0.98]"
            >
              <Scale className="w-4 h-4 text-slate-500 dark:text-dark-muted" />
              <span className="hidden sm:inline">Compare</span>
            </button>
          </div>
        </div>
      </div>

      {/* =========================================================================
          2. PERFORMANCE SUMMARY CARDS (4 Cards)
          ========================================================================= */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Card 1: ResNet-18 Accuracy */}
        <div className="group relative rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border p-6 shadow-xs hover:shadow-md hover:border-indigo-300 dark:hover:border-indigo-700/60 transition-all duration-200 flex flex-col justify-between overflow-hidden">
          <div className="absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl from-indigo-500/10 via-indigo-500/5 to-transparent rounded-bl-full pointer-events-none" />
          
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-500 dark:text-dark-muted uppercase tracking-wider">
                CNN Baseline
              </span>
              <div className="w-9 h-9 rounded-2xl bg-indigo-50 dark:bg-indigo-950/70 border border-indigo-100 dark:border-indigo-800/40 text-indigo-600 dark:text-dark-resnet flex items-center justify-center shadow-2xs group-hover:scale-105 transition-transform">
                <Cpu className="w-4 h-4" />
              </div>
            </div>

            <div className="mt-4">
              <span className="text-3xl sm:text-4xl font-extrabold font-mono text-slate-900 dark:text-dark-text tracking-tight">
                {resnetAcc.toFixed(2)}%
              </span>
              <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 mt-1">
                ResNet-18 Accuracy
              </h3>
              <p className="text-[11px] text-slate-500 dark:text-dark-muted mt-0.5">
                Residual CNN with local 3×3 receptive fields
              </p>
            </div>
          </div>

          <div className="mt-5 pt-3.5 border-t border-slate-100 dark:border-dark-border flex items-center justify-between text-[11px]">
            <span className="inline-flex items-center gap-1 font-mono text-indigo-600 dark:text-dark-resnet font-semibold">
              <Layers className="w-3 h-3" />
              11.18M Params
            </span>
            <span className="text-slate-400 dark:text-dark-muted font-mono">
              Macro F1: {resnetMacroF1}%
            </span>
          </div>
        </div>

        {/* Card 2: Swin-Tiny Accuracy */}
        <div className="group relative rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border p-6 shadow-xs hover:shadow-md hover:border-violet-300 dark:hover:border-violet-700/60 transition-all duration-200 flex flex-col justify-between overflow-hidden">
          <div className="absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl from-violet-500/10 via-violet-500/5 to-transparent rounded-bl-full pointer-events-none" />

          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-500 dark:text-dark-muted uppercase tracking-wider">
                Vision Transformer
              </span>
              <div className="w-9 h-9 rounded-2xl bg-violet-50 dark:bg-violet-950/70 border border-violet-100 dark:border-violet-800/40 text-violet-600 dark:text-dark-swin flex items-center justify-center shadow-2xs group-hover:scale-105 transition-transform">
                <Sparkles className="w-4 h-4" />
              </div>
            </div>

            <div className="mt-4">
              <span className="text-3xl sm:text-4xl font-extrabold font-mono text-violet-600 dark:text-dark-swin tracking-tight">
                {swinAcc.toFixed(2)}%
              </span>
              <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 mt-1">
                Swin-Tiny Accuracy
              </h3>
              <p className="text-[11px] text-slate-500 dark:text-dark-muted mt-0.5">
                Hierarchical shifted local window attention
              </p>
            </div>
          </div>

          <div className="mt-5 pt-3.5 border-t border-slate-100 dark:border-dark-border flex items-center justify-between text-[11px]">
            <span className="inline-flex items-center gap-1 font-mono text-violet-600 dark:text-dark-swin font-semibold">
              <Zap className="w-3 h-3" />
              27.53M Params
            </span>
            <span className="text-slate-400 dark:text-dark-muted font-mono">
              Macro F1: {swinMacroF1}%
            </span>
          </div>
        </div>

        {/* Card 3: Swin Accuracy Advantage */}
        <div className="group relative rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border p-6 shadow-xs hover:shadow-md hover:border-emerald-300 dark:hover:border-emerald-700/60 transition-all duration-200 flex flex-col justify-between overflow-hidden">
          <div className="absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl from-emerald-500/10 via-emerald-500/5 to-transparent rounded-bl-full pointer-events-none" />

          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-500 dark:text-dark-muted uppercase tracking-wider">
                ViT Advantage
              </span>
              <div className="w-9 h-9 rounded-2xl bg-emerald-50 dark:bg-emerald-950/70 border border-emerald-100 dark:border-emerald-800/40 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shadow-2xs group-hover:scale-105 transition-transform">
                <TrendingUp className="w-4 h-4" />
              </div>
            </div>

            <div className="mt-4">
              <span className="text-3xl sm:text-4xl font-extrabold font-mono text-emerald-600 dark:text-emerald-400 tracking-tight">
                +{accAdvantage.toFixed(2)}%
              </span>
              <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 mt-1">
                Accuracy Lead
              </h3>
              <p className="text-[11px] text-slate-500 dark:text-dark-muted mt-0.5">
                Swin-Tiny percentage points over ResNet-18
              </p>
            </div>
          </div>

          <div className="mt-5 pt-3.5 border-t border-slate-100 dark:border-dark-border flex items-center justify-between text-[11px]">
            <span className="inline-flex items-center gap-1 font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
              <Award className="w-3 h-3" />
              +2.46% Macro F1
            </span>
            <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
              10/10 Class Lead
            </span>
          </div>
        </div>

        {/* Card 4: Test Samples */}
        <div className="group relative rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border p-6 shadow-xs hover:shadow-md hover:border-sky-300 dark:hover:border-sky-700/60 transition-all duration-200 flex flex-col justify-between overflow-hidden">
          <div className="absolute top-0 right-0 w-28 h-28 bg-gradient-to-bl from-sky-500/10 via-sky-500/5 to-transparent rounded-bl-full pointer-events-none" />

          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-500 dark:text-dark-muted uppercase tracking-wider">
                Evaluation Split
              </span>
              <div className="w-9 h-9 rounded-2xl bg-sky-50 dark:bg-sky-950/70 border border-sky-100 dark:border-sky-800/40 text-sky-600 dark:text-sky-400 flex items-center justify-center shadow-2xs group-hover:scale-105 transition-transform">
                <Database className="w-4 h-4" />
              </div>
            </div>

            <div className="mt-4">
              <span className="text-3xl sm:text-4xl font-extrabold font-mono text-slate-900 dark:text-dark-text tracking-tight">
                {totalTestSamples.toLocaleString()}
              </span>
              <h3 className="text-xs font-bold text-slate-800 dark:text-slate-200 mt-1">
                Held-Out Test Images
              </h3>
              <p className="text-[11px] text-slate-500 dark:text-dark-muted mt-0.5">
                Standardized Rapidata/Animals-10 test set
              </p>
            </div>
          </div>

          <div className="mt-5 pt-3.5 border-t border-slate-100 dark:border-dark-border flex items-center justify-between text-[11px]">
            <span className="inline-flex items-center gap-1 font-mono text-sky-600 dark:text-sky-400 font-semibold">
              <ShieldCheck className="w-3 h-3" />
              10 Balanced Classes
            </span>
            <span className="text-slate-400 dark:text-dark-muted font-mono">
              224×224 RGB
            </span>
          </div>
        </div>
      </div>

      {/* =========================================================================
          3. FEATURED PROJECT SECTION: ANIMALVISION AI BANNER
          ========================================================================= */}
      <div className="relative rounded-3xl bg-gradient-to-br from-indigo-900/10 via-slate-900/5 to-violet-900/10 dark:from-indigo-950/40 dark:via-dark-card dark:to-violet-950/40 border border-indigo-200/60 dark:border-indigo-800/50 p-6 sm:p-10 shadow-xs overflow-hidden">
        {/* Decorative background glow circles */}
        <div className="absolute -top-12 -right-12 w-64 h-64 bg-indigo-500/10 dark:bg-indigo-500/15 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -bottom-12 -left-12 w-64 h-64 bg-violet-500/10 dark:bg-violet-500/15 rounded-full blur-3xl pointer-events-none" />

        <div className="relative max-w-4xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white dark:bg-dark-secondary border border-indigo-100 dark:border-indigo-900 text-indigo-700 dark:text-dark-resnet text-xs font-semibold shadow-2xs">
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            <span>Featured Project • Rapidata/Animals-10 Classification</span>
          </div>

          <div className="space-y-2">
            <h2 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-900 dark:text-dark-text tracking-tight">
              Convolutional <span className="text-slate-400 dark:text-slate-500 font-normal">vs</span> Vision Transformer
            </h2>
            <p className="text-sm sm:text-base text-slate-600 dark:text-dark-muted leading-relaxed max-w-3xl">
              An empirical investigation into architectural inductive bias. Comparing residual convolutional local receptive fields against shifted-window self-attention (<code className="font-mono text-indigo-600 dark:text-dark-resnet">W-MSA / SW-MSA</code>) under standardized 224×224 RGB image resolution.
            </p>
          </div>

          {/* 10 Animal Classes Quick Badges */}
          <div className="space-y-2.5 pt-1">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-500 dark:text-dark-muted">
              <span>10 Canonical Animal Classes (3,534 Test Images)</span>
              <span className="font-mono">100% Pretrained Backbone Frozen</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
              {animalClasses.map((animal) => (
                <div
                  key={animal.name}
                  className="px-3 py-2 rounded-xl bg-white/80 dark:bg-dark-secondary/80 border border-slate-200/70 dark:border-dark-border flex items-center justify-between text-xs hover:border-indigo-300 dark:hover:border-indigo-600/60 transition shadow-2xs"
                >
                  <span className="flex items-center gap-2 font-medium text-slate-800 dark:text-dark-text">
                    <span className="text-base">{animal.emoji}</span>
                    <span>{animal.name}</span>
                  </span>
                  <span className="font-mono text-[10px] text-slate-400 dark:text-dark-muted">
                    {animal.count}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Primary Action Button */}
          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              onClick={() => setActiveTab('predict')}
              className="px-6 py-3 rounded-2xl bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-500 text-white font-bold text-xs sm:text-sm shadow-md transition-all flex items-center gap-2.5 active:scale-[0.98]"
            >
              <Sparkles className="w-4 h-4 text-indigo-200" />
              <span>Open Animal Prediction Workspace</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={() => setActiveTab('compare')}
              className="px-5 py-3 rounded-2xl bg-white dark:bg-dark-card hover:bg-slate-50 dark:hover:bg-dark-elevated text-slate-700 dark:text-dark-text border border-slate-200/80 dark:border-dark-border font-semibold text-xs sm:text-sm transition-all flex items-center gap-2 active:scale-[0.98]"
            >
              <Scale className="w-4 h-4 text-slate-500 dark:text-dark-muted" />
              <span>Full Benchmark Report</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
