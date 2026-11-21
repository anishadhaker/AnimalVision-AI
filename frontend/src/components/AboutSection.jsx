import React, { useState } from 'react';
import {
  BookOpen,
  Award,
  Layers,
  Database,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Sparkles,
  ShieldCheck,
  ExternalLink
} from 'lucide-react';
import { ANIMALS10_EMOJIS } from '../data/animals10MockData';

export default function AboutSection({ metricsData, modelsInfo }) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const animalClasses = [
    { name: 'Butterfly', emoji: '🦋', desc: 'Class 0 (Lepidoptera symmetrical wing venation & antennae)' },
    { name: 'Cat', emoji: '🐱', desc: 'Class 1 (Feline triangular ears, whisker pads & pupil contours)' },
    { name: 'Chicken', emoji: '🐔', desc: 'Class 2 (Avian plumage, comb, beak & standing stance)' },
    { name: 'Cow', emoji: '🐮', desc: 'Class 3 (Bovine wide muzzle, horn geometry & herd profile)' },
    { name: 'Dog', emoji: '🐶', desc: 'Class 4 (Canine muzzle, varied ear sets & fur textures)' },
    { name: 'Elephant', emoji: '🐘', desc: 'Class 5 (Pachyderm prehensile trunk, tusks & large ears)' },
    { name: 'Horse', emoji: '🐴', desc: 'Class 6 (Equine elongated head, upright mane & muscular neck)' },
    { name: 'Sheep', emoji: '🐑', desc: 'Class 7 (Ovine curled fleece wool & distinct facial profile)' },
    { name: 'Spider', emoji: '🕷️', desc: 'Class 8 (Arachnid eight articulated legs & cephalothorax)' },
    { name: 'Squirrel', emoji: '🐿️', desc: 'Class 9 (Small rodent bushy arched tail & alert posture)' },
  ];

  return (
    <div className="space-y-6 sm:space-y-8 animate-fadeIn max-w-5xl mx-auto">
      {/* 1. Header Banner */}
      <div className="p-6 sm:p-8 rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-3">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-100 dark:border-indigo-800/40 text-indigo-700 dark:text-dark-resnet text-xs font-semibold">
          <BookOpen className="w-3.5 h-3.5 text-indigo-600 dark:text-dark-resnet" />
          <span>Research Methodology &amp; Engineering Documentation</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-dark-text tracking-tight">
          About AnimalVision AI
        </h1>
        <p className="text-xs sm:text-sm text-slate-600 dark:text-dark-muted leading-relaxed max-w-3xl">
          AnimalVision AI is an empirical comparative study investigating the performance and representation trade-offs between two dominant computer vision paradigms:
          <strong className="text-indigo-600 dark:text-dark-resnet font-semibold"> Residual Convolutional Networks (ResNet-18)</strong> and
          <strong className="text-violet-600 dark:text-dark-swin font-semibold"> Hierarchical Vision Transformers (Swin-Tiny)</strong>.
          Evaluated under standardized conditions across 10 natural animal categories from the Animals-10 dataset using 3,534 held-out test images.
        </p>
      </div>

      {/* 2. Supported Classes & Dataset Distribution */}
      <div className="p-6 rounded-3xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-slate-900 dark:text-dark-text">
              Animals-10 Dataset &amp; Class Partition
            </h2>
            <p className="text-xs text-slate-500 dark:text-dark-muted">
              3,534 held-out test images sampled from Rapidata/Animals-10 at 224×224 resolution:
            </p>
          </div>
          <span className="text-xs font-semibold text-slate-600 dark:text-dark-muted bg-slate-100 dark:bg-dark-secondary px-2.5 py-1 rounded-full">
            10 Classes
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5">
          {animalClasses.map((item) => (
            <div
              key={item.name}
              className="p-3 rounded-xl bg-slate-50/70 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border flex flex-col items-center justify-center text-center hover:bg-white dark:hover:bg-dark-elevated hover:border-indigo-200 dark:hover:border-indigo-500/40 transition"
            >
              <span className="text-2xl mb-1">{item.emoji}</span>
              <span className="text-xs font-bold text-slate-900 dark:text-dark-text">{item.name}</span>
              <p className="text-[10px] text-slate-400 dark:text-dark-muted mt-1 leading-tight">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* 3. Three Core Research Takeaways */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Inductive Bias */}
        <div className="p-5 rounded-2xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-2">
          <div className="w-9 h-9 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-100 dark:border-indigo-800/40 flex items-center justify-center text-indigo-600 dark:text-dark-resnet">
            <Layers className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-dark-text">Inductive Bias (ResNet-18)</h3>
          <p className="text-xs text-slate-500 dark:text-dark-muted leading-relaxed">
            ResNet-18 enforces spatial translation equivariance and localized receptive fields with compact 11.18M parameters, achieving 96.49% held-out test accuracy.
          </p>
        </div>

        {/* Accuracy */}
        <div className="p-5 rounded-2xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-2">
          <div className="w-9 h-9 rounded-xl bg-violet-50 dark:bg-violet-950/60 border border-violet-100 dark:border-violet-800/40 flex items-center justify-center text-violet-600 dark:text-dark-swin">
            <Award className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-dark-text">Accuracy Lead (Swin-Tiny)</h3>
          <p className="text-xs text-slate-500 dark:text-dark-muted leading-relaxed">
            Swin-Tiny achieves <strong className="text-slate-800 dark:text-slate-200">98.64% test accuracy</strong> and <strong className="text-slate-800 dark:text-slate-200">98.38% Macro F1</strong> (+2.15% lead over ResNet-18), excelling at cross-window context modeling.
          </p>
        </div>

        {/* Data Efficiency */}
        <div className="p-5 rounded-2xl bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs space-y-2">
          <div className="w-9 h-9 rounded-xl bg-violet-50 dark:bg-violet-950/60 border border-violet-100 dark:border-violet-800/40 flex items-center justify-center text-violet-600 dark:text-dark-swin">
            <Database className="w-4 h-4" />
          </div>
          <h3 className="text-sm font-bold text-slate-900 dark:text-dark-text">Data Efficiency Scaling</h3>
          <p className="text-xs text-slate-500 dark:text-dark-muted leading-relaxed">
            Swin-Tiny sustains a +3.88% accuracy lead even at 10% data (1,650 samples), demonstrating strong representation transfer and sample efficiency.
          </p>
        </div>
      </div>

      {/* 4. Benchmark Setup & Technical Specifications Accordion */}
      <div className="bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border rounded-3xl p-5 shadow-xs space-y-3">
        <button
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="w-full flex items-center justify-between text-xs font-bold text-slate-700 dark:text-dark-text hover:text-indigo-600 dark:hover:text-dark-resnet transition"
        >
          <span className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-indigo-600 dark:text-dark-resnet" />
            <span>Benchmark Setup &amp; Technical Specifications</span>
          </span>
          <span className="flex items-center gap-1 text-slate-400 dark:text-dark-muted">
            {showTechnicalDetails ? 'Collapse' : 'Show details'}
            {showTechnicalDetails ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </span>
        </button>

        {showTechnicalDetails && (
          <div className="pt-3 border-t border-slate-100 dark:border-dark-border space-y-4 animate-fadeIn text-xs text-slate-600 dark:text-dark-muted">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border">
                <span className="text-slate-400 dark:text-dark-muted block text-[10px] font-bold uppercase">Test Dataset</span>
                <span className="font-bold text-slate-900 dark:text-dark-text text-sm mt-0.5 block font-mono">3,534 Images</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border">
                <span className="text-slate-400 dark:text-dark-muted block text-[10px] font-bold uppercase">ResNet-18 Test Acc</span>
                <span className="font-bold text-indigo-600 dark:text-dark-resnet text-sm mt-0.5 block font-mono">96.49%</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border">
                <span className="text-slate-400 dark:text-dark-muted block text-[10px] font-bold uppercase">Swin-Tiny Test Acc</span>
                <span className="font-bold text-violet-600 dark:text-dark-swin text-sm mt-0.5 block font-mono">98.64%</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border">
                <span className="text-slate-400 dark:text-dark-muted block text-[10px] font-bold uppercase">Input Resolution</span>
                <span className="font-bold text-slate-900 dark:text-dark-text text-sm mt-0.5 block font-mono">224×224 RGB</span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-dark-secondary border border-slate-200/60 dark:border-dark-border flex flex-wrap items-center justify-between gap-2 text-slate-500 dark:text-dark-muted font-mono text-[11px]">
              <span>PyTorch 2.4+ • Torchvision • FastAPI • React 18 • Vite • Tailwind CSS</span>
              <span className="text-emerald-700 dark:text-emerald-400 font-semibold font-sans">✓ Verified Checkpoints Preserved</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
