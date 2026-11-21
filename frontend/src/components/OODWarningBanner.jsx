import React, { useState } from 'react';
import {
  AlertTriangle,
  AlertOctagon,
  HelpCircle,
  ShieldAlert,
  Info,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Layers,
  Activity,
  CheckCircle2,
} from 'lucide-react';
import Badge from './ui/Badge';

export default function OODWarningBanner({ oodResult }) {
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  if (!oodResult || oodResult.is_supported) {
    return null;
  }

  const { status, label, title, warning_message, detail, recommendation, signals } = oodResult;

  // Determine styling theme based on OOD classification status
  const isUnsupportedAnimal = status === 'potentially_unsupported_animal';
  const isNonAnimal = status === 'potentially_non_animal';
  const isUncertain = status === 'uncertain';

  const themeConfig = {
    potentially_unsupported_animal: {
      border: 'border-amber-300 dark:border-amber-800/80',
      bg: 'bg-gradient-to-br from-amber-50 via-amber-50/50 to-orange-50/60 dark:from-amber-950/40 dark:via-dark-card dark:to-orange-950/30',
      badgeVariant: 'amber',
      iconBg: 'bg-amber-100 dark:bg-amber-900/60 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-800',
      icon: AlertTriangle,
      headline: 'Potentially Unsupported Animal Image',
      subhead: 'Animal species appears to be outside our 10 supported classes',
      colorText: 'text-amber-900 dark:text-amber-200',
      tagText: 'Unsupported Animal',
    },
    potentially_non_animal: {
      border: 'border-rose-300 dark:border-rose-800/80',
      bg: 'bg-gradient-to-br from-rose-50 via-rose-50/50 to-red-50/60 dark:from-rose-950/40 dark:via-dark-card dark:to-red-950/30',
      badgeVariant: 'rose',
      iconBg: 'bg-rose-100 dark:bg-rose-900/60 text-rose-700 dark:text-rose-300 border-rose-200 dark:border-rose-800',
      icon: AlertOctagon,
      headline: 'Potentially Non-Animal Object Detected',
      subhead: 'Subject appears to be a vehicle, electronic device, or inanimate object',
      colorText: 'text-rose-900 dark:text-rose-200',
      tagText: 'Non-Animal Object',
    },
    uncertain: {
      border: 'border-slate-300 dark:border-slate-700',
      bg: 'bg-gradient-to-br from-slate-50 via-slate-50/60 to-indigo-50/40 dark:from-slate-900/50 dark:via-dark-card dark:to-indigo-950/20',
      badgeVariant: 'neutral',
      iconBg: 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-700',
      icon: HelpCircle,
      headline: 'Uncertain / Ambiguous Image',
      subhead: 'Unable to confidently determine if this image belongs to a supported class',
      colorText: 'text-slate-900 dark:text-slate-200',
      tagText: 'Ambiguous Input',
    },
  };

  const currentTheme = themeConfig[status] || themeConfig.uncertain;
  const IconComponent = currentTheme.icon;

  return (
    <div
      className={`rounded-3xl p-5 sm:p-6 border ${currentTheme.border} ${currentTheme.bg} transition-all duration-200 shadow-sm`}
    >
      <div className="flex flex-col md:flex-row items-start justify-between gap-4">
        {/* Left Icon and Text Content */}
        <div className="flex items-start gap-4">
          <div
            className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 border ${currentTheme.iconBg}`}
          >
            <IconComponent className="w-6 h-6" />
          </div>

          <div className="space-y-1.5 max-w-3xl">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant={currentTheme.badgeVariant} size="sm" dot={true}>
                {label || currentTheme.tagText}
              </Badge>
              <span className="text-[11px] font-semibold text-slate-500 dark:text-dark-muted px-2 py-0.5 rounded-full bg-white/80 dark:bg-dark-secondary/80 border border-slate-200/60 dark:border-dark-border">
                Out-of-Distribution Warning
              </span>
            </div>

            <h3 className={`text-base sm:text-lg font-bold tracking-tight ${currentTheme.colorText}`}>
              {currentTheme.headline}
            </h3>

            <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
              {warning_message}
            </p>

            {detail && (
              <p className="text-xs text-slate-600 dark:text-dark-muted leading-relaxed">
                {detail}
              </p>
            )}

            {/* Supported Classes Reminder Pill */}
            <div className="pt-2">
              <div className="p-3 rounded-xl bg-white/70 dark:bg-dark-elevated/60 border border-slate-200/80 dark:border-dark-border/80 text-xs">
                <div className="flex items-center gap-1.5 font-bold text-slate-700 dark:text-slate-200 mb-1">
                  <Info className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                  <span>AnimalVision AI 10 Supported Classes:</span>
                </div>
                <p className="text-slate-600 dark:text-dark-muted text-[11px] leading-relaxed">
                  Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel.
                </p>
                <div className="mt-1.5 pt-1.5 border-t border-slate-200/60 dark:border-dark-border/60 text-[11px] text-amber-800 dark:text-amber-300/90 font-medium">
                  <strong>Notice:</strong> Predictions below are <em>forced nearest-candidate mappings</em>. Model agreement or high softmax certainty does <u>not</u> confirm ground-truth membership for unsupported subjects.
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Action: Technical Diagnostic Signals Toggle */}
        <div className="shrink-0 self-end md:self-start">
          <button
            type="button"
            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-white/80 dark:bg-dark-secondary border border-slate-200 dark:border-dark-border text-slate-700 dark:text-dark-text hover:bg-slate-50 dark:hover:bg-dark-elevated transition-colors shadow-2xs"
          >
            <Activity className="w-3.5 h-3.5 text-indigo-500" />
            <span>{showTechnicalDetails ? 'Hide Diagnostics' : 'OOD Diagnostics'}</span>
            {showTechnicalDetails ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Expandable Technical Signals Panel */}
      {showTechnicalDetails && signals && (
        <div className="mt-4 pt-4 border-t border-slate-200/70 dark:border-dark-border/80 animate-fadeIn">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500 dark:text-dark-muted mb-2.5 flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
            <span>Energy-Based Detection Signals (Heuristic Layer)</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            <div className="p-2.5 rounded-xl bg-white/80 dark:bg-dark-elevated border border-slate-200/70 dark:border-dark-border">
              <span className="text-[10px] text-slate-500 dark:text-dark-muted block">Dual Free Energy</span>
              <span className="text-sm font-bold font-mono text-slate-800 dark:text-slate-200">
                {signals.energy_score}
              </span>
              <span className="text-[9px] text-slate-400 block mt-0.5">Norm: ≥ 3.0</span>
            </div>

            <div className="p-2.5 rounded-xl bg-white/80 dark:bg-dark-elevated border border-slate-200/70 dark:border-dark-border">
              <span className="text-[10px] text-slate-500 dark:text-dark-muted block">Swin-Tiny Energy</span>
              <span className="text-sm font-bold font-mono text-violet-700 dark:text-violet-400">
                {signals.swin_energy}
              </span>
              <span className="text-[9px] text-slate-400 block mt-0.5">Norm: ≥ 4.5</span>
            </div>

            <div className="p-2.5 rounded-xl bg-white/80 dark:bg-dark-elevated border border-slate-200/70 dark:border-dark-border">
              <span className="text-[10px] text-slate-500 dark:text-dark-muted block">Logit Margin (Δz)</span>
              <span className="text-sm font-bold font-mono text-indigo-700 dark:text-indigo-400">
                {signals.logit_margin}
              </span>
              <span className="text-[9px] text-slate-400 block mt-0.5">Norm: ≥ 2.0</span>
            </div>

            <div className="p-2.5 rounded-xl bg-white/80 dark:bg-dark-elevated border border-slate-200/70 dark:border-dark-border">
              <span className="text-[10px] text-slate-500 dark:text-dark-muted block">Model Consensus</span>
              <span className="text-sm font-bold text-slate-800 dark:text-slate-200">
                {signals.agreement ? 'Agreed' : 'Diverged'}
              </span>
              <span className="text-[9px] text-slate-400 block mt-0.5">Dual architecture</span>
            </div>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-dark-muted mt-2">
            * Detection operates as a post-hoc out-of-distribution layer combining Helmholtz free energy, logit margin, and dual-model divergence without retraining models.
          </p>
        </div>
      )}
    </div>
  );
}
