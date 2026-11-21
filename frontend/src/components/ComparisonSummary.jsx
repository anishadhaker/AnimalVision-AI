import React from 'react';
import { CheckCircle2, AlertTriangle, ArrowRight, Sparkles, Scale, Info } from 'lucide-react';
import Badge from './ui/Badge';
import { ANIMALS10_EMOJIS } from '../data/animals10MockData';

export default function ComparisonSummary({ result }) {
  if (!result || !result.cnn || !result.swin) return null;

  const { cnn, swin, agreement } = result;
  const cnnConf = Number(cnn.confidence);
  const swinConf = Number(swin.confidence);

  // Dynamic calculation of confidence difference in percentage points (pp)
  const diffPp = Math.abs(swinConf - cnnConf).toFixed(2);
  const isEqualConf = Math.abs(swinConf - cnnConf) < 0.001;
  const higherModel = swinConf > cnnConf ? 'Swin-Tiny' : 'ResNet-18';
  const higherConf = Math.max(swinConf, cnnConf);

  const primaryEmoji = ANIMALS10_EMOJIS[cnn.predicted_class?.toLowerCase()] || '🐾';
  const swinEmoji = ANIMALS10_EMOJIS[swin.predicted_class?.toLowerCase()] || '🐾';

  const isOod = result.ood_detection && !result.ood_detection.is_supported;

  return (
    <div
      className={`relative rounded-3xl p-5 sm:p-6 border transition-all duration-200 overflow-hidden ${
        agreement
          ? 'bg-gradient-to-br from-emerald-50/70 via-white to-teal-50/40 dark:from-emerald-950/30 dark:via-dark-card dark:to-teal-950/20 border-emerald-200/80 dark:border-emerald-800/50 shadow-xs'
          : 'bg-gradient-to-br from-amber-50/70 via-white to-orange-50/40 dark:from-amber-950/30 dark:via-dark-card dark:to-orange-950/20 border-amber-200/80 dark:border-amber-800/50 shadow-xs'
      }`}
    >
      {/* Subtle corner decorative glow */}
      <div
        className={`absolute top-0 right-0 w-32 h-32 rounded-bl-full pointer-events-none ${
          agreement
            ? 'bg-gradient-to-bl from-emerald-500/10 via-transparent to-transparent'
            : 'bg-gradient-to-bl from-amber-500/10 via-transparent to-transparent'
        }`}
      />

      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-5">
        {/* Model Agreement / Disagreement Content */}
        <div className="flex items-start gap-4">
          <div
            className={`w-12 h-12 rounded-2xl flex items-center justify-center shrink-0 shadow-2xs ${
              agreement
                ? 'bg-emerald-100 dark:bg-emerald-900/60 text-emerald-700 dark:text-emerald-300 border border-emerald-200/70 dark:border-emerald-800/60'
                : 'bg-amber-100 dark:bg-amber-900/60 text-amber-700 dark:text-amber-300 border border-amber-200/70 dark:border-amber-800/60'
            }`}
          >
            {agreement ? (
              <CheckCircle2 className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
            ) : (
              <AlertTriangle className="w-6 h-6 text-amber-600 dark:text-amber-400" />
            )}
          </div>

          <div className="space-y-1">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant={agreement ? 'emerald' : 'amber'}
                size="sm"
                dot={true}
              >
                {agreement ? 'Model Agreement' : 'Model Disagreement'}
              </Badge>
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded-full bg-slate-100 dark:bg-dark-secondary text-slate-600 dark:text-dark-muted border border-slate-200/60 dark:border-dark-border">
                Confidence Difference: {diffPp} pp
              </span>
            </div>

            <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-dark-text tracking-tight pt-0.5">
              {agreement ? (
                <span className="flex flex-wrap items-center gap-1.5">
                  <span>Both models predicted:</span>
                  <span className="inline-flex items-center gap-1 text-emerald-700 dark:text-emerald-400 font-extrabold capitalize bg-emerald-100/60 dark:bg-emerald-950/60 px-2 py-0.5 rounded-lg border border-emerald-200/80 dark:border-emerald-800/50">
                    <span>{primaryEmoji}</span>
                    <span>{cnn.predicted_class}</span>
                  </span>
                </span>
              ) : (
                <span className="flex flex-wrap items-center gap-1.5">
                  <span>Predictions differ:</span>
                  <span className="text-indigo-700 dark:text-dark-resnet font-bold capitalize">
                    {primaryEmoji} {cnn.predicted_class} (CNN)
                  </span>
                  <span className="text-slate-400">vs</span>
                  <span className="text-violet-700 dark:text-dark-swin font-bold capitalize">
                    {swinEmoji} {swin.predicted_class} (ViT)
                  </span>
                </span>
              )}
            </h3>

            <p className="text-xs text-slate-600 dark:text-dark-muted leading-relaxed max-w-2xl">
              {isOod
                ? agreement
                  ? `Both models converged on '${cnn.predicted_class}' as their nearest candidate, but out-of-distribution analysis indicates this subject lies outside the 10 supported classes. Model agreement does not validate an out-of-domain classification.`
                  : `The models diverged on candidate classes (${cnn.predicted_class} vs ${swin.predicted_class}) due to differing inductive biases when evaluating an out-of-distribution subject.`
                : agreement
                  ? `Both models converged on the same predicted class for this image. Note that model agreement does not guarantee ground-truth correctness.`
                  : 'The models produced different top-1 candidates for this image due to differing architectural inductive biases.'}
            </p>
          </div>
        </div>

        {/* Neutral Confidence Indicator */}
        <div className="flex md:flex-col items-center md:items-end justify-between w-full md:w-auto pt-3 md:pt-0 border-t md:border-t-0 border-slate-200/60 dark:border-dark-border/80 shrink-0">
          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
            {isEqualConf ? 'Confidence Status' : 'Higher Confidence'}
          </span>
          {isEqualConf ? (
            <>
              <span className="text-sm sm:text-base font-extrabold font-mono mt-0.5 text-slate-700 dark:text-slate-200">
                Equal Confidence ({cnnConf}%)
              </span>
              <span className="text-[11px] text-slate-400 dark:text-slate-500 font-mono hidden md:inline">
                0.00 pp difference
              </span>
            </>
          ) : (
            <>
              <span className={`text-sm sm:text-base font-extrabold font-mono mt-0.5 ${
                higherModel === 'Swin-Tiny' ? 'text-violet-700 dark:text-dark-swin' : 'text-indigo-700 dark:text-dark-resnet'
              }`}>
                {higherModel} ({higherConf}%)
              </span>
              <span className="text-[11px] text-slate-400 dark:text-slate-500 font-mono hidden md:inline">
                +{diffPp} pp difference
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
