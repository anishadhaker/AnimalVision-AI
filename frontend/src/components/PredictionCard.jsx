import React from 'react';
import { Award, Zap, Layers, Sparkles, CheckCircle2, Info, ChevronRight } from 'lucide-react';
import { Card, CardHeader, CardContent, CardFooter } from './ui/Card';
import Badge from './ui/Badge';
import { ANIMALS10_EMOJIS } from '../data/animals10MockData';

export default function PredictionCard({
  modelKey,
  data,
  hasHigherConfidence = false,
  isEqualConfidence = false,
  isAgreement = false,
  oodResult = null,
}) {
  if (!data) return null;

  const isSwin = modelKey === 'swin';
  const isOod = oodResult && !oodResult.is_supported;
  const emoji = ANIMALS10_EMOJIS[data.predicted_class?.toLowerCase()] || '🐾';

  // Distinct creative visual identity for CNN vs ViT
  const theme = isSwin
    ? {
        variant: 'violet',
        modelTag: 'ViT Architecture',
        accentText: 'text-violet-600 dark:text-dark-swin',
        badgeBg: 'bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-dark-swin border-violet-200/80 dark:border-violet-800/50',
        barColor: 'bg-gradient-to-r from-violet-500 to-purple-600 dark:from-violet-400 dark:to-purple-500',
        cardBorder: hasHigherConfidence ? 'border-violet-400/80 dark:border-violet-600/70 shadow-md shadow-violet-500/5' : 'border-slate-200/80 dark:border-dark-border',
        cardGlow: 'bg-gradient-to-br from-violet-500/5 via-transparent to-transparent',
        topHighlight: 'bg-violet-50/70 dark:bg-violet-950/40 border border-violet-200/60 dark:border-violet-800/40',
        label: 'Swin-Tiny (Vision Transformer)',
        architecture: 'Shifted-Window Self-Attention (W-MSA)',
        params: '27.53M Params',
        testAccuracy: '98.64%',
      }
    : {
        variant: 'indigo',
        modelTag: 'CNN Architecture',
        accentText: 'text-indigo-600 dark:text-dark-resnet',
        badgeBg: 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-dark-resnet border-indigo-200/80 dark:border-indigo-800/50',
        barColor: 'bg-gradient-to-r from-indigo-500 to-blue-600 dark:from-indigo-400 dark:to-blue-500',
        cardBorder: hasHigherConfidence ? 'border-indigo-400/80 dark:border-indigo-600/70 shadow-md shadow-indigo-500/5' : 'border-slate-200/80 dark:border-dark-border',
        cardGlow: 'bg-gradient-to-br from-indigo-500/5 via-transparent to-transparent',
        topHighlight: 'bg-indigo-50/70 dark:bg-indigo-950/40 border border-indigo-200/60 dark:border-indigo-800/40',
        label: 'ResNet-18 (Residual CNN)',
        architecture: 'Residual Convolutions (3×3 Kernels)',
        params: '11.18M Params',
        testAccuracy: '96.49%',
      };

  // Extract Top 3 predictions safely from either data.top3 or data.probabilities
  const top3 = data.top3 && data.top3.length > 0
    ? data.top3.slice(0, 3)
    : data.probabilities
    ? Object.entries(data.probabilities)
        .map(([cls, prob]) => ({ class_name: cls, probability: Number(prob) }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3)
    : [{ class_name: data.predicted_class, probability: data.confidence }];

  return (
    <Card className={`relative flex flex-col justify-between overflow-hidden transition-all duration-200 ${theme.cardBorder}`}>
      {/* Decorative ambient gradient corner glow */}
      <div className={`absolute top-0 right-0 w-36 h-36 rounded-bl-full pointer-events-none ${theme.cardGlow}`} />

      <div>
        {/* Header: Architecture & Higher Confidence Indicator */}
        <CardHeader className="flex flex-row items-center justify-between pb-3.5 border-b border-slate-100 dark:border-dark-border/80">
          <div className="flex items-center gap-2.5">
            <span className={`px-2.5 py-1 rounded-lg text-[11px] font-bold font-mono uppercase tracking-wider border ${theme.badgeBg}`}>
              {isSwin ? 'Swin-Tiny' : 'ResNet-18'}
            </span>
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-slate-900 dark:text-dark-text leading-tight">
                {theme.label}
              </h3>
              <p className="text-[10px] text-slate-400 dark:text-dark-muted font-mono mt-0.5">
                {theme.architecture}
              </p>
            </div>
          </div>

          <div>
            {hasHigherConfidence ? (
              <Badge variant={isSwin ? 'violet' : 'indigo'} size="xs" dot={true}>
                Higher Confidence
              </Badge>
            ) : isEqualConfidence ? (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-slate-100 dark:bg-dark-secondary text-slate-600 dark:text-dark-muted border border-slate-200/60 dark:border-dark-border">
                Equal Confidence
              </span>
            ) : (
              <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">
                Evaluated
              </span>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-4 pt-4">
          {/* Primary Prediction Block */}
          <div className={`p-4 rounded-2xl border transition-all ${theme.topHighlight}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3.5">
                <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl bg-white dark:bg-dark-card border border-slate-200/90 dark:border-dark-border shadow-xs flex items-center justify-center text-3xl sm:text-4xl shrink-0 select-none">
                  {emoji}
                </div>
                <div>
                  <span className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider block">
                    {isOod ? 'Nearest Candidate (Forced)' : 'Predicted Class'}
                  </span>
                  <h4 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-dark-text capitalize tracking-tight mt-0.5">
                    {data.predicted_class}
                  </h4>
                  <span className="text-[11px] font-mono text-slate-500 dark:text-dark-muted">
                    {isOod ? '⚠️ Out-of-Distribution Output' : 'Animals-10 Category'}
                  </span>
                </div>
              </div>

              <div className="text-right">
                <span className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider block">
                  Confidence
                </span>
                <p className={`text-2xl sm:text-3xl font-black font-mono tracking-tight mt-0.5 ${theme.accentText}`}>
                  {typeof data.confidence === 'number' ? data.confidence.toFixed(2) : data.confidence}%
                </p>
                <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                  Softmax certainty
                </span>
              </div>
            </div>

            {/* Visual Probability Bar for Primary Class */}
            <div className="mt-3.5 pt-3 border-t border-slate-200/60 dark:border-dark-border/60">
              <div className="flex items-center justify-between text-[11px] font-mono mb-1 text-slate-500 dark:text-dark-muted">
                <span>Single-Image Certainty</span>
                <span className="font-bold text-slate-700 dark:text-slate-300">{data.confidence}%</span>
              </div>
              <div className="h-2 w-full rounded-full bg-slate-200/70 dark:bg-dark-elevated overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ${theme.barColor}`}
                  style={{ width: `${Math.min(100, Math.max(2, data.confidence))}%` }}
                />
              </div>
            </div>

            {isOod && (
              <div className="mt-3 px-3 py-2 rounded-xl bg-amber-500/10 dark:bg-amber-950/30 border border-amber-500/20 text-[11px] text-amber-800 dark:text-amber-300 leading-relaxed">
                <strong>OOD Notice:</strong> Softmax score reflects relative candidate weight among the 10 supported classes, not true out-of-distribution probability.
              </div>
            )}
          </div>

          {/* Educational Note: Single-Image Confidence vs Held-Out Test Accuracy */}
          <div className="p-3.5 rounded-2xl bg-slate-50/80 dark:bg-dark-secondary/70 border border-slate-200/70 dark:border-dark-border/80 text-[11px] space-y-2">
            <div className="flex items-center gap-1.5 font-bold text-slate-700 dark:text-slate-200 text-xs">
              <Info className="w-3.5 h-3.5 text-indigo-500 dark:text-dark-resnet shrink-0" />
              <span>Confidence vs. Held-Out Accuracy</span>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center font-mono">
              <div className="p-2.5 rounded-xl bg-white dark:bg-dark-card border border-slate-200/60 dark:border-dark-border/80 shadow-2xs">
                <span className="text-[10px] text-slate-400 dark:text-slate-500 block uppercase font-sans font-bold tracking-wider">
                  Current Image
                </span>
                <span className="text-sm font-bold text-slate-900 dark:text-dark-text mt-0.5 block">
                  {data.confidence}%
                </span>
                <span className="text-[9px] text-slate-400 dark:text-slate-500 block font-sans mt-0.5">
                  Softmax Certainty
                </span>
              </div>

              <div className="p-2.5 rounded-xl bg-white dark:bg-dark-card border border-slate-200/60 dark:border-dark-border/80 shadow-2xs">
                <span className="text-[10px] text-slate-400 dark:text-slate-500 block uppercase font-sans font-bold tracking-wider">
                  Overall Test Set
                </span>
                <span className={`text-sm font-bold mt-0.5 block ${theme.accentText}`}>
                  {theme.testAccuracy}
                </span>
                <span className="text-[9px] text-slate-400 dark:text-slate-500 block font-sans mt-0.5">
                  3,534 Test Images
                </span>
              </div>
            </div>

            <p className="text-[10.5px] text-slate-500 dark:text-dark-muted leading-relaxed pt-0.5">
              Confidence represents the model's certainty for this specific image and does not equal its overall benchmark accuracy.
            </p>
          </div>

          {/* Top 3 Predicted Classes Breakdown */}
          <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-dark-border/80">
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Top-3 Candidate Predictions
              </span>
              <span className="text-slate-400 dark:text-slate-500 font-mono text-[10px]">
                Softmax Distribution
              </span>
            </div>

            <div className="space-y-1.5">
              {top3.map((item, idx) => {
                const isTop = idx === 0;
                const itemEmoji = ANIMALS10_EMOJIS[item.class_name?.toLowerCase()] || '🐾';
                const prob = typeof item.probability === 'number' ? item.probability : Number(item.probability) || 0;

                return (
                  <div
                    key={item.class_name || idx}
                    className={`p-2.5 rounded-xl flex items-center justify-between text-xs transition-all ${
                      isTop
                        ? `${theme.topHighlight} font-bold shadow-2xs`
                        : 'bg-slate-50/50 dark:bg-dark-secondary/40 border border-slate-100 dark:border-dark-border/40 font-medium'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono w-4 shrink-0">
                        #{idx + 1}
                      </span>
                      <span className="text-base shrink-0 select-none">{itemEmoji}</span>
                      <span className={`capitalize truncate text-xs ${isTop ? 'text-slate-900 dark:text-dark-text font-bold' : 'text-slate-700 dark:text-slate-300'}`}>
                        {item.class_name}
                      </span>
                    </div>

                    <div className="flex items-center gap-2.5 shrink-0">
                      <div className="w-16 sm:w-28 h-1.5 bg-slate-200/80 dark:bg-dark-elevated rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isTop ? theme.barColor : 'bg-slate-400 dark:bg-slate-600'
                          }`}
                          style={{ width: `${Math.min(100, Math.max(1, prob))}%` }}
                        />
                      </div>
                      <span className={`font-mono text-xs w-12 text-right ${isTop ? `${theme.accentText} font-bold` : 'text-slate-500 dark:text-dark-muted'}`}>
                        {prob.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </div>

      {/* Footer: Architecture Specification */}
      <CardFooter className="py-3 px-4 sm:px-5 flex items-center justify-between text-[11px] text-slate-500 dark:text-dark-muted border-t border-slate-100 dark:border-dark-border/80 bg-slate-50/40 dark:bg-dark-secondary/30">
        <span className="font-medium text-slate-600 dark:text-slate-300 flex items-center gap-1.5">
          {isSwin ? <Sparkles className="w-3.5 h-3.5 text-violet-500" /> : <Layers className="w-3.5 h-3.5 text-indigo-500" />}
          <span>{isSwin ? 'Vision Transformer' : 'Residual CNN'}</span>
        </span>
        <span className="font-mono text-slate-400 dark:text-slate-500">
          {theme.params}
        </span>
      </CardFooter>
    </Card>
  );
}
