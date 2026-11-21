import React, { useState } from 'react';
import { Grid, Eye, CheckCircle2, AlertTriangle, Info, Sparkles, ArrowRight } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import { ANIMALS10_CLASSES, ANIMALS10_EMOJIS } from '../data/animals10MockData';

export default function ConfusionMatrixViewer({
  resnetMatrix,
  swinMatrix,
  classes = ANIMALS10_CLASSES,
}) {
  const [activeModel, setActiveModel] = useState('both'); // 'both' | 'resnet' | 'swin'
  const [hoveredCell, setHoveredCell] = useState(null); // { model, r, c, val, support, pct }

  // Fallbacks if not provided
  const rCM = resnetMatrix && resnetMatrix.length === 10 ? resnetMatrix : [];
  const sCM = swinMatrix && swinMatrix.length === 10 ? swinMatrix : [];

  const getRowSupport = (matrix, r) => {
    if (!matrix[r]) return 0;
    return matrix[r].reduce((sum, val) => sum + val, 0);
  };

  const getDiagonalTotal = (matrix) => {
    let total = 0;
    for (let i = 0; i < Math.min(matrix.length, 10); i++) {
      if (matrix[i] && matrix[i][i]) total += matrix[i][i];
    }
    return total;
  };

  const rTotalCorrect = getDiagonalTotal(rCM);
  const sTotalCorrect = getDiagonalTotal(sCM);
  const totalTestImages = 3534;

  const renderSingleMatrix = (matrix, modelKey, label, themeColor) => {
    const isSwin = modelKey === 'swin';
    const totalCorrect = isSwin ? sTotalCorrect : rTotalCorrect;
    const accuracy = ((totalCorrect / totalTestImages) * 100).toFixed(2);

    return (
      <div className="space-y-3">
        {/* Model Subheader */}
        <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-dark-border/80">
          <div className="flex items-center gap-2">
            <Badge variant={isSwin ? 'violet' : 'indigo'} size="sm">
              {isSwin ? 'SWIN-TINY' : 'RESNET-18'}
            </Badge>
            <span className="text-xs sm:text-sm font-bold text-slate-900 dark:text-dark-text">
              {label}
            </span>
          </div>

          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-slate-500 dark:text-dark-muted hidden sm:inline">
              Correct: <strong className="text-slate-800 dark:text-slate-200">{totalCorrect}</strong>/{totalTestImages}
            </span>
            <span
              className={`font-bold px-2 py-0.5 rounded-md ${
                isSwin
                  ? 'bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-dark-swin border border-violet-200/70 dark:border-violet-800/40'
                  : 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-dark-resnet border border-indigo-200/70 dark:border-indigo-800/40'
              }`}
            >
              {accuracy}% Acc
            </span>
          </div>
        </div>

        {/* 10x10 Matrix Grid Table */}
        <div className="overflow-x-auto pb-1">
          <table className="min-w-full text-[10px] sm:text-xs text-center border-collapse">
            <thead>
              <tr>
                <th className="p-1 text-left font-mono text-[9px] uppercase tracking-wider text-slate-400 dark:text-slate-500 w-16">
                  True \ Pred
                </th>
                {classes.map((cls, idx) => (
                  <th
                    key={cls}
                    className="p-1 font-mono text-[9px] sm:text-[10px] text-slate-500 dark:text-slate-400 font-semibold"
                    title={`Predicted: ${cls} (#${idx})`}
                  >
                    <span className="block">{cls.slice(0, 3).toUpperCase()}</span>
                  </th>
                ))}
                <th className="p-1 font-mono text-[9px] uppercase tracking-wider text-slate-400 dark:text-slate-500 w-10">
                  Total
                </th>
              </tr>
            </thead>
            <tbody>
              {classes.map((trueClass, r) => {
                const support = getRowSupport(matrix, r);
                const emoji = ANIMALS10_EMOJIS[trueClass.toLowerCase()] || '🐾';

                return (
                  <tr key={trueClass} className="hover:bg-slate-50/50 dark:hover:bg-dark-secondary/40 transition-colors">
                    {/* Row header: True Class */}
                    <td className="p-1 text-left font-medium text-slate-700 dark:text-slate-300 flex items-center gap-1 truncate w-16">
                      <span className="text-xs select-none">{emoji}</span>
                      <span className="truncate capitalize">{trueClass.slice(0, 4)}</span>
                    </td>

                    {/* Cells */}
                    {classes.map((predClass, c) => {
                      const val = matrix[r] ? matrix[r][c] : 0;
                      const isDiagonal = r === c;
                      const pct = support > 0 ? ((val / support) * 100).toFixed(1) : 0;

                      // Cell background styling based on diagonal accuracy or error intensity
                      let cellClass = 'text-slate-300 dark:text-slate-600 bg-transparent';
                      if (isDiagonal) {
                        cellClass = isSwin
                          ? 'bg-violet-100 dark:bg-violet-950/70 text-violet-900 dark:text-violet-200 font-bold border border-violet-200/80 dark:border-violet-800/60'
                          : 'bg-indigo-100 dark:bg-indigo-950/70 text-indigo-900 dark:text-indigo-200 font-bold border border-indigo-200/80 dark:border-indigo-800/60';
                      } else if (val > 0) {
                        cellClass = val > 5
                          ? 'bg-rose-100 dark:bg-rose-950/60 text-rose-800 dark:text-rose-300 font-bold'
                          : 'bg-amber-50 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 font-medium';
                      }

                      return (
                        <td
                          key={predClass}
                          onMouseEnter={() =>
                            setHoveredCell({
                              model: isSwin ? 'Swin-Tiny' : 'ResNet-18',
                              trueClass,
                              predClass,
                              val,
                              support,
                              pct,
                              isDiagonal,
                            })
                          }
                          onMouseLeave={() => setHoveredCell(null)}
                          className={`p-1.5 sm:p-2 text-center rounded-md font-mono text-[10px] sm:text-xs transition-colors cursor-pointer ${cellClass}`}
                          title={`True: ${trueClass}, Predicted: ${predClass} -> ${val} images (${pct}%)`}
                        >
                          {val > 0 ? val : '·'}
                        </td>
                      );
                    })}

                    {/* Support count */}
                    <td className="p-1 font-mono text-[10px] text-slate-400 dark:text-slate-500 font-medium w-10">
                      {support}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <Grid className="w-4 h-4 text-indigo-600 dark:text-dark-resnet" />
            <CardTitle>10×10 Confusion Matrices</CardTitle>
            <Badge variant="default" size="xs">
              3,534 Test Images
            </Badge>
          </div>
          <CardDescription>
            Side-by-side distribution of ground truth labels (rows) vs predicted classes (columns).
          </CardDescription>
        </div>

        {/* View Switcher: Both, ResNet, Swin */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-100 dark:bg-dark-secondary border border-slate-200/80 dark:border-dark-border text-xs font-semibold">
          <button
            type="button"
            onClick={() => setActiveModel('both')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activeModel === 'both'
                ? 'bg-white dark:bg-dark-card text-slate-900 dark:text-dark-text shadow-2xs'
                : 'text-slate-500 dark:text-dark-muted hover:text-slate-900'
            }`}
          >
            Side-by-Side
          </button>
          <button
            type="button"
            onClick={() => setActiveModel('resnet')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activeModel === 'resnet'
                ? 'bg-indigo-600 text-white shadow-2xs'
                : 'text-slate-500 dark:text-dark-muted hover:text-indigo-600'
            }`}
          >
            ResNet-18
          </button>
          <button
            type="button"
            onClick={() => setActiveModel('swin')}
            className={`px-2.5 py-1 rounded-lg transition-all ${
              activeModel === 'swin'
                ? 'bg-violet-600 text-white shadow-2xs'
                : 'text-slate-500 dark:text-dark-muted hover:text-violet-600'
            }`}
          >
            Swin-Tiny
          </button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6 pt-2">
        {/* Hover Inspector Pill */}
        <div className="p-2.5 rounded-xl bg-slate-50 dark:bg-dark-secondary border border-slate-200/70 dark:border-dark-border/80 flex items-center justify-between text-xs min-h-[38px]">
          {hoveredCell ? (
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-bold text-slate-900 dark:text-dark-text">
                [{hoveredCell.model}]
              </span>
              <span className="text-slate-500 dark:text-dark-muted">
                True: <strong className="capitalize text-slate-800 dark:text-slate-200">{hoveredCell.trueClass}</strong>
              </span>
              <ArrowRight className="w-3 h-3 text-slate-400" />
              <span className="text-slate-500 dark:text-dark-muted">
                Predicted: <strong className="capitalize text-slate-800 dark:text-slate-200">{hoveredCell.predClass}</strong>
              </span>
              <span className="text-slate-300 dark:text-slate-600">•</span>
              <span className="font-mono font-bold text-slate-900 dark:text-dark-text">
                {hoveredCell.val} images ({hoveredCell.pct}% of true class support)
              </span>
              {hoveredCell.isDiagonal ? (
                <Badge variant="emerald" size="xs">Correct</Badge>
              ) : hoveredCell.val > 0 ? (
                <Badge variant="rose" size="xs">Misclassified</Badge>
              ) : null}
            </div>
          ) : (
            <span className="text-[11px] text-slate-400 dark:text-slate-500 italic flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5" />
              Hover over any cell in the 10×10 grid to inspect exact class pair sample counts and error percentages.
            </span>
          )}
        </div>

        {/* Matrix Grids */}
        {activeModel === 'both' ? (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {renderSingleMatrix(rCM, 'resnet', 'ResNet-18 (Residual CNN)', 'indigo')}
            {renderSingleMatrix(sCM, 'swin', 'Swin-Tiny (Vision Transformer)', 'violet')}
          </div>
        ) : activeModel === 'resnet' ? (
          <div>{renderSingleMatrix(rCM, 'resnet', 'ResNet-18 (Residual CNN)', 'indigo')}</div>
        ) : (
          <div>{renderSingleMatrix(sCM, 'swin', 'Swin-Tiny (Vision Transformer)', 'violet')}</div>
        )}

        {/* Empirical Confusion Insights Callout */}
        <div className="p-3.5 rounded-xl bg-slate-50/70 dark:bg-dark-secondary/60 border border-slate-200/60 dark:border-dark-border text-xs text-slate-600 dark:text-dark-muted space-y-1">
          <span className="font-bold text-slate-800 dark:text-slate-200 flex items-center gap-1.5 text-[11px] uppercase tracking-wider">
            <Sparkles className="w-3 h-3 text-indigo-600 dark:text-dark-resnet" />
            Key Confusion Observations
          </span>
          <p className="text-[11px] leading-relaxed">
            - <strong>Quadruped Semantic Overlap:</strong> In ResNet-18, the most frequent confusion occurred between <code>cow</code> and <code>sheep</code> (13 cows predicted as sheep) and <code>cow</code> and <code>horse</code> (12 cows predicted as horses).
            <br />
            - <strong>Attention Feature Disambiguation:</strong> Swin-Tiny reduced <code>cow</code> $\rightarrow$ <code>sheep</code> confusion to just 3 images (a 76.9% reduction), demonstrating superior discrimination on fine-grained visual fur and silhouette textures.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
