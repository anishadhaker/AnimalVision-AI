import React from 'react';
import ImageUploader from './ImageUploader';
import PredictionCard from './PredictionCard';
import ComparisonSummary from './ComparisonSummary';
import OODWarningBanner from './OODWarningBanner';
import OverallPerformanceSection from './OverallPerformanceSection';
import { Card } from './ui/Card';
import Badge from './ui/Badge';
import StatusPill from './ui/StatusPill';
import {
  Sparkles,
  AlertCircle,
  FileQuestion,
  Loader2,
  RefreshCw,
  Layers,
  Zap,
  ArrowRight,
  RotateCcw,
  CheckCircle2,
} from 'lucide-react';

export default function PredictionSection({
  selectedImage,
  onImageSelect,
  onPredict,
  isLoading,
  samples,
  onSelectSample,
  predictionResult,
  errorMsg,
  setActiveTab,
  backendStatus = true,
}) {
  return (
    <div className="space-y-7 animate-fadeIn max-w-5xl mx-auto pb-12">
      {/* =========================================================================
          A. PAGE HEADER
          ========================================================================= */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-100 dark:border-dark-border/80">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-slate-900 dark:text-dark-text">
              Animal Prediction
            </h1>
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-dark-resnet border border-indigo-200/70 dark:border-indigo-800/50">
              Dual Inference Engine
            </span>
          </div>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-dark-muted mt-1 leading-relaxed">
            Side-by-side comparative inference: ResNet-18 (Residual CNN) vs Swin-Tiny (Vision Transformer) on 10 animal categories.
          </p>
        </div>

        {/* Small Backend Connection Status */}
        <div className="flex items-center gap-2 self-start sm:self-auto shrink-0">
          <StatusPill
            active={backendStatus}
            activeText="Models Ready"
            inactiveText="Backend Disconnected"
            detail="10 Classes"
          />
        </div>
      </div>

      {/* Backend Offline Warning Banner */}
      {!backendStatus && (
        <div className="p-4 rounded-2xl bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/40 text-amber-800 dark:text-amber-300 text-xs sm:text-sm flex items-center justify-between gap-3 shadow-2xs">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0" />
            <span>FastAPI backend is offline. Inference requires the server running on port 8000.</span>
          </div>
          <button
            onClick={() => window.location.reload()}
            className="font-bold underline hover:text-amber-950 dark:hover:text-amber-100 shrink-0"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* =========================================================================
          B. IMAGE UPLOAD & SELECTION WORKSPACE
          ========================================================================= */}
      <div className="space-y-5">
        <ImageUploader
          selectedImage={selectedImage}
          onImageSelect={onImageSelect}
          samples={samples}
          onSelectSample={onSelectSample}
          isLoading={isLoading}
        />

        {/* Primary Prediction Action Button (Prominent & Eye-Catching) */}
        {selectedImage && (
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
            <button
              type="button"
              disabled={isLoading || !backendStatus}
              onClick={onPredict}
              className={`w-full sm:w-auto min-w-[320px] px-8 py-3.5 rounded-2xl font-bold text-sm text-white shadow-lg transition-all duration-200 flex items-center justify-center gap-2.5 active:scale-[0.98] ${
                isLoading || !backendStatus
                  ? 'bg-slate-400 dark:bg-slate-700 cursor-not-allowed opacity-80'
                  : 'bg-gradient-to-r from-indigo-600 via-purple-600 to-violet-600 hover:from-indigo-700 hover:via-purple-700 hover:to-violet-700 shadow-indigo-500/20 hover:shadow-indigo-500/30 hover:scale-[1.01]'
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-white" />
                  <span>Processing Dual Forward Passes...</span>
                </>
              ) : predictionResult ? (
                <>
                  <RefreshCw className="w-4 h-4 text-indigo-200" />
                  <span>Re-run Dual Classification</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 text-indigo-200" />
                  <span>Classify with Both Models</span>
                  <ArrowRight className="w-4 h-4 text-indigo-200 ml-1" />
                </>
              )}
            </button>

            {predictionResult && (
              <button
                type="button"
                onClick={() => onImageSelect(null)}
                className="w-full sm:w-auto px-5 py-3.5 rounded-2xl bg-slate-100 dark:bg-dark-secondary hover:bg-slate-200 dark:hover:bg-dark-elevated text-slate-700 dark:text-dark-text border border-slate-200/80 dark:border-dark-border text-sm font-semibold flex items-center justify-center gap-2 transition active:scale-[0.98]"
              >
                <RotateCcw className="w-4 h-4 text-slate-500" />
                <span>Try Another Image</span>
              </button>
            )}
          </div>
        )}

        {/* General Error Notification State */}
        {errorMsg && (
          <div className="p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-800/40 text-rose-700 dark:text-rose-300 text-xs sm:text-sm flex items-center justify-between gap-3 animate-fadeIn shadow-2xs">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="w-5 h-5 shrink-0 text-rose-600 dark:text-rose-400" />
              <span>
                <strong>Inference Notice:</strong> {errorMsg}
              </span>
            </div>
            <button
              onClick={() => onImageSelect(null)}
              className="font-semibold text-rose-700 dark:text-rose-300 underline hover:text-rose-900 shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>

      {/* Initial Empty State Banner */}
      {!selectedImage && !predictionResult && !isLoading && (
        <Card className="p-8 sm:p-12 text-center border-dashed border-slate-200 dark:border-dark-border shadow-xs">
          <div className="w-14 h-14 rounded-3xl bg-slate-100 dark:bg-dark-secondary border border-slate-200 dark:border-dark-border flex items-center justify-center text-slate-400 dark:text-slate-500 mx-auto mb-3.5 shadow-2xs">
            <FileQuestion className="w-7 h-7" />
          </div>
          <h3 className="text-sm sm:text-base font-bold text-slate-800 dark:text-dark-text">
            No Image Selected Yet
          </h3>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-dark-muted mt-1.5 max-w-md mx-auto leading-relaxed">
            Drag and drop an animal image above, browse your device, or click any of the 10 quick test samples to benchmark ResNet-18 against Swin-Tiny.
          </p>
        </Card>
      )}

      {/* =========================================================================
          C. ANIMATED AI PROCESSING STATE
          ========================================================================= */}
      {isLoading && (
        <div className="space-y-6 pt-3 animate-fadeIn">
          {/* Animated Processing Card */}
          <div className="rounded-3xl p-6 sm:p-8 bg-gradient-to-r from-indigo-50/80 via-purple-50/80 to-violet-50/80 dark:from-indigo-950/40 dark:via-purple-950/40 dark:to-violet-950/40 border border-indigo-200/80 dark:border-indigo-800/50 text-center space-y-4 shadow-sm">
            <div className="w-14 h-14 rounded-2xl bg-white dark:bg-dark-card border border-indigo-100 dark:border-indigo-900 flex items-center justify-center mx-auto text-indigo-600 dark:text-dark-resnet shadow-sm">
              <Loader2 className="w-7 h-7 animate-spin text-indigo-600 dark:text-dark-resnet" />
            </div>

            <div className="space-y-1">
              <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-dark-text">
                Both models are analyzing your image
              </h3>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-dark-muted max-w-lg mx-auto leading-relaxed">
                Evaluating ResNet-18 (Residual CNN) and Swin-Tiny (Vision Transformer) dual forward passes across 10 animal categories at 224×224 resolution...
              </p>
            </div>

            {/* Indeterminate Animated Scanning Beam */}
            <div className="max-w-md mx-auto h-2 rounded-full bg-slate-200/70 dark:bg-dark-elevated overflow-hidden relative">
              <div className="h-full w-2/5 bg-gradient-to-r from-indigo-500 via-purple-500 to-violet-500 rounded-full animate-[pulseGlow_1.5s_ease-in-out_infinite]" />
            </div>
          </div>

          {/* Skeleton Placeholders for Dual Model Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 animate-pulse">
            <div className="h-80 rounded-3xl bg-slate-100 dark:bg-dark-secondary border border-slate-200 dark:border-dark-border" />
            <div className="h-80 rounded-3xl bg-slate-100 dark:bg-dark-secondary border border-slate-200 dark:border-dark-border" />
          </div>
        </div>
      )}

      {/* =========================================================================
          D. PREDICTION RESULTS PRESENTATION
          ========================================================================= */}
      {predictionResult && !isLoading && (
        <div className="space-y-6 pt-2 animate-fadeIn">
          {/* 1. Out-of-Distribution & Unsupported Image Warning Banner */}
          {predictionResult.ood_detection && (
            <OODWarningBanner oodResult={predictionResult.ood_detection} />
          )}

          {/* 2. Consensus / Divergence Summary Banner */}
          <ComparisonSummary result={predictionResult} />

          {/* 3. Side-by-Side Model Prediction Cards */}
          {(() => {
            const cnnConf = Number(predictionResult.cnn?.confidence || 0);
            const swinConf = Number(predictionResult.swin?.confidence || 0);
            const isEqualConf = Math.abs(cnnConf - swinConf) < 0.001;
            const isCnnHigher = !isEqualConf && cnnConf > swinConf;
            const isSwinHigher = !isEqualConf && swinConf > cnnConf;

            return (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <PredictionCard
                  modelKey="cnn"
                  data={predictionResult.cnn}
                  hasHigherConfidence={isCnnHigher}
                  isEqualConfidence={isEqualConf}
                  isAgreement={predictionResult.agreement}
                  oodResult={predictionResult.ood_detection}
                />
                <PredictionCard
                  modelKey="swin"
                  data={predictionResult.swin}
                  hasHigherConfidence={isSwinHigher}
                  isEqualConfidence={isEqualConf}
                  isAgreement={predictionResult.agreement}
                  oodResult={predictionResult.ood_detection}
                />
              </div>
            );
          })()}

          {/* 4. Overall Model Performance Section (Verified Test Benchmarks) */}
          <OverallPerformanceSection />

          {/* 5. Bottom Action & Navigation Footer */}
          <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-4 p-5 rounded-2xl bg-slate-50/70 dark:bg-dark-secondary/50 border border-slate-200/70 dark:border-dark-border text-xs text-slate-500 dark:text-dark-muted">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
              <span>
                Classification complete. Evaluated using official ImageNet-pretrained frozen backbones.
              </span>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              <button
                type="button"
                onClick={() => onImageSelect(null)}
                className="font-bold text-indigo-600 dark:text-dark-resnet hover:underline flex items-center gap-1"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Try Another Image</span>
              </button>
              <span className="text-slate-300 dark:text-slate-700">•</span>
              <button
                type="button"
                onClick={() => setActiveTab('compare')}
                className="font-semibold text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-dark-resnet flex items-center gap-1 transition-colors"
              >
                <span>View Full Benchmarks</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
