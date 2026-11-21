import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X, RefreshCw, Trash2, CheckCircle2, Sparkles, FolderOpen } from 'lucide-react';
import { Card, CardContent } from './ui/Card';
import Badge from './ui/Badge';
import Button from './ui/Button';
import { ANIMALS10_SAMPLES, ANIMALS10_EMOJIS } from '../data/animals10MockData';

export default function ImageUploader({
  selectedImage,
  onImageSelect,
  samples = [],
  onSelectSample,
  isLoading = false,
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelect(file);
      }
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onImageSelect(e.target.files[0]);
    }
  };

  const openFileDialog = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  // 10 canonical Animals-10 sample images
  const activeSamples = samples && samples.length === 10 ? samples : ANIMALS10_SAMPLES;

  // Format file size nicely
  const formattedSize = selectedImage?.file?.size
    ? `${(selectedImage.file.size / 1024).toFixed(1)} KB`
    : 'Native Resolution';

  return (
    <div className="space-y-4">
      {/* Hidden native file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/jpg"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Upload Drop Zone / Active Image Preview */}
      {!selectedImage ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={openFileDialog}
          className={`relative group rounded-3xl p-8 sm:p-12 text-center transition-all duration-200 cursor-pointer flex flex-col items-center justify-center min-h-[250px] border-2 border-dashed ${
            isDragOver
              ? 'border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/40 scale-[1.005] shadow-lg shadow-indigo-500/10'
              : 'border-slate-300/80 dark:border-dark-border hover:border-indigo-400 dark:hover:border-indigo-500/70 bg-gradient-to-b from-slate-50/50 to-white dark:from-dark-card dark:to-dark-secondary hover:shadow-md shadow-xs'
          }`}
        >
          {/* Animated decorative center icon */}
          <div className="w-16 h-16 rounded-3xl bg-gradient-to-br from-indigo-50 to-violet-50 dark:from-indigo-950/70 dark:to-violet-950/70 border border-indigo-100 dark:border-indigo-800/40 flex items-center justify-center text-indigo-600 dark:text-dark-resnet mb-4 group-hover:scale-110 group-hover:shadow-md transition-all duration-200 shadow-2xs">
            <UploadCloud className="w-8 h-8 text-indigo-600 dark:text-dark-resnet" />
          </div>

          <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-dark-text tracking-tight">
            Drag and drop your animal image here
          </h3>
          <p className="text-xs sm:text-sm text-slate-500 dark:text-dark-muted mt-1.5 max-w-md leading-relaxed">
            Upload any natural animal photo to evaluate ResNet-18 vs. Swin-Tiny. Supported formats: <strong className="font-mono text-slate-700 dark:text-slate-300">JPG, PNG, WebP</strong> (up to 10MB).
          </p>

          <div className="mt-5 flex items-center gap-3">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                openFileDialog();
              }}
              className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 dark:bg-indigo-600 dark:hover:bg-indigo-500 text-white font-semibold text-xs sm:text-sm shadow-sm transition-all flex items-center gap-2 active:scale-[0.98]"
            >
              <FolderOpen className="w-4 h-4 text-indigo-200" />
              <span>Browse Local Image</span>
            </button>
          </div>
        </div>
      ) : (
        /* Image Preview Card when an image is loaded */
        <Card className="overflow-hidden border-slate-200/90 dark:border-dark-border shadow-xs">
          <CardContent className="p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row items-center gap-6">
              {/* Preview Image Frame */}
              <div className="relative shrink-0 w-36 h-36 sm:w-48 sm:h-48 rounded-2xl overflow-hidden bg-slate-100 dark:bg-dark-secondary border border-slate-200 dark:border-dark-border shadow-inner group">
                <img
                  src={selectedImage.url}
                  alt="Selected input"
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />
                <span className="absolute bottom-2.5 left-2.5 text-[10px] font-semibold bg-slate-900/80 backdrop-blur-md text-white px-2 py-0.5 rounded-md font-mono shadow-xs">
                  Standard 224×224 Input
                </span>
              </div>

              {/* Info and Change Action */}
              <div className="flex-1 text-center sm:text-left space-y-2.5 w-full">
                <div className="flex flex-wrap items-center justify-center sm:justify-start gap-2">
                  <Badge variant="emerald" dot={true}>
                    Ready for Dual Model Inference
                  </Badge>
                  <span className="text-[11px] font-mono text-slate-400 dark:text-slate-500">
                    {formattedSize}
                  </span>
                </div>

                <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-dark-text truncate max-w-md">
                  {selectedImage.name || 'animal_sample.jpg'}
                </h3>
                <p className="text-xs text-slate-500 dark:text-dark-muted leading-relaxed max-w-xl">
                  Image is loaded and ready. Clicking Predict will pass this photo through both ResNet-18 (local 3×3 convolutions) and Swin-Tiny (shifted-window self-attention) to compute softmax certainty.
                </p>

                <div className="pt-2 flex flex-wrap items-center justify-center sm:justify-start gap-2.5">
                  <button
                    type="button"
                    onClick={openFileDialog}
                    disabled={isLoading}
                    className="px-3.5 py-1.5 rounded-xl bg-slate-100 dark:bg-dark-secondary hover:bg-slate-200 dark:hover:bg-dark-elevated text-slate-700 dark:text-dark-text border border-slate-200 dark:border-dark-border text-xs font-semibold flex items-center gap-1.5 transition active:scale-[0.98]"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-slate-500" />
                    <span>Change Image</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onImageSelect(null)}
                    disabled={isLoading}
                    className="px-3.5 py-1.5 rounded-xl bg-rose-50 dark:bg-rose-950/40 hover:bg-rose-100 dark:hover:bg-rose-900/50 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800/40 text-xs font-semibold flex items-center gap-1.5 transition active:scale-[0.98]"
                  >
                    <Trash2 className="w-3.5 h-3.5 text-rose-500" />
                    <span>Remove</span>
                  </button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 1-Click Animals-10 Sample Thumbnails Selector */}
      {activeSamples && activeSamples.length > 0 && (
        <Card className="p-4 sm:p-5 border-slate-200/80 dark:border-dark-border shadow-xs">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-dark-border/80">
            <div className="flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5 text-indigo-600 dark:text-dark-resnet" />
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600 dark:text-slate-300">
                1-Click Animals-10 Test Samples
              </span>
            </div>
            <span className="text-[10px] text-slate-400 dark:text-slate-500 hidden sm:inline font-mono">
              10 held-out classes • Native resolution
            </span>
          </div>

          <div className="grid grid-cols-5 sm:grid-cols-10 gap-2 pt-3">
            {activeSamples.map((sample) => {
              const isSelected = selectedImage?.name?.toLowerCase().includes(sample.class_name.toLowerCase());
              const emoji = sample.emoji || ANIMALS10_EMOJIS[sample.class_name?.toLowerCase()] || '🐾';
              return (
                <button
                  key={sample.target_idx}
                  type="button"
                  onClick={() => onSelectSample(sample)}
                  disabled={isLoading}
                  className={`group relative p-1.5 rounded-2xl border flex flex-col items-center text-center transition-all duration-150 active:scale-[0.96] ${
                    isSelected
                      ? 'border-indigo-600 dark:border-dark-resnet bg-indigo-50/80 dark:bg-indigo-950/70 ring-2 ring-indigo-500/20 shadow-xs'
                      : 'border-slate-200 dark:border-dark-border hover:border-slate-300 dark:hover:border-slate-600 bg-slate-50/70 dark:bg-dark-secondary hover:bg-white dark:hover:bg-dark-card'
                  }`}
                  title={`Test Sample: ${sample.class_name} (Class #${sample.target_idx})`}
                >
                  <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl overflow-hidden bg-white dark:bg-dark-card border border-slate-200 dark:border-dark-border flex items-center justify-center shrink-0 shadow-2xs">
                    {sample.thumbnail_url || sample.data_url ? (
                      <img
                        src={sample.thumbnail_url || sample.data_url}
                        alt={sample.class_name}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
                      />
                    ) : (
                      <span className="text-xl select-none">{emoji}</span>
                    )}
                  </div>
                  <span className="text-[10px] font-bold text-slate-700 dark:text-slate-300 capitalize mt-1 truncate max-w-full">
                    {sample.class_name}
                  </span>
                </button>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
