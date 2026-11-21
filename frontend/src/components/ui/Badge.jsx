import React from 'react';

const VARIANTS = {
  default:
    'bg-slate-100 dark:bg-dark-elevated text-slate-700 dark:text-dark-text border-slate-200/80 dark:border-dark-border',
  indigo:
    'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-dark-resnet border-indigo-200/70 dark:border-indigo-800/50',
  violet:
    'bg-violet-50 dark:bg-violet-950/60 text-violet-700 dark:text-dark-swin border-violet-200/70 dark:border-violet-800/50',
  emerald:
    'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-400 border-emerald-200/70 dark:border-emerald-800/50',
  amber:
    'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-400 border-amber-200/70 dark:border-amber-800/50',
  rose:
    'bg-rose-50 dark:bg-rose-950/50 text-rose-700 dark:text-rose-400 border-rose-200/70 dark:border-rose-800/50',
  outline:
    'bg-transparent text-slate-600 dark:text-dark-muted border-slate-300 dark:border-dark-border',
};

const SIZES = {
  xs: 'text-[10px] px-1.5 py-0.5',
  sm: 'text-[11px] px-2 py-0.5',
  md: 'text-xs px-2.5 py-1',
};

export default function Badge({
  children,
  variant = 'default',
  size = 'sm',
  className = '',
  dot = false,
  dotColor = null,
  ...props
}) {
  const variantClass = VARIANTS[variant] || VARIANTS.default;
  const sizeClass = SIZES[size] || SIZES.sm;

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium rounded-full border tracking-wide transition-colors ${variantClass} ${sizeClass} ${className}`}
      {...props}
    >
      {dot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            dotColor
              ? dotColor
              : variant === 'emerald'
              ? 'bg-emerald-500'
              : variant === 'rose'
              ? 'bg-rose-500'
              : variant === 'amber'
              ? 'bg-amber-500'
              : variant === 'indigo'
              ? 'bg-indigo-500'
              : variant === 'violet'
              ? 'bg-violet-500'
              : 'bg-slate-400'
          }`}
        />
      )}
      {children}
    </span>
  );
}
