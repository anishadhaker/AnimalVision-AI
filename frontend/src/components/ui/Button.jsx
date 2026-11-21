import React from 'react';
import { Loader2 } from 'lucide-react';

const VARIANTS = {
  primary:
    'bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white font-medium shadow-xs hover:shadow-indigo-500/20 border border-transparent focus:outline-none focus:ring-2 focus:ring-indigo-500/40 focus:ring-offset-1 dark:focus:ring-offset-dark-bg',
  secondary:
    'bg-white dark:bg-dark-card hover:bg-slate-50 dark:hover:bg-dark-elevated text-slate-700 dark:text-dark-text border border-slate-200/90 dark:border-dark-border shadow-xs focus:outline-none focus:ring-2 focus:ring-slate-400/20',
  outline:
    'bg-transparent hover:bg-slate-100 dark:hover:bg-dark-elevated text-slate-700 dark:text-dark-text border border-slate-200 dark:border-dark-border focus:outline-none focus:ring-2 focus:ring-slate-400/20',
  ghost:
    'bg-transparent hover:bg-slate-100 dark:hover:bg-dark-elevated text-slate-600 dark:text-dark-muted hover:text-slate-900 dark:hover:text-dark-text border border-transparent',
  indigo:
    'bg-indigo-50 dark:bg-indigo-950/50 hover:bg-indigo-100 dark:hover:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 border border-indigo-200/80 dark:border-indigo-800/50 font-medium',
  violet:
    'bg-violet-600 hover:bg-violet-700 active:bg-violet-800 text-white font-medium shadow-xs hover:shadow-violet-500/20 border border-transparent focus:outline-none focus:ring-2 focus:ring-violet-500/40 focus:ring-offset-1 dark:focus:ring-offset-dark-bg',
};

const SIZES = {
  xs: 'text-xs px-2.5 py-1 rounded-lg gap-1.5',
  sm: 'text-xs px-3 py-1.5 rounded-xl gap-1.5',
  md: 'text-xs sm:text-sm px-3.5 py-2 rounded-xl gap-2',
  lg: 'text-sm px-4 py-2.5 rounded-xl gap-2',
  icon: 'p-2 rounded-xl',
};

export default function Button({
  children,
  variant = 'secondary',
  size = 'md',
  className = '',
  icon: Icon = null,
  iconRight: IconRight = null,
  loading = false,
  disabled = false,
  type = 'button',
  ...props
}) {
  const variantClass = VARIANTS[variant] || VARIANTS.secondary;
  const sizeClass = SIZES[size] || SIZES.md;

  return (
    <button
      type={type}
      disabled={disabled || loading}
      className={`inline-flex items-center justify-center font-medium transition-all duration-150 select-none disabled:opacity-50 disabled:pointer-events-none active:scale-[0.98] ${variantClass} ${sizeClass} ${className}`}
      {...props}
    >
      {loading ? (
        <Loader2 className="w-4 h-4 animate-spin shrink-0" />
      ) : Icon ? (
        <Icon className="w-4 h-4 shrink-0" />
      ) : null}
      {children}
      {!loading && IconRight && <IconRight className="w-4 h-4 shrink-0" />}
    </button>
  );
}
