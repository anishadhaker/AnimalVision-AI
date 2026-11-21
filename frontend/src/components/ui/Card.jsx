import React from 'react';

export function Card({ children, className = '', hover = false, elevated = false, ...props }) {
  return (
    <div
      className={`rounded-2xl transition-all duration-150 ${
        elevated
          ? 'ai-card-elevated'
          : 'bg-white dark:bg-dark-card border border-slate-200/80 dark:border-dark-border shadow-xs'
      } ${
        hover
          ? 'hover:border-slate-300 dark:hover:border-slate-700 hover:shadow-sm dark:hover:shadow-card-dark'
          : ''
      } ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className = '', ...props }) {
  return (
    <div className={`p-4 sm:p-5 border-b border-slate-100 dark:border-dark-border/80 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className = '', ...props }) {
  return (
    <h3 className={`text-sm sm:text-base font-bold text-slate-900 dark:text-dark-text tracking-tight ${className}`} {...props}>
      {children}
    </h3>
  );
}

export function CardDescription({ children, className = '', ...props }) {
  return (
    <p className={`text-xs text-slate-500 dark:text-dark-muted mt-0.5 leading-relaxed ${className}`} {...props}>
      {children}
    </p>
  );
}

export function CardContent({ children, className = '', ...props }) {
  return (
    <div className={`p-4 sm:p-5 ${className}`} {...props}>
      {children}
    </div>
  );
}

export function CardFooter({ children, className = '', ...props }) {
  return (
    <div className={`p-4 sm:p-5 border-t border-slate-100 dark:border-dark-border/80 bg-slate-50/50 dark:bg-dark-elevated/40 rounded-b-2xl ${className}`} {...props}>
      {children}
    </div>
  );
}

export default Card;
