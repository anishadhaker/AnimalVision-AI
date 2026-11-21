import React from 'react';

export default function StatusPill({
  active = true,
  activeText = 'Engine Online',
  inactiveText = 'Offline',
  detail = '',
  className = '',
}) {
  return (
    <div
      className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-medium border transition-colors ${
        active
          ? 'bg-emerald-50/80 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400 border-emerald-200/80 dark:border-emerald-800/40'
          : 'bg-rose-50/80 dark:bg-rose-950/30 text-rose-700 dark:text-rose-400 border-rose-200/80 dark:border-rose-800/40'
      } ${className}`}
    >
      <span className="relative flex h-2 w-2 shrink-0">
        {active && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            active ? 'bg-emerald-500' : 'bg-rose-500'
          }`}
        />
      </span>
      <span className="truncate">{active ? activeText : inactiveText}</span>
      {detail && (
        <>
          <span className="text-slate-300 dark:text-slate-600">•</span>
          <span className="text-slate-500 dark:text-slate-400 font-mono text-[10px] truncate">{detail}</span>
        </>
      )}
    </div>
  );
}
