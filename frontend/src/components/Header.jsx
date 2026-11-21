import React from 'react';
import {
  Menu,
  Sun,
  Moon,
  PanelLeft,
  PanelLeftClose,
  ChevronRight,
  Sparkles,
  Cpu,
} from 'lucide-react';

const TAB_METADATA = {
  dashboard: {
    title: 'Dashboard',
    badge: 'Overview',
  },
  predict: {
    title: 'Animal Prediction',
    badge: 'Live Inference',
  },
  recognition: {
    title: 'Animal Prediction',
    badge: 'Live Inference',
  },
  recognize: {
    title: 'Animal Prediction',
    badge: 'Live Inference',
  },
  compare: {
    title: 'Model Comparison',
    badge: 'Benchmark',
  },
  comparison: {
    title: 'Model Comparison',
    badge: 'Benchmark',
  },
  efficiency: {
    title: 'Data Efficiency',
    badge: 'Data Scaling',
  },
  'data-efficiency': {
    title: 'Data Efficiency',
    badge: 'Data Scaling',
  },
  dataEfficiency: {
    title: 'Data Efficiency',
    badge: 'Data Scaling',
  },
  analytics: {
    title: 'Analytics',
    badge: 'Held-out Test (3,534)',
  },
  about: {
    title: 'About Project',
    badge: 'Methodology',
  },
};

export default function Header({
  activeTab,
  setActiveTab,
  setMobileOpen,
  backendStatus,
  theme,
  toggleTheme,
  collapsed,
  setCollapsed,
}) {
  const currentMeta = TAB_METADATA[activeTab] || {
    title: 'AnimalVision AI',
    badge: 'Vision Platform',
  };
  const isDark = theme === 'dark';

  return (
    <header className="sticky top-0 z-30 bg-white/80 dark:bg-[#0B1020]/80 backdrop-blur-xl border-b border-slate-200/70 dark:border-white/[0.08] px-4 sm:px-6 lg:px-8 py-3.5 flex items-center justify-between gap-4 transition-colors duration-200">
      {/* Left: Mobile Toggle / Desktop Collapse Toggle & Breadcrumb Title */}
      <div className="flex items-center gap-3 min-w-0">
        {/* Mobile Menu Hamburger */}
        <button
          onClick={() => setMobileOpen(true)}
          className="md:hidden p-2 -ml-1.5 rounded-xl text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-white/[0.06] transition shrink-0"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Desktop Sidebar Toggle Button */}
        <button
          onClick={() => setCollapsed && setCollapsed(!collapsed)}
          className="hidden md:flex p-1.5 -ml-1 rounded-lg text-slate-400 hover:text-slate-700 dark:text-slate-400 dark:hover:text-white hover:bg-slate-100/80 dark:hover:bg-white/[0.06] transition shrink-0"
          title={collapsed ? 'Expand sidebar (Ctrl+\\)' : 'Collapse sidebar (Ctrl+\\)'}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <PanelLeft className="w-4 h-4" /> : <PanelLeftClose className="w-4 h-4" />}
        </button>

        {/* Vertical subtle divider */}
        <div className="hidden sm:block h-4 w-px bg-slate-200/80 dark:bg-white/[0.1]" />

        {/* Breadcrumb Navigation */}
        <nav aria-label="Breadcrumbs" className="flex items-center gap-2 min-w-0">
          <button
            type="button"
            onClick={() => setActiveTab('dashboard')}
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-400 dark:text-slate-400 hover:text-indigo-600 dark:hover:text-indigo-300 transition truncate group"
          >
            <Sparkles className="w-3 h-3 text-indigo-500 opacity-70 group-hover:opacity-100 transition-opacity" />
            <span>AnimalVision</span>
          </button>
          <ChevronRight className="w-3.5 h-3.5 text-slate-300 dark:text-slate-600 shrink-0" />
          <h1 className="text-sm sm:text-base font-bold text-slate-900 dark:text-white tracking-tight truncate">
            {currentMeta.title}
          </h1>
          {/* Subtle contextual badge */}
          {currentMeta.badge && (
            <span className="hidden lg:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 dark:bg-white/[0.05] text-slate-600 dark:text-slate-400 border border-slate-200/70 dark:border-white/[0.06]">
              {currentMeta.badge}
            </span>
          )}
        </nav>
      </div>

      {/* Right: Dual Model Pill, Real-time Status & Theme Toggle */}
      <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
        {/* Dual Model Pill (Desktop) */}
        <div className="hidden md:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-50 dark:bg-white/[0.04] border border-slate-200/80 dark:border-white/[0.06] text-[11px] text-slate-600 dark:text-slate-300 font-medium">
          <Cpu className="w-3.5 h-3.5 text-indigo-500" />
          <span className="font-semibold text-slate-800 dark:text-slate-200">ResNet-18</span>
          <span className="text-slate-400 dark:text-slate-500 text-[10px]">vs</span>
          <span className="font-semibold text-slate-800 dark:text-slate-200">Swin-Tiny</span>
        </div>

        {/* Real-time Status indicator */}
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-50 dark:bg-white/[0.04] border border-slate-200/80 dark:border-white/[0.06] text-[11px] font-medium text-slate-600 dark:text-slate-300">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              backendStatus ? 'bg-emerald-500 animate-pulse shadow-xs shadow-emerald-500/60' : 'bg-rose-500'
            }`}
          />
          <span className="hidden sm:inline">
            {backendStatus ? 'FastAPI Online' : 'Offline'}
          </span>
        </div>

        {/* Theme Toggle Button */}
        <button
          onClick={toggleTheme}
          type="button"
          aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          className="p-2 rounded-xl bg-slate-100/80 dark:bg-white/[0.06] hover:bg-slate-200/80 dark:hover:bg-white/[0.1] border border-slate-200/80 dark:border-white/[0.08] text-slate-700 dark:text-slate-200 transition-all duration-200 flex items-center justify-center relative group shadow-2xs"
        >
          {isDark ? (
            <Sun className="w-4 h-4 text-amber-400 transition-transform duration-300 group-hover:rotate-90" />
          ) : (
            <Moon className="w-4 h-4 text-indigo-600 transition-transform duration-300 group-hover:-rotate-12" />
          )}
          <span className="sr-only">Toggle theme</span>
        </button>
      </div>
    </header>
  );
}
