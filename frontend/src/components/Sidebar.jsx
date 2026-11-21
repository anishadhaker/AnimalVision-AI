import React from 'react';
import {
  LayoutDashboard,
  Sparkles,
  Scale,
  Database,
  BarChart3,
  BookOpen,
  X,
  PanelLeftClose,
  PanelLeft,
  ChevronRight,
  Activity,
  Layers,
} from 'lucide-react';

const NAV_ITEMS = [
  {
    id: 'dashboard',
    altIds: [],
    label: 'Dashboard',
    icon: LayoutDashboard,
    badge: null,
  },
  {
    id: 'predict',
    altIds: ['recognition', 'recognize'],
    label: 'Animal Prediction',
    icon: Sparkles,
    badge: 'Live',
  },
  {
    id: 'compare',
    altIds: ['comparison'],
    label: 'Model Comparison',
    icon: Scale,
    badge: null,
  },
  {
    id: 'efficiency',
    altIds: ['data-efficiency', 'dataEfficiency'],
    label: 'Data Efficiency',
    icon: Database,
    badge: null,
  },
  {
    id: 'analytics',
    altIds: [],
    label: 'Analytics',
    icon: BarChart3,
    badge: null,
  },
  {
    id: 'about',
    altIds: [],
    label: 'About Project',
    icon: BookOpen,
    badge: null,
  },
];

export default function Sidebar({
  activeTab,
  setActiveTab,
  mobileOpen,
  setMobileOpen,
  backendStatus,
  collapsed,
  setCollapsed,
  theme,
  toggleTheme,
}) {
  const isItemActive = (item) => {
    if (activeTab === item.id) return true;
    if (item.altIds && item.altIds.includes(activeTab)) return true;
    return false;
  };

  const handleNav = (id) => {
    setActiveTab(id);
    if (setMobileOpen) setMobileOpen(false);
  };

  const toggleCollapsed = () => {
    if (setCollapsed) {
      setCollapsed(!collapsed);
    }
  };

  const renderContent = (isMobile = false) => {
    const isMini = !isMobile && collapsed;

    return (
      <div className="flex flex-col h-full bg-white dark:bg-[#0E1528] border-r border-slate-200/80 dark:border-white/[0.08] select-none transition-colors duration-200">
        {/* Brand Header */}
        <div
          className={`flex items-center border-b border-slate-100 dark:border-white/[0.06] transition-all duration-200 ${
            isMini ? 'p-3 justify-center' : 'px-4 py-4 justify-between'
          }`}
        >
          <button
            type="button"
            onClick={() => handleNav('dashboard')}
            className={`flex items-center text-left group focus:outline-none ${isMini ? 'justify-center' : 'gap-3 min-w-0'}`}
            title="AnimalVision AI — Dashboard"
          >
            {/* Luminous AI Logo Glyph */}
            <div className="relative flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-violet-500 shadow-md shadow-indigo-500/25 ring-1 ring-white/20 dark:ring-white/10 group-hover:scale-105 group-hover:shadow-indigo-500/40 transition-all duration-200 shrink-0">
              {/* Geometric AI Vision Sparkle Glyph */}
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className="w-5 h-5 text-white"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 2L14.4 8.6L21 11L14.4 13.4L12 20L9.6 13.4L3 11L9.6 8.6L12 2Z" fill="currentColor" fillOpacity="0.25" />
                <circle cx="12" cy="11" r="2.5" fill="white" />
              </svg>
              {/* Subtle ambient light dot */}
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-violet-300 ring-2 ring-white dark:ring-[#0E1528]" />
            </div>

            {!isMini && (
              <div className="truncate">
                <div className="flex items-center gap-1.5">
                  <span className="text-[14px] font-bold text-slate-900 dark:text-white tracking-tight">
                    AnimalVision
                  </span>
                  <span className="px-1.5 py-0.5 rounded-md bg-gradient-to-r from-indigo-500/10 to-violet-500/10 dark:from-indigo-400/20 dark:to-violet-400/20 text-indigo-600 dark:text-indigo-300 text-[10px] font-extrabold uppercase tracking-wider border border-indigo-200/60 dark:border-indigo-500/30">
                    AI
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 dark:text-slate-400 font-medium truncate">
                  CNN vs Swin Transformer
                </p>
              </div>
            )}
          </button>

          {/* Desktop Collapse Toggle in expanded state */}
          {!isMobile && !isMini && (
            <button
              onClick={toggleCollapsed}
              type="button"
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100/80 dark:hover:bg-white/[0.06] transition shrink-0"
              title="Collapse sidebar (Ctrl+\)"
              aria-label="Collapse sidebar"
            >
              <PanelLeftClose className="w-4 h-4" />
            </button>
          )}

          {/* Mobile Close Button */}
          {isMobile && setMobileOpen && (
            <button
              onClick={() => setMobileOpen(false)}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-white/[0.06] transition"
              aria-label="Close sidebar"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Navigation Items (6 required items) */}
        <nav className={`flex-1 py-3 space-y-1 overflow-y-auto ${isMini ? 'px-2' : 'px-3'}`}>
          {!isMini && (
            <div className="px-3 pb-2 pt-1 text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
              Core Platform
            </div>
          )}

          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isItemActive(item);

            return (
              <div key={item.id} className="relative group">
                <button
                  type="button"
                  onClick={() => handleNav(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={`w-full flex items-center rounded-xl text-xs transition-all duration-200 relative ${
                    isMini ? 'p-2.5 justify-center' : 'gap-3 px-3 py-2.5'
                  } ${
                    active
                      ? 'bg-gradient-to-r from-indigo-50/90 via-violet-50/50 to-transparent dark:from-indigo-950/50 dark:via-violet-950/30 dark:to-transparent text-indigo-900 dark:text-indigo-200 font-semibold shadow-2xs'
                      : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100/70 dark:hover:bg-white/[0.04] font-medium'
                  }`}
                >
                  {/* Active Accent Indicator Bar */}
                  {active && (
                    <span
                      className={`absolute rounded-r-md bg-gradient-to-b from-indigo-500 to-violet-500 shadow-sm shadow-indigo-500/40 ${
                        isMini
                          ? 'left-0 top-2 bottom-2 w-1'
                          : 'left-0 top-1.5 bottom-1.5 w-1'
                      }`}
                    />
                  )}

                  <Icon
                    className={`w-4 h-4 shrink-0 transition-transform duration-200 ${
                      active
                        ? 'text-indigo-600 dark:text-indigo-400 scale-105'
                        : 'text-slate-400 dark:text-slate-400 group-hover:text-slate-700 dark:group-hover:text-slate-200'
                    }`}
                  />

                  {!isMini && (
                    <span className="truncate text-left flex-1 min-w-0">
                      {item.label}
                    </span>
                  )}

                  {/* Optional Live badge or active chevron */}
                  {!isMini && item.badge && !active && (
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider bg-indigo-50 dark:bg-indigo-950/80 text-indigo-600 dark:text-indigo-300 border border-indigo-200/50 dark:border-indigo-800/40">
                      {item.badge}
                    </span>
                  )}

                  {active && !isMini && (
                    <ChevronRight className="w-3.5 h-3.5 text-indigo-500/80 dark:text-indigo-400 shrink-0" />
                  )}
                </button>

                {/* Floating Tooltip in Collapsed Mode */}
                {isMini && (
                  <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center z-50 pointer-events-none animate-fadeIn">
                    <div className="bg-slate-900 text-slate-50 dark:bg-slate-100 dark:text-slate-900 text-xs font-medium px-2.5 py-1.5 rounded-lg shadow-xl whitespace-nowrap flex items-center gap-2 border border-slate-700/50 dark:border-slate-200/50">
                      <span>{item.label}</span>
                      {item.badge && (
                        <span className="px-1 py-0.2 rounded text-[9px] font-bold uppercase bg-indigo-500 text-white dark:bg-indigo-600">
                          {item.badge}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </nav>

        {/* Footer Area: High-tech AI Architecture Card */}
        <div
          className={`border-t border-slate-100 dark:border-white/[0.06] transition-all duration-200 ${
            isMini ? 'p-2 flex flex-col items-center gap-2' : 'p-3'
          }`}
        >
          {isMini ? (
            <>
              {/* Mini Status Dot with rich tooltip */}
              <div
                className="p-2 rounded-xl hover:bg-slate-100 dark:hover:bg-white/[0.06] transition cursor-help flex items-center justify-center relative group"
                title={backendStatus ? 'FastAPI Online' : 'FastAPI Offline'}
              >
                <span
                  className={`w-2.5 h-2.5 rounded-full ${
                    backendStatus
                      ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50 animate-pulse'
                      : 'bg-rose-500 shadow-sm shadow-rose-500/50'
                  }`}
                />
                <div className="absolute left-full ml-3 top-1/2 -translate-y-1/2 hidden group-hover:flex items-center z-50 pointer-events-none animate-fadeIn">
                  <div className="bg-slate-900 text-slate-50 dark:bg-slate-100 dark:text-slate-900 text-xs font-medium px-2.5 py-1.5 rounded-lg shadow-xl whitespace-nowrap border border-slate-700/50 dark:border-slate-200/50">
                    {backendStatus ? 'FastAPI Online • Ready' : 'FastAPI Offline'}
                  </div>
                </div>
              </div>

              {/* Mini expand button */}
              <button
                onClick={toggleCollapsed}
                type="button"
                className="p-2 rounded-xl text-slate-400 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-white/[0.06] transition"
                title="Expand sidebar (Ctrl+\)"
                aria-label="Expand sidebar"
              >
                <PanelLeft className="w-4 h-4" />
              </button>
            </>
          ) : (
            <div className="rounded-xl bg-slate-50/90 dark:bg-white/[0.03] border border-slate-200/70 dark:border-white/[0.06] p-2.5 space-y-2">
              <div className="flex items-center justify-between text-[11px]">
                <div className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      backendStatus
                        ? 'bg-emerald-500 shadow-xs shadow-emerald-500/60 animate-pulse'
                        : 'bg-rose-500'
                    }`}
                  />
                  <span>{backendStatus ? 'FastAPI Online' : 'Engine Offline'}</span>
                </div>
                <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">
                  v2.0
                </span>
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 flex items-center justify-between pt-1 border-t border-slate-200/50 dark:border-white/[0.04]">
                <span>Dual Models</span>
                <span className="font-medium text-slate-700 dark:text-slate-300">ResNet + Swin</span>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <>
      {/* Desktop Persistent Sidebar */}
      <aside
        className={`hidden md:block fixed inset-y-0 left-0 z-20 transition-all duration-300 ease-in-out ${
          collapsed ? 'w-16' : 'w-64'
        }`}
      >
        {renderContent(false)}
      </aside>

      {/* Mobile Drawer Overlay */}
      {mobileOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div
            className="fixed inset-0 bg-slate-900/50 dark:bg-black/70 backdrop-blur-xs transition-opacity"
            onClick={() => setMobileOpen(false)}
          />
          <div className="relative w-64 max-w-xs flex-1 flex flex-col bg-white dark:bg-[#0E1528] shadow-2xl z-10 animate-fadeIn">
            {renderContent(true)}
          </div>
        </div>
      )}
    </>
  );
}
