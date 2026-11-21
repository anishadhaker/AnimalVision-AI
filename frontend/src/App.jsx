import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import DashboardSection from './components/DashboardSection';
import PredictionSection from './components/PredictionSection';
import ComparisonSection from './components/ComparisonSection';
import DataEfficiencySection from './components/DataEfficiencySection';
import AnalyticsSection from './components/AnalyticsSection';
import AboutSection from './components/AboutSection';
import { AlertCircle, RefreshCw } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('animalvision_sidebar_collapsed') === 'true';
    }
    return false;
  });

  const handleToggleSidebar = (val) => {
    setSidebarCollapsed(val);
    try {
      localStorage.setItem('animalvision_sidebar_collapsed', String(val));
    } catch (e) {}
  };

  // Keyboard shortcut listener for sidebar collapse (Ctrl+\ or Ctrl+B)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === '\\' || e.key === 'b' || e.key === 'B')) {
        e.preventDefault();
        handleToggleSidebar(!sidebarCollapsed);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sidebarCollapsed]);

  // Theme Management (Light / Dark Mode with persistence & system preference)
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('animalvision_theme');
      if (saved === 'dark' || saved === 'light') return saved;
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
    }
    return 'light';
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    try {
      localStorage.setItem('animalvision_theme', theme);
    } catch (e) {}
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const [backendStatus, setBackendStatus] = useState(false);
  const [modelsInfo, setModelsInfo] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [dataEfficiencyData, setDataEfficiencyData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [samples, setSamples] = useState([]);

  // Fetch initial Animals-10 backend data & model metadata
  useEffect(() => {
    async function initData() {
      try {
        const healthRes = await fetch('/api/animals10/health');
        if (healthRes.ok) {
          setBackendStatus(true);
        } else {
          setBackendStatus(false);
        }

        const infoRes = await fetch('/api/animals10/models-info');
        if (infoRes.ok) {
          const info = await infoRes.json();
          setModelsInfo(info);
        }

        const metricsRes = await fetch('/api/animals10/evaluation-metrics');
        if (metricsRes.ok) {
          const m = await metricsRes.json();
          setMetricsData(m);
          if (m.data_efficiency) {
            setDataEfficiencyData(m.data_efficiency);
          }
        }

        const deRes = await fetch('/api/animals10/data-efficiency');
        if (deRes.ok) {
          const de = await deRes.json();
          setDataEfficiencyData(de);
        }

        const histRes = await fetch('/api/animals10/history');
        if (histRes.ok) {
          const h = await histRes.json();
          setHistoryData(h);
        }

        const samplesRes = await fetch('/api/animals10/sample-images');
        if (samplesRes.ok) {
          const s = await samplesRes.json();
          setSamples(s.samples || []);
        }
      } catch (err) {
        console.warn('Backend connection warning:', err);
        setBackendStatus(false);
      }
    }

    initData();
  }, []);

  const handleImageSelect = (file) => {
    if (!file) {
      setSelectedImage(null);
      setPredictionResult(null);
      setErrorMsg(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setSelectedImage({ file, url, name: file.name });
    setPredictionResult(null);
    setErrorMsg(null);
  };

  const handleSelectSample = async (sample) => {
    try {
      if (sample.data_url) {
        const res = await fetch(sample.data_url);
        const blob = await res.blob();
        const file = new File([blob], `${sample.class_name}_sample.jpg`, { type: 'image/jpeg' });

        setSelectedImage({
          file,
          url: sample.data_url,
          name: `${sample.class_name}_sample.jpg`,
        });
      } else {
        // Fallback placeholder canvas generator for mock sample preview
        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, 224, 224);
        ctx.font = '84px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(sample.emoji || '🐾', 112, 112);
        const dataUrl = canvas.toDataURL('image/jpeg');
        const res = await fetch(dataUrl);
        const blob = await res.blob();
        const file = new File([blob], `${sample.class_name}_sample.jpg`, { type: 'image/jpeg' });

        setSelectedImage({
          file,
          url: dataUrl,
          name: `${sample.class_name}_sample.jpg`,
        });
      }
      setPredictionResult(null);
      setErrorMsg(null);
    } catch (err) {
      console.error('Failed to load sample image:', err);
    }
  };

  const handlePredict = async () => {
    if (!selectedImage || !selectedImage.file) return;

    setIsLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedImage.file);

    try {
      const res = await fetch('/api/animals10/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: 'Prediction request failed' }));
        throw new Error(errJson.detail || 'Prediction failed');
      }

      const data = await res.json();
      setPredictionResult(data);
    } catch (err) {
      setErrorMsg(err.message || 'Error communicating with backend API.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-light-bg dark:bg-dark-bg text-light-text dark:text-dark-text flex antialiased font-sans transition-colors duration-150">
      {/* 1. Left Collapsible Sidebar Navigation */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        mobileOpen={mobileOpen}
        setMobileOpen={setMobileOpen}
        backendStatus={backendStatus}
        collapsed={sidebarCollapsed}
        setCollapsed={handleToggleSidebar}
        theme={theme}
        toggleTheme={toggleTheme}
      />

      {/* 2. Main Application Container */}
      <div
        className={`flex-1 flex flex-col min-w-0 transition-all duration-300 ease-in-out ${
          sidebarCollapsed ? 'md:pl-16' : 'md:pl-64'
        }`}
      >
        {/* Top Header */}
        <Header
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          setMobileOpen={setMobileOpen}
          backendStatus={backendStatus}
          theme={theme}
          toggleTheme={toggleTheme}
          collapsed={sidebarCollapsed}
          setCollapsed={handleToggleSidebar}
        />

        {/* Workspace Canvas Container */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto space-y-6">
          {/* Subtle Offline Warning Banner */}
          {!backendStatus && (
            <div className="p-3.5 sm:p-4 rounded-2xl bg-amber-50/90 dark:bg-amber-950/25 border border-amber-200/80 dark:border-amber-800/40 text-amber-900 dark:text-amber-300 text-xs flex items-center justify-between gap-3 shadow-xs">
              <div className="flex items-center gap-2.5">
                <AlertCircle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
                <span>
                  <strong>Backend Offline:</strong> Ensure the FastAPI server is active on{' '}
                  <code className="bg-amber-100 dark:bg-amber-900/40 text-amber-950 dark:text-amber-200 px-1.5 py-0.5 rounded font-mono font-bold">
                    http://127.0.0.1:8000
                  </code>.
                </span>
              </div>
              <button
                onClick={() => window.location.reload()}
                className="inline-flex items-center gap-1 font-semibold text-amber-800 dark:text-amber-300 hover:text-amber-950 dark:hover:text-amber-100 transition shrink-0"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry</span>
              </button>
            </div>
          )}

          {/* Tab 1: Dashboard */}
          {activeTab === 'dashboard' && (
            <DashboardSection
              setActiveTab={setActiveTab}
              modelsInfo={modelsInfo}
              metricsData={metricsData}
            />
          )}

          {/* Tab 2: Animal Prediction (supports 'predict' & legacy 'recognition') */}
          {(activeTab === 'predict' || activeTab === 'recognition' || activeTab === 'recognize') && (
            <PredictionSection
              selectedImage={selectedImage}
              onImageSelect={handleImageSelect}
              onPredict={handlePredict}
              isLoading={isLoading}
              samples={samples}
              onSelectSample={handleSelectSample}
              predictionResult={predictionResult}
              errorMsg={errorMsg}
              setActiveTab={setActiveTab}
              backendStatus={backendStatus}
            />
          )}

          {/* Tab 3: Model Comparison (supports 'compare' & 'comparison') */}
          {(activeTab === 'compare' || activeTab === 'comparison') && (
            <ComparisonSection
              metricsData={metricsData}
              dataEfficiencyData={dataEfficiencyData}
            />
          )}

          {/* Tab 4: Data Efficiency (Dedicated page & experiment view) */}
          {(activeTab === 'efficiency' || activeTab === 'data-efficiency' || activeTab === 'dataEfficiency') && (
            <DataEfficiencySection dataEfficiencyData={dataEfficiencyData} />
          )}

          {/* Tab 5: Analytics */}
          {activeTab === 'analytics' && (
            <AnalyticsSection
              metricsData={metricsData}
              historyData={historyData}
              dataEfficiencyData={dataEfficiencyData}
            />
          )}

          {/* Tab 6: About Project */}
          {activeTab === 'about' && (
            <AboutSection
              metricsData={metricsData}
              modelsInfo={modelsInfo}
            />
          )}
        </main>

        {/* Global SaaS Minimal Footer */}
        <footer className="border-t border-slate-200/80 dark:border-dark-border bg-white dark:bg-dark-card py-4 px-6 text-xs text-slate-500 dark:text-dark-muted flex flex-col sm:flex-row items-center justify-between gap-2.5 transition-colors duration-150">
          <div className="flex items-center gap-2">
            <span className="font-bold text-slate-900 dark:text-dark-text">AnimalVision AI</span>
            <span className="text-slate-300 dark:text-slate-600">•</span>
            <span>ResNet-18 vs Swin-Tiny Benchmark</span>
          </div>
          <div className="font-mono text-[11px] text-slate-400 dark:text-slate-500">
            PyTorch 2.4+ • FastAPI • React 18 • Vite • Tailwind CSS
          </div>
        </footer>
      </div>
    </div>
  );
}
