import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { GeneratePage } from './pages/GeneratePage';
import { AnimationStudioPage } from './pages/AnimationStudioPage';
import { DatasetStudioPage } from './pages/DatasetStudioPage';
import { ModelLabPage } from './pages/ModelLabPage';
import { EvaluationLabPage } from './pages/EvaluationLabPage';
import { AssetLibraryPage } from './pages/AssetLibraryPage';
import { DataManagerPage } from './pages/DataManagerPage';
import { AssetStudioPage } from './pages/AssetStudioPage';
import { SystemHealth, ModelsInfo, TrainingStatus } from './types';
import { fetchHealth, fetchModels, fetchTrainingStatus } from './services/api';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('generate');
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [modelsInfo, setModelsInfo] = useState<ModelsInfo | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [selectedEngine, setSelectedEngine] = useState<string>('procedural_baseline');

  const refreshSystemData = async () => {
    try {
      const [h, m, t] = await Promise.all([
        fetchHealth(),
        fetchModels(),
        fetchTrainingStatus(),
      ]);
      setHealth(h);
      setModelsInfo(m);
      setTrainingStatus(t);
    } catch (e) {
      console.error('Failed to fetch system data:', e);
    }
  };

  useEffect(() => {
    refreshSystemData();
    const interval = setInterval(refreshSystemData, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-dark-950 font-sans">
      {/* Top Navbar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        health={health}
        modelsInfo={modelsInfo}
        trainingStatus={trainingStatus}
        selectedEngine={selectedEngine}
        setSelectedEngine={setSelectedEngine}
        onRefreshHealth={refreshSystemData}
      />

      {/* Main Pages */}
      <main className="flex-1 flex overflow-hidden">
        {activeTab === 'generate' && (
          <GeneratePage modelsInfo={modelsInfo} selectedEngine={selectedEngine} />
        )}
        {activeTab === 'animation' && <AnimationStudioPage />}
        {activeTab === 'composition' && <AssetStudioPage />}
        {activeTab === 'dataset' && <DatasetStudioPage />}
        {activeTab === 'model_lab' && (
          <ModelLabPage
            modelsInfo={modelsInfo}
            trainingStatus={trainingStatus}
            onRefreshModels={refreshSystemData}
          />
        )}
        {activeTab === 'evaluation' && <EvaluationLabPage />}
        {activeTab === 'assets' && <AssetLibraryPage />}
        {activeTab === 'data_manager' && <DataManagerPage />}
      </main>
    </div>
  );
};

export default App;
