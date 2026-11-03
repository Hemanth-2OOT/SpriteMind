import React, { useState, useEffect } from 'react';
import {
  Database,
  HardDrive,
  Trash2,
  RefreshCw,
  Archive,
  ShieldAlert,
  Search,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Layers,
  FileCode,
  FileCheck,
} from 'lucide-react';

interface StorageDashboard {
  status: string;
  database_path: string;
  total_files_indexed: number;
  total_disk_bytes: number;
  total_disk_mb: number;
  breakdown: Record<string, { count: number; bytes: number; mb: number }>;
}

export const DataManagerPage: React.FC = () => {
  const [activeSubTab, setActiveSubTab] = useState<'dashboard' | 'explorer' | 'trash' | 'ops'>('dashboard');
  const [dashboard, setDashboard] = useState<StorageDashboard | null>(null);
  const [selectedTable, setSelectedTable] = useState('storage_registry');
  const [searchQuery, setSearchQuery] = useState('');
  const [records, setRecords] = useState<any[]>([]);
  const [trashItems, setTrashItems] = useState<any[]>([]);
  const [backups, setBackups] = useState<any[]>([]);
  const [cleanupScan, setCleanupScan] = useState<any | null>(null);
  const [statusMsg, setStatusMsg] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const loadDashboard = async () => {
    try {
      setIsLoading(true);
      const res = await fetch('/api/storage/dashboard');
      if (res.ok) {
        const data = await res.json();
        setDashboard(data);
      }
      const trashRes = await fetch('/api/storage/trash');
      if (trashRes.ok) {
        setTrashItems(await trashRes.json());
      }
      const backupRes = await fetch('/api/storage/backups');
      if (backupRes.ok) {
        setBackups(await backupRes.json());
      }
    } catch (e) {
      console.error('Failed to load storage dashboard:', e);
    } finally {
      setIsLoading(false);
    }
  };

  const loadTableRecords = async (table: string, query: string) => {
    try {
      const q = query ? `&query=${encodeURIComponent(query)}` : '';
      const res = await fetch(`/api/storage/records?table=${table}${q}`);
      if (res.ok) {
        const data = await res.json();
        setRecords(data.records || []);
      }
    } catch (e) {
      console.error('Failed to load records:', e);
    }
  };

  useEffect(() => {
    loadDashboard();
  }, []);

  useEffect(() => {
    if (activeSubTab === 'explorer') {
      loadTableRecords(selectedTable, searchQuery);
    }
  }, [activeSubTab, selectedTable, searchQuery]);

  const handleRestoreTrash = async (trashId: string) => {
    try {
      const res = await fetch('/api/storage/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trash_id: trashId }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Restore failed');
      }
      setStatusMsg('Asset restored successfully!');
      loadDashboard();
    } catch (e: any) {
      alert(`Restore failed: ${e.message}`);
    }
  };

  const handlePermanentDelete = async (trashId: string) => {
    if (!confirm('Permanently delete this trashed asset? This cannot be undone.')) return;
    try {
      const res = await fetch('/api/storage/permanent_delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trash_id: trashId, confirm: true }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Delete failed');
      }
      setStatusMsg('Item permanently deleted.');
      loadDashboard();
    } catch (e: any) {
      alert(`Delete failed: ${e.message}`);
    }
  };

  const handleEmptyTrash = async () => {
    if (!confirm('Are you sure you want to permanently empty ALL items in trash?')) return;
    try {
      const res = await fetch('/api/storage/empty_trash', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Empty trash failed');
      }
      setStatusMsg('Trash emptied.');
      loadDashboard();
    } catch (e: any) {
      alert(`Empty trash failed: ${e.message}`);
    }
  };

  const handleScanCleanup = async () => {
    try {
      const res = await fetch('/api/storage/cleanup_scan');
      if (res.ok) {
        const data = await res.json();
        setCleanupScan(data);
        setStatusMsg(`Dry-run scan complete: ${data.total_candidates} disposable files found.`);
      }
    } catch (e: any) {
      alert(`Scan failed: ${e.message}`);
    }
  };

  const handleExecuteCleanup = async () => {
    if (!confirm('Execute cleanup of disposable cache files?')) return;
    try {
      const res = await fetch('/api/storage/cleanup_execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm: true }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Cleanup execution failed');
      }
      const data = await res.json();
      setStatusMsg(`Cleaned up ${data.removed_count} files (${data.freed_kb} KB freed).`);
      setCleanupScan(null);
      loadDashboard();
    } catch (e: any) {
      alert(`Cleanup failed: ${e.message}`);
    }
  };

  const handleCreateBackup = async () => {
    try {
      setIsLoading(true);
      const res = await fetch('/api/storage/backup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ include_db: true, include_benchmarks: true, include_assets_meta: true }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Backup failed');
      }
      const data = await res.json();
      setStatusMsg(`Backup created: ${data.filename} (${data.size_kb} KB, SHA-256 verified).`);
      loadDashboard();
    } catch (e: any) {
      alert(`Backup failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSyncDatabase = async () => {
    try {
      setIsLoading(true);
      const res = await fetch('/api/storage/sync', { method: 'POST' });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Sync failed');
      }
      const data = await res.json();
      setStatusMsg(`Database synchronized: ${data.files_indexed} files indexed.`);
      loadDashboard();
    } catch (e: any) {
      alert(`Sync failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-dark-950 p-6 overflow-y-auto custom-scrollbar">
      <div className="max-w-6xl w-full mx-auto space-y-6">
        {/* Header */}
        <div className="flex justify-between items-center pb-4 border-b border-dark-750">
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold text-white flex items-center gap-2">
                <Database className="w-5 h-5 text-purple-400" />
                Data & Storage Manager
              </h1>
              <span className="text-[10px] px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded font-mono">
                Stage 5.5
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Auditable relational storage catalog, immutable model protection, recoverable trash, and verified backups
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleSyncDatabase}
              disabled={isLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-dark-800 hover:bg-dark-750 text-slate-200 text-xs rounded-lg border border-dark-700 transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
              Rescan Disk & Sync
            </button>
            <button
              onClick={handleCreateBackup}
              disabled={isLoading}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium rounded-lg shadow-sm transition-colors"
            >
              <Archive className="w-3.5 h-3.5" />
              Create Verified Backup
            </button>
          </div>
        </div>

        {/* Protection Banner */}
        <div className="p-3.5 bg-purple-950/20 border border-purple-500/30 rounded-xl flex items-center justify-between text-xs text-purple-200">
          <div className="flex items-center gap-2.5">
            <ShieldAlert className="w-4 h-4 text-purple-400 shrink-0" />
            <div>
              <span className="font-semibold text-purple-300">Model & Training Protection Active:</span>
              <span className="ml-1 text-slate-300">
                Checkpoints (`models/checkpoints/`), training datasets (`data/synthetic/`), and benchmark records are strictly immutable. Deleting generated assets will NEVER modify or retrain models.
              </span>
            </div>
          </div>
        </div>

        {statusMsg && (
          <div className="p-3 bg-emerald-950/30 border border-emerald-500/30 rounded-lg text-xs text-emerald-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{statusMsg}</span>
          </div>
        )}

        {/* Navigation Tabs */}
        <div className="flex gap-2 border-b border-dark-800 pb-2">
          <button
            onClick={() => setActiveSubTab('dashboard')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeSubTab === 'dashboard' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-dark-900'
            }`}
          >
            📊 Storage Overview
          </button>
          <button
            onClick={() => setActiveSubTab('explorer')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeSubTab === 'explorer' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-dark-900'
            }`}
          >
            🗄️ Database Explorer
          </button>
          <button
            onClick={() => setActiveSubTab('trash')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeSubTab === 'trash' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-dark-900'
            }`}
          >
            🗑️ Trash Bin ({trashItems.length})
          </button>
          <button
            onClick={() => setActiveSubTab('ops')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
              activeSubTab === 'ops' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-dark-900'
            }`}
          >
            🧹 Cleanup & Maintenance
          </button>
        </div>

        {/* SUBTAB 1: DASHBOARD */}
        {activeSubTab === 'dashboard' && dashboard && (
          <div className="space-y-6">
            <div className="grid grid-cols-4 gap-4">
              <div className="p-4 bg-dark-900 border border-dark-750 rounded-xl">
                <span className="text-xs text-slate-400">Total Indexed Files</span>
                <p className="text-2xl font-bold text-white mt-1">{dashboard.total_files_indexed}</p>
                <span className="text-[10px] text-slate-500 font-mono">SQLite Catalog: data/spritemind.db</span>
              </div>
              <div className="p-4 bg-dark-900 border border-dark-750 rounded-xl">
                <span className="text-xs text-slate-400">Total Storage Footprint</span>
                <p className="text-2xl font-bold text-purple-300 mt-1">{dashboard.total_disk_mb} MB</p>
                <span className="text-[10px] text-slate-500 font-mono">{(dashboard.total_disk_bytes / 1024).toFixed(0)} KB on disk</span>
              </div>
              <div className="p-4 bg-dark-900 border border-dark-750 rounded-xl">
                <span className="text-xs text-slate-400">Model Checkpoints</span>
                <p className="text-2xl font-bold text-indigo-300 mt-1">
                  {dashboard.breakdown['PROTECTED_MODEL']?.mb || 0} MB
                </p>
                <span className="text-[10px] text-emerald-400 font-mono">Protected Immutability</span>
              </div>
              <div className="p-4 bg-dark-900 border border-dark-750 rounded-xl">
                <span className="text-xs text-slate-400">Training Datasets</span>
                <p className="text-2xl font-bold text-cyan-300 mt-1">
                  {dashboard.breakdown['PROTECTED_TRAINING']?.mb || 0} MB
                </p>
                <span className="text-[10px] text-emerald-400 font-mono">Protected Immutability</span>
              </div>
            </div>

            {/* Classification Breakdown Table */}
            <div className="bg-dark-900 border border-dark-750 rounded-xl p-4">
              <h3 className="text-sm font-semibold text-white mb-3">Storage Classification Breakdown</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-dark-850 text-slate-400 border-b border-dark-750">
                    <tr>
                      <th className="p-2.5">Classification Tier</th>
                      <th className="p-2.5">Status</th>
                      <th className="p-2.5">File Count</th>
                      <th className="p-2.5">Disk Size</th>
                      <th className="p-2.5">Policy</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-800 text-slate-300">
                    {Object.entries(dashboard.breakdown).map(([key, val]) => (
                      <tr key={key} className="hover:bg-dark-850/50">
                        <td className="p-2.5 font-mono text-purple-300">{key}</td>
                        <td className="p-2.5">
                          {key.startsWith('PROTECTED') ? (
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-950 text-red-300 border border-red-800 font-mono">
                              IMMUTABLE
                            </span>
                          ) : (
                            <span className="px-1.5 py-0.5 rounded text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-800 font-mono">
                              DELETABLE / TRASHABLE
                            </span>
                          )}
                        </td>
                        <td className="p-2.5">{val.count} files</td>
                        <td className="p-2.5 font-medium">{val.mb} MB</td>
                        <td className="p-2.5 text-slate-400">
                          {key === 'PROTECTED_MODEL' && 'Trained weights (.pt). Cannot be deleted by user.'}
                          {key === 'PROTECTED_TRAINING' && 'Synthetic training pairs (.jsonl). Never modified.'}
                          {key === 'PROTECTED_RESEARCH' && 'Historical benchmark results and research reports.'}
                          {key === 'USER_GENERATED' && 'Safe to move to trash, restore, or export.'}
                          {key === 'TEMPORARY_REGENERABLE' && 'Can be purged anytime without side effects.'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* SUBTAB 2: DATABASE EXPLORER */}
        {activeSubTab === 'explorer' && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <select
                value={selectedTable}
                onChange={(e) => setSelectedTable(e.target.value)}
                className="bg-dark-900 text-slate-200 text-xs px-3 py-2 rounded-lg border border-dark-750 focus:outline-none"
              >
                <option value="storage_registry">storage_registry (All Tracked Files)</option>
                <option value="checkpoints">checkpoints (Model Registry)</option>
                <option value="datasets">datasets (Training Data Sets)</option>
                <option value="assets">assets (User Asset Registry)</option>
                <option value="benchmarks">benchmarks (Evaluation Logs)</option>
                <option value="trash">trash (Deleted Items Log)</option>
                <option value="audit_log">audit_log (Storage Mutation Events)</option>
              </select>

              <div className="flex-1 relative">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder={`Search ${selectedTable}...`}
                  className="w-full bg-dark-900 text-slate-200 text-xs pl-8 pr-3 py-2 rounded-lg border border-dark-750 focus:outline-none focus:border-purple-500"
                />
              </div>
            </div>

            <div className="bg-dark-900 border border-dark-750 rounded-xl overflow-hidden">
              <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                <table className="w-full text-left text-xs">
                  <thead className="bg-dark-850 text-slate-400 sticky top-0 border-b border-dark-750 z-10">
                    <tr>
                      {records.length > 0 &&
                        Object.keys(records[0]).slice(0, 6).map((col) => (
                          <th key={col} className="p-2.5 font-mono capitalize">
                            {col}
                          </th>
                        ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-800 text-slate-300">
                    {records.map((r, idx) => (
                      <tr key={idx} className="hover:bg-dark-850/50">
                        {Object.values(r).slice(0, 6).map((val: any, cidx) => (
                          <td key={cidx} className="p-2.5 truncate max-w-xs font-mono">
                            {typeof val === 'object' ? JSON.stringify(val) : String(val ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {records.length === 0 && (
                      <tr>
                        <td colSpan={6} className="p-8 text-center text-slate-500">
                          No matching records found in `{selectedTable}`
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* SUBTAB 3: TRASH BIN */}
        {activeSubTab === 'trash' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-sm font-semibold text-white">Recoverable Trash Bin</h3>
                <p className="text-xs text-slate-400">
                  User assets moved here remain recoverable until permanently emptied.
                </p>
              </div>
              {trashItems.length > 0 && (
                <button
                  onClick={handleEmptyTrash}
                  className="px-3 py-1.5 bg-rose-950/60 hover:bg-rose-900 text-rose-300 text-xs font-medium rounded-lg border border-rose-800 transition-colors"
                >
                  Empty Trash
                </button>
              )}
            </div>

            {trashItems.length > 0 ? (
              <div className="bg-dark-900 border border-dark-750 rounded-xl overflow-hidden">
                <table className="w-full text-left text-xs">
                  <thead className="bg-dark-850 text-slate-400 border-b border-dark-750">
                    <tr>
                      <th className="p-2.5">Original Path</th>
                      <th className="p-2.5">Trashed Date</th>
                      <th className="p-2.5">Reason</th>
                      <th className="p-2.5">Size</th>
                      <th className="p-2.5 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-dark-800 text-slate-300">
                    {trashItems.map((item) => (
                      <tr key={item.trash_id} className="hover:bg-dark-850/50">
                        <td className="p-2.5 font-mono text-purple-300">{item.original_path}</td>
                        <td className="p-2.5">{item.deleted_at}</td>
                        <td className="p-2.5 text-slate-400">{item.reason || 'User deletion'}</td>
                        <td className="p-2.5 font-mono">{(item.size_bytes / 1024).toFixed(1)} KB</td>
                        <td className="p-2.5 text-right space-x-2">
                          <button
                            onClick={() => handleRestoreTrash(item.trash_id)}
                            className="px-2.5 py-1 bg-emerald-950/60 hover:bg-emerald-900 text-emerald-300 text-[11px] rounded border border-emerald-800"
                          >
                            Restore
                          </button>
                          <button
                            onClick={() => handlePermanentDelete(item.trash_id)}
                            className="px-2.5 py-1 bg-rose-950/60 hover:bg-rose-900 text-rose-300 text-[11px] rounded border border-rose-800"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-12 text-center bg-dark-900/40 rounded-xl border border-dark-800 text-slate-400">
                <Trash2 className="w-8 h-8 mx-auto text-slate-600 mb-2" />
                <p className="text-sm font-medium">Trash is Empty</p>
                <p className="text-xs text-slate-500 mt-1">
                  Deleted user assets will be preserved here and can be restored at any time.
                </p>
              </div>
            )}
          </div>
        )}

        {/* SUBTAB 4: CLEANUP & MAINTENANCE */}
        {activeSubTab === 'ops' && (
          <div className="space-y-6">
            <div className="p-4 bg-dark-900 border border-dark-750 rounded-xl space-y-4">
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="text-sm font-semibold text-white">Storage Optimization & Purge</h3>
                  <p className="text-xs text-slate-400">
                    Find and safely clean compiled Python bytecode (.pyc) and test cache directories without touching assets or models.
                  </p>
                </div>
                <button
                  onClick={handleScanCleanup}
                  className="px-3 py-1.5 bg-dark-800 hover:bg-dark-750 text-slate-200 text-xs rounded-lg border border-dark-700 transition-colors"
                >
                  Dry-Run Scan
                </button>
              </div>

              {cleanupScan && (
                <div className="p-4 bg-dark-950 rounded-lg border border-dark-800 space-y-3">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300">
                      Discovered <strong className="text-purple-300">{cleanupScan.total_candidates}</strong> disposable files ({cleanupScan.reclaimable_kb} KB reclaimable)
                    </span>
                    {cleanupScan.total_candidates > 0 && (
                      <button
                        onClick={handleExecuteCleanup}
                        className="px-3 py-1 bg-amber-600 hover:bg-amber-500 text-white text-xs font-medium rounded shadow-sm"
                      >
                        Execute Purge
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Backups List */}
            <div className="p-4 bg-dark-900 border border-dark-750 rounded-xl space-y-3">
              <h3 className="text-sm font-semibold text-white">Verified Backup Archives</h3>
              <p className="text-xs text-slate-400">
                Self-contained ZIP archives with manifest.json and per-file SHA-256 validation.
              </p>

              {backups.length > 0 ? (
                <div className="divide-y divide-dark-800">
                  {backups.map((b) => (
                    <div key={b.filename} className="py-2.5 flex justify-between items-center text-xs">
                      <div>
                        <span className="font-mono text-purple-300">{b.filename}</span>
                        <div className="text-[10px] text-slate-500">{b.modified_at}</div>
                      </div>
                      <span className="font-mono text-slate-400">{b.size_kb} KB</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-slate-500">No backup archives created yet.</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
