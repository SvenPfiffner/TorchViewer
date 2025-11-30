import { X, Box, Hash, Activity, Maximize2, Layers } from 'lucide-react';
import { type Node } from '@xyflow/react';

interface DetailsPanelProps {
    selectedNode: Node | null;
    onClose: () => void;
}

export default function DetailsPanel({ selectedNode, onClose }: DetailsPanelProps) {
    if (!selectedNode) return null;

    const data = selectedNode.data;
    const metadata = data.metadata as any || {};
    const label = data.label as string;
    const type = metadata.type || 'Operation';

    return (
        <div className="absolute top-4 right-4 w-80 bg-white/80 backdrop-blur-xl rounded-3xl shadow-2xl shadow-slate-500/10 border border-white/50 overflow-hidden flex flex-col animate-in slide-in-from-right-10 fade-in duration-300">
            {/* Header */}
            <div className="p-5 border-b border-slate-100 bg-white/50">
                <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg shadow-blue-500/20 text-white">
                            <Layers size={20} />
                        </div>
                        <div>
                            <h2 className="font-bold text-slate-800 text-lg leading-tight">
                                {label.includes(':') ? label.split(':')[1].trim() : label}
                            </h2>
                            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">
                                {type}
                            </span>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 hover:bg-slate-100 rounded-full text-slate-400 hover:text-slate-600 transition-colors"
                    >
                        <X size={18} />
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="p-5 overflow-y-auto max-h-[calc(100vh-200px)] space-y-6">

                {/* Parameters Section */}
                {Object.keys(metadata).length > 0 && (
                    <div className="space-y-3">
                        <div className="flex items-center gap-2 text-slate-400 text-xs font-bold uppercase tracking-widest">
                            <Hash size={12} />
                            <span>Parameters</span>
                        </div>
                        <div className="grid grid-cols-1 gap-2">
                            {Object.entries(metadata).map(([key, value]) => (
                                key !== 'type' && (
                                    <div key={key} className="flex flex-col p-3 bg-white/60 rounded-xl border border-slate-100 shadow-sm">
                                        <span className="text-xs font-medium text-slate-400 mb-1 capitalize">
                                            {key.replace(/_/g, ' ')}
                                        </span>
                                        <span className="text-sm font-semibold text-slate-700 font-mono">
                                            {String(value)}
                                        </span>
                                    </div>
                                )
                            ))}
                        </div>
                    </div>
                )}

                {/* Raw Data Section (Collapsible-ish) */}
                <div className="space-y-3">
                    <div className="flex items-center gap-2 text-slate-400 text-xs font-bold uppercase tracking-widest">
                        <Activity size={12} />
                        <span>Raw Data</span>
                    </div>
                    <div className="p-3 bg-slate-50/50 rounded-xl border border-slate-100 font-mono text-xs text-slate-600 break-all">
                        {JSON.stringify(data, null, 2)}
                    </div>
                </div>
            </div>
        </div>
    );
}
