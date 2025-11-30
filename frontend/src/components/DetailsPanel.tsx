
import { X } from 'lucide-react';

interface DetailsPanelProps {
    node: any;
    onClose: () => void;
}

export default function DetailsPanel({ node, onClose }: DetailsPanelProps) {
    if (!node) return null;

    const metadata = node.data.metadata || {};
    const isLayer = node.data.op === 'call_module';

    return (
        <div className="h-full bg-[#1a1a1a] border-l border-white/10 flex flex-col w-80 absolute right-0 top-0 z-10 shadow-2xl">
            <div className="p-4 border-b border-white/10 flex items-center justify-between bg-[#242424]">
                <h2 className="font-bold text-lg">Node Details</h2>
                <button onClick={onClose} className="p-1 hover:bg-white/10 rounded-md transition-colors">
                    <X size={18} />
                </button>
            </div>

            <div className="p-4 overflow-y-auto flex-1 space-y-6">
                {/* Header Info */}
                <div>
                    <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Name</div>
                    <div className="text-xl font-mono text-blue-400 break-all">{node.data.label}</div>
                </div>

                <div>
                    <div className="text-xs text-slate-400 uppercase tracking-wider mb-1">Type</div>
                    <div className="font-medium">{metadata.type || node.data.op}</div>
                </div>

                {/* Parameters Table */}
                {isLayer && Object.keys(metadata).length > 1 && (
                    <div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">Parameters</div>
                        <div className="bg-black/20 rounded-lg overflow-hidden border border-white/5">
                            <table className="w-full text-sm">
                                <tbody>
                                    {Object.entries(metadata).map(([key, value]) => {
                                        if (key === 'type') return null;
                                        return (
                                            <tr key={key} className="border-b border-white/5 last:border-0">
                                                <td className="px-3 py-2 text-slate-400 font-mono text-xs">{key}</td>
                                                <td className="px-3 py-2 text-right font-mono text-xs text-slate-200">
                                                    {String(value)}
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Raw Data */}
                <div>
                    <div className="text-xs text-slate-400 uppercase tracking-wider mb-2">Raw Data</div>
                    <pre className="text-[10px] bg-black/40 p-3 rounded-lg overflow-x-auto font-mono text-slate-400">
                        {JSON.stringify(node.data, null, 2)}
                    </pre>
                </div>
            </div>
        </div>
    );
}
