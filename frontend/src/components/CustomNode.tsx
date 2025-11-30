
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Box, Layers, ArrowRight } from 'lucide-react';

export default function CustomNode({ data, selected }: NodeProps) {
    const metadata = data.metadata as any || {};
    const isLayer = data.op === 'call_module';
    const label = data.label as string;
    const type = metadata.type || 'Op';

    return (
        <div className={`
      px-4 py-3 shadow-lg rounded-xl border transition-all duration-300 min-w-[180px]
      ${selected
                ? 'bg-slate-800/90 border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.3)]'
                : 'bg-slate-900/80 border-white/10 hover:border-white/20 hover:bg-slate-800/80'
            }
      backdrop-blur-md
    `}>
            <Handle type="target" position={Position.Top} className="!bg-blue-500 !w-3 !h-3 !-top-1.5" />

            <div className="flex items-center gap-3">
                <div className={`
          p-2 rounded-lg 
          ${isLayer ? 'bg-blue-500/20 text-blue-400' : 'bg-purple-500/20 text-purple-400'}
        `}>
                    {isLayer ? <Layers size={18} /> : <Box size={18} />}
                </div>

                <div className="flex flex-col">
                    <span className="text-xs font-medium text-slate-400 uppercase tracking-wider">
                        {isLayer ? type : 'Operation'}
                    </span>
                    <span className="text-sm font-bold text-slate-100">
                        {isLayer ? label.split(':')[1]?.trim() : label}
                    </span>
                </div>
            </div>

            {/* Quick stats for Conv/Linear */}
            {isLayer && (metadata.in_features || metadata.in_channels) && (
                <div className="mt-3 pt-3 border-t border-white/5 flex items-center gap-2 text-xs text-slate-400">
                    <span>{metadata.in_features || metadata.in_channels}</span>
                    <ArrowRight size={12} />
                    <span>{metadata.out_features || metadata.out_channels}</span>
                </div>
            )}

            <Handle type="source" position={Position.Bottom} className="!bg-blue-500 !w-3 !h-3 !-bottom-1.5" />
        </div>
    );
}
