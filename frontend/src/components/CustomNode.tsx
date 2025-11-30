import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Box, Layers, ArrowRight, Grid3x3, Share2, Activity, Minimize2, Sliders, Zap } from 'lucide-react';

const getLayerStyle = (type: string) => {
  const t = type.toLowerCase();
  if (t.includes('conv')) return {
    bg: 'bg-emerald-900/80',
    border: 'border-emerald-500/30',
    hover: 'hover:border-emerald-500/60',
    iconBg: 'bg-emerald-500/20',
    iconColor: 'text-emerald-400',
    icon: Grid3x3
  };
  if (t.includes('linear')) return {
    bg: 'bg-blue-900/80',
    border: 'border-blue-500/30',
    hover: 'hover:border-blue-500/60',
    iconBg: 'bg-blue-500/20',
    iconColor: 'text-blue-400',
    icon: Share2
  };
  if (t.includes('relu') || t.includes('sigmoid') || t.includes('tanh')) return {
    bg: 'bg-rose-900/80',
    border: 'border-rose-500/30',
    hover: 'hover:border-rose-500/60',
    iconBg: 'bg-rose-500/20',
    iconColor: 'text-rose-400',
    icon: Activity
  };
  if (t.includes('pool')) return {
    bg: 'bg-cyan-900/80',
    border: 'border-cyan-500/30',
    hover: 'hover:border-cyan-500/60',
    iconBg: 'bg-cyan-500/20',
    iconColor: 'text-cyan-400',
    icon: Minimize2
  };
  if (t.includes('norm')) return {
    bg: 'bg-amber-900/80',
    border: 'border-amber-500/30',
    hover: 'hover:border-amber-500/60',
    iconBg: 'bg-amber-500/20',
    iconColor: 'text-amber-400',
    icon: Sliders
  };

  // Default for other layers
  return {
    bg: 'bg-slate-900/80',
    border: 'border-white/10',
    hover: 'hover:border-white/20',
    iconBg: 'bg-purple-500/20',
    iconColor: 'text-purple-400',
    icon: Layers
  };
};

export default function CustomNode({ data, selected }: NodeProps) {
  const metadata = data.metadata as any || {};
  const isLayer = data.op === 'call_module';
  const label = data.label as string;
  const type = metadata.type || 'Op';

  const style = isLayer ? getLayerStyle(type) : {
    bg: 'bg-zinc-900/90',
    border: 'border-zinc-700',
    hover: 'hover:border-zinc-500',
    iconBg: 'bg-zinc-700',
    iconColor: 'text-zinc-400',
    icon: Box
  };

  const Icon = style.icon;

  return (
    <div className={`
      px-4 py-3 shadow-lg rounded-xl border transition-all duration-300 min-w-[180px]
      ${style.bg}
      ${selected
        ? 'border-blue-500 shadow-[0_0_20px_rgba(59,130,246,0.3)]'
        : `${style.border} ${style.hover}`
      }
      backdrop-blur-md
    `}>
      <Handle type="target" position={Position.Top} className="!bg-blue-500 !w-3 !h-3 !-top-1.5" />

      <div className="flex items-center gap-3">
        <div className={`
          p-2 rounded-lg 
          ${style.iconBg} ${style.iconColor}
        `}>
          <Icon size={18} />
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
