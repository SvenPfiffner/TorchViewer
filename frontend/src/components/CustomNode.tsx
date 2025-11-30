import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Box, Layers, ArrowRight, Grid3x3, Share2, Activity, Minimize2, Sliders } from 'lucide-react';

const getLayerStyle = (type: string) => {
  const t = type.toLowerCase();
  if (t.includes('conv')) return {
    bg: 'bg-emerald-50/80',
    border: 'border-emerald-200',
    hover: 'hover:border-emerald-300',
    iconBg: 'bg-emerald-100',
    iconColor: 'text-emerald-600',
    icon: Grid3x3,
    shadow: 'shadow-emerald-500/10'
  };
  if (t.includes('linear')) return {
    bg: 'bg-blue-50/80',
    border: 'border-blue-200',
    hover: 'hover:border-blue-300',
    iconBg: 'bg-blue-100',
    iconColor: 'text-blue-600',
    icon: Share2,
    shadow: 'shadow-blue-500/10'
  };
  if (t.includes('relu') || t.includes('sigmoid') || t.includes('tanh')) return {
    bg: 'bg-rose-50/80',
    border: 'border-rose-200',
    hover: 'hover:border-rose-300',
    iconBg: 'bg-rose-100',
    iconColor: 'text-rose-600',
    icon: Activity,
    shadow: 'shadow-rose-500/10'
  };
  if (t.includes('pool')) return {
    bg: 'bg-cyan-50/80',
    border: 'border-cyan-200',
    hover: 'hover:border-cyan-300',
    iconBg: 'bg-cyan-100',
    iconColor: 'text-cyan-600',
    icon: Minimize2,
    shadow: 'shadow-cyan-500/10'
  };
  if (t.includes('norm')) return {
    bg: 'bg-amber-50/80',
    border: 'border-amber-200',
    hover: 'hover:border-amber-300',
    iconBg: 'bg-amber-100',
    iconColor: 'text-amber-600',
    icon: Sliders,
    shadow: 'shadow-amber-500/10'
  };

  // Default for other layers
  return {
    bg: 'bg-white/80',
    border: 'border-slate-200',
    hover: 'hover:border-slate-300',
    iconBg: 'bg-slate-100',
    iconColor: 'text-slate-600',
    icon: Layers,
    shadow: 'shadow-slate-500/10'
  };
};

export default function CustomNode({ data, selected }: NodeProps) {
  const metadata = data.metadata as any || {};
  const isLayer = data.op === 'call_module';
  const label = data.label as string;
  const type = metadata.type || 'Op';

  const style = isLayer ? getLayerStyle(type) : {
    bg: 'bg-white/90',
    border: 'border-slate-200',
    hover: 'hover:border-slate-300',
    iconBg: 'bg-slate-100',
    iconColor: 'text-slate-500',
    icon: Box,
    shadow: 'shadow-slate-500/5'
  };

  const Icon = style.icon;

  return (
    <div className={`
      px-5 py-4 shadow-xl rounded-2xl border transition-all duration-300 min-w-[200px]
      ${style.bg}
      ${selected
        ? 'border-blue-400 ring-4 ring-blue-400/10 shadow-blue-500/20'
        : `${style.border} ${style.hover} ${style.shadow}`
      }
      backdrop-blur-xl
    `}>
      <Handle type="target" position={Position.Top} className="!bg-slate-400 !w-3 !h-3 !-top-1.5 !border-2 !border-white" />

      <div className="flex items-center gap-4">
        <div className={`
          p-2.5 rounded-xl shadow-sm
          ${style.iconBg} ${style.iconColor}
        `}>
          <Icon size={20} strokeWidth={2} />
        </div>

        <div className="flex flex-col">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-0.5">
            {isLayer ? type : 'Operation'}
          </span>
          <span className="text-base font-bold text-slate-800">
            {isLayer ? label.split(':')[1]?.trim() : label}
          </span>
        </div>
      </div>

      {/* Quick stats for Conv/Linear */}
      {isLayer && (metadata.in_features || metadata.in_channels) && (
        <div className="mt-4 pt-3 border-t border-slate-200/60 flex items-center gap-2 text-xs font-medium text-slate-500">
          <span className="bg-white/50 px-1.5 py-0.5 rounded-md border border-slate-100">
            {metadata.in_features || metadata.in_channels}
          </span>
          <ArrowRight size={12} className="text-slate-300" />
          <span className="bg-white/50 px-1.5 py-0.5 rounded-md border border-slate-100">
            {metadata.out_features || metadata.out_channels}
          </span>
        </div>
      )}

      {/* Shape Info */}
      {metadata.shape && (
        <div className="mt-2 flex items-center justify-center gap-1.5">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Output</span>
          <span className="text-[10px] font-mono text-slate-500 bg-slate-50 px-2 py-0.5 rounded-md border border-slate-200 shadow-sm">
            {String(metadata.shape).replace(/,/g, ', ')}
          </span>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-slate-400 !w-3 !h-3 !-bottom-1.5 !border-2 !border-white" />
    </div>
  );
}
