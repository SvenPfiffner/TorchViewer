import { useState, useCallback } from 'react';
import { ReactFlowProvider, type Node } from '@xyflow/react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { Code2, Play, AlertCircle, Loader2, Sparkles } from 'lucide-react';
import Editor from './components/Editor';
import Viewer from './components/Viewer';
import DetailsPanel from './components/DetailsPanel';

function App() {
  const [code, setCode] = useState<string>(`import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(20 * 24 * 24, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return self.linear(x.view(x.size(0), -1))

model = SimpleModel()`);

  const [graphData, setGraphData] = useState<{ nodes: any[], edges: any[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);

  const handleVisualize = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to analyze code');
      }

      setGraphData(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const onNodeClick = useCallback((_event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden text-slate-800 font-sans">
      {/* Header */}
      <header className="h-16 border-b border-white/20 bg-white/60 backdrop-blur-xl flex items-center justify-between px-6 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg shadow-blue-500/20">
            <Sparkles className="text-white" size={20} />
          </div>
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-800 to-slate-600">
            TorchViz
          </h1>
        </div>

        <button
          onClick={handleVisualize}
          disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 bg-white hover:bg-slate-50 text-slate-700 rounded-xl font-semibold transition-all shadow-lg shadow-slate-200/50 border border-white/50 disabled:opacity-50 disabled:cursor-not-allowed active:scale-95"
        >
          {loading ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} className="fill-current" />}
          <span>Visualize</span>
        </button>
      </header>

      {/* Main Content */}
      <div className="flex-1 relative">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={40} minSize={30} className="flex flex-col bg-white/40 backdrop-blur-md border-r border-white/20">
            <div className="flex-1 relative p-4">
              <div className="absolute inset-0 bg-white/50 backdrop-blur-sm rounded-2xl border border-white/40 shadow-inner overflow-hidden m-4">
                <Editor code={code} onChange={(val) => setCode(val || '')} />
              </div>
            </div>
            {error && (
              <div className="m-4 p-4 bg-rose-50/90 border border-rose-100 text-rose-600 rounded-xl flex items-start gap-3 shadow-lg shadow-rose-500/5 backdrop-blur-md">
                <AlertCircle size={20} className="shrink-0 mt-0.5" />
                <pre className="text-xs font-mono whitespace-pre-wrap">{error}</pre>
              </div>
            )}
          </Panel>

          <PanelResizeHandle className="w-1.5 bg-white/20 hover:bg-blue-500/20 transition-colors" />

          <Panel className="relative bg-slate-50/30">
            <ReactFlowProvider>
              <Viewer
                graphData={graphData}
                onNodeClick={onNodeClick}
                onPaneClick={onPaneClick}
              />
            </ReactFlowProvider>

            <DetailsPanel
              selectedNode={selectedNode}
              onClose={() => setSelectedNode(null)}
            />
          </Panel>
        </PanelGroup>
      </div>
    </div>
  )
}

export default App
