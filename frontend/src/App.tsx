import { useState } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import CodeEditor from './components/Editor'
import Viewer from './components/Viewer'
import DetailsPanel from './components/DetailsPanel'
import { Play } from 'lucide-react'
import { type Node } from '@xyflow/react'

const DEFAULT_CODE = `
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(20 * 24 * 24, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        return x

model = SimpleModel()
`

function App() {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [graphData, setGraphData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)

  const handleVisualize = async () => {
    setLoading(true)
    setError(null)
    setSelectedNode(null)
    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      })
      const data = await response.json()

      if (data.error) {
        setError(data.error)
      } else {
        setGraphData(data)
      }
    } catch (err) {
      setError("Failed to connect to backend. Is it running?")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="h-screen w-screen bg-[#0f0f11] text-white flex flex-col">
      {/* Header */}
      <header className="h-14 border-b border-white/10 flex items-center px-4 justify-between bg-[#1a1a1a]">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-md"></div>
          <h1 className="font-bold text-lg tracking-tight">TorchViz</h1>
        </div>
        <button
          onClick={handleVisualize}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-md text-sm font-medium transition-colors"
        >
          <Play size={16} />
          {loading ? 'Processing...' : 'Visualize'}
        </button>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden relative">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={40} minSize={20}>
            <div className="h-full flex flex-col">
              <div className="flex-1 relative">
                <CodeEditor code={code} onChange={(val) => setCode(val || '')} />
              </div>
              {error && (
                <div className="p-4 bg-red-900/20 border-t border-red-500/30 text-red-200 text-sm font-mono overflow-auto max-h-40">
                  {error}
                </div>
              )}
            </div>
          </Panel>

          <PanelResizeHandle className="w-1 bg-white/5 hover:bg-blue-500/50 transition-colors" />

          <Panel minSize={30}>
            <div className="h-full bg-[#0a0a0c] relative">
              <Viewer
                graphData={graphData}
                onNodeClick={(node) => setSelectedNode(node)}
              />
              {!graphData && !loading && (
                <div className="absolute inset-0 flex items-center justify-center text-white/30 pointer-events-none">
                  <p>Run the model to see the graph</p>
                </div>
              )}

              {/* Details Panel Overlay */}
              {selectedNode && (
                <DetailsPanel
                  node={selectedNode}
                  onClose={() => setSelectedNode(null)}
                />
              )}
            </div>
          </Panel>
        </PanelGroup>
      </div>
    </div>
  )
}

export default App
