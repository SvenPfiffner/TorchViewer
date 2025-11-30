import { useEffect, useCallback } from 'react';
import { ReactFlow, Controls, Background, useNodesState, useEdgesState, addEdge, type Connection, type Edge, type Node, MarkerType, Position } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import dagre from 'dagre';
import CustomNode from './CustomNode';

const nodeTypes = {
    custom: CustomNode,
};

const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
    const dagreGraph = new dagre.graphlib.Graph();
    dagreGraph.setDefaultEdgeLabel(() => ({}));

    const isHorizontal = direction === 'LR';
    dagreGraph.setGraph({ rankdir: direction });

    nodes.forEach((node) => {
        dagreGraph.setNode(node.id, { width: 220, height: 100 });
    });

    edges.forEach((edge) => {
        dagreGraph.setEdge(edge.source, edge.target);
    });

    dagre.layout(dagreGraph);

    const layoutedNodes = nodes.map((node) => {
        const nodeWithPosition = dagreGraph.node(node.id);
        return {
            ...node,
            targetPosition: isHorizontal ? Position.Left : Position.Top,
            sourcePosition: isHorizontal ? Position.Right : Position.Bottom,
            // We are shifting the dagre node position (anchor=center center) to the top left
            // so it matches the React Flow node anchor point (top left).
            position: {
                x: nodeWithPosition.x - 220 / 2,
                y: nodeWithPosition.y - 100 / 2,
            },
        };
    });

    return { nodes: layoutedNodes, edges };
};

interface ViewerProps {
    graphData: any;
    onNodeClick: (node: Node) => void;
}

export default function Viewer({ graphData, onNodeClick }: ViewerProps) {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

    // Update graph when data changes
    useEffect(() => {
        if (graphData) {
            const initialNodes = graphData.nodes.map((n: any) => ({
                id: n.id,
                position: { x: 0, y: 0 }, // Layout will handle this
                data: {
                    label: n.target,
                    op: n.op,
                    metadata: n.metadata
                },
                type: 'custom'
            }));

            const initialEdges = graphData.edges.map((e: any) => ({
                id: `${e.source}-${e.target}`,
                source: e.source,
                target: e.target,
                animated: true,
                style: { stroke: '#3b82f6', strokeWidth: 2 },
                markerEnd: {
                    type: MarkerType.ArrowClosed,
                    color: '#3b82f6',
                },
            }));

            const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
                initialNodes,
                initialEdges
            );

            setNodes(layoutedNodes);
            setEdges(layoutedEdges);
        }
    }, [graphData, setNodes, setEdges]);

    const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={(_, node) => onNodeClick(node)}
                nodeTypes={nodeTypes}
                colorMode="dark"
                fitView
            >
                <Controls />
                <Background color="#cbd5e1" gap={20} size={1} />
                <Controls className="bg-white/80 backdrop-blur-md border border-white/50 shadow-lg !text-slate-600 [&>button]:!border-slate-200 [&>button:hover]:!bg-slate-100" />
            </ReactFlow>
        </div>
    );
}
