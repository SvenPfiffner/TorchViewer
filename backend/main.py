
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.fx as fx
import io
import contextlib
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

@app.post("/analyze")
async def analyze_model(request: CodeRequest):
    try:
        # Create a safe globals dictionary
        safe_globals = {
            "torch": torch,
            "nn": nn,
            "fx": fx,
        }
        
        # Execute the code to define the model class/instance
        # We expect the code to define a variable named 'model' or a class that we can instantiate
        exec_globals = safe_globals.copy()
        exec(request.code, exec_globals)
        
        model = exec_globals.get("model")
        
        if model is None:
            # Try to find the first nn.Module subclass in the executed code
            for name, obj in exec_globals.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module:
                    try:
                        model = obj()
                        break
                    except:
                        pass
                        
        if model is None:
            return {"error": "No model found. Please define a variable 'model' or a class inheriting from nn.Module."}
            
        if not isinstance(model, nn.Module):
             return {"error": "Found 'model' variable but it is not an instance of nn.Module."}

        # Trace the model
        tracer = fx.Tracer()
        graph = tracer.trace(model)
        
        # Shape Inference
        example_input = exec_globals.get("example_input")
        if example_input is not None:
            try:
                from torch.fx.passes.shape_prop import ShapeProp
                # Wrap model in GraphModule for ShapeProp
                gm = fx.GraphModule(model, graph)
                ShapeProp(gm).propagate(example_input)
                # Graph is modified in-place with tensor_meta
            except Exception as e:
                print(f"DEBUG: Shape inference failed: {e}")

        nodes = []
        edges = []
        
        # Helper to get module from path
        def get_module_by_name(module, access_string):
            names = access_string.split(sep='.')
            return reduce(getattr, names, module)

        from functools import reduce

        for node in graph.nodes:
            metadata = {}
            
            # Extract Shape Information
            if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
                tensor_meta = node.meta['tensor_meta']
                print(f"DEBUG: Node {node.name} has tensor_meta: {tensor_meta}")
                # tensor_meta can be a TensorMetadata, or a tuple/list of them
                def extract_shape(tm):
                    if hasattr(tm, 'shape'):
                        return list(tm.shape)
                    return None
                
                if hasattr(tensor_meta, 'shape'):
                    metadata['shape'] = extract_shape(tensor_meta)
                elif isinstance(tensor_meta, (list, tuple)):
                    metadata['shape'] = [extract_shape(tm) for tm in tensor_meta]
                
                print(f"DEBUG: Extracted shape for {node.name}: {metadata.get('shape')}")
            else:
                print(f"DEBUG: Node {node.name} has NO tensor_meta")

            if node.op == 'call_module':
                try:
                    print(f"DEBUG: Getting module for target: {node.target}")
                    submod = get_module_by_name(model, node.target)
                    print(f"DEBUG: Found submod: {submod}")
                    metadata['type'] = submod.__class__.__name__
                    # Extract common parameters
                    for attr in ['in_features', 'out_features', 'kernel_size', 'stride', 'padding', 'dilation', 'groups', 'bias', 'in_channels', 'out_channels']:
                        if hasattr(submod, attr):
                            val = getattr(submod, attr)
                            # Convert tensor/tuple to string/list for JSON serialization
                            if isinstance(val, (tuple, list)):
                                metadata[attr] = list(val)
                            elif hasattr(val, 'item') and (not hasattr(val, 'numel') or val.numel() == 1): # Tensor scalar
                                metadata[attr] = val.item()
                            elif hasattr(val, 'shape'): # Tensor
                                metadata[attr] = str(list(val.shape))
                            else:
                                metadata[attr] = val
                except Exception as e:
                    print(f"DEBUG: Error getting module: {e}")
                    metadata['type'] = 'Unknown'
            
            # Format target name
            target_name = str(node.target)
            if node.op == 'call_function':
                if hasattr(node.target, '__name__'):
                    target_name = node.target.__name__
                    # Handle torch.cat specifically or others if needed
                    if target_name == 'cat':
                        target_name = 'torch.cat'
                    elif target_name == 'add':
                        target_name = 'Add'
            elif node.op == 'call_method':
                target_name = str(node.target)
            elif node.op == 'call_module':
                target_name = str(node.target)

            node_info = {
                "id": node.name,
                "op": node.op,
                "target": target_name,
                "args": [str(arg) for arg in node.args],
                "kwargs": {k: str(v) for k, v in node.kwargs.items()},
                "metadata": metadata
            }
            
            nodes.append(node_info)
            
            for input_node in node.all_input_nodes:
                edges.append({
                    "source": input_node.name,
                    "target": node.name
                })
                
        return {
            "nodes": nodes,
            "edges": edges,
            "code": str(graph) # The generated python code from the graph
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
