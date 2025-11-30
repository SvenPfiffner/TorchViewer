# TorchViewer - Interactive PyTorch Model Visualizer

![TorchViewer Demo](assets/demo.png)

**TorchViewer** is a small, interactive web application designed to help visualize PyTorch models. By combining a React Flow frontend with a `torch.fx` backend, TorchViewer transforms code into beautiful, explorable graphs in real-time.

## ✨ Key Features

*   **Live Code Editing**: Write or paste your PyTorch model code directly in the browser.
*   **Interactive Visualization**: Explore your model's architecture with a zoomable, pannable graph.
*   **Automatic Shape Inference**: Just define an `example_input`, and TorchViewer will calculate and display the output dimensions for every layer (e.g., `Output [1, 64, 128, 128]`).
*   **Detailed Metadata**: Click on any node to reveal a floating details panel with comprehensive layer information (kernel size, stride, padding, etc.).
*   **Visual Differentiation**: Layers are instantly recognizable through distinct color coding and iconography:
    *   🟢 **Convolution**: Emerald Green
    *   🔵 **Linear**: Blue
    *   🔴 **Activation**: Rose Red
    *   🟡 **Normalization**: Amber
    *   ⚪ **Pooling**: Cyan

## 🚀 Installation

### Prerequisites
*   **Python 3.8+**
*   **Node.js 16+**

### Quick Start

1.  **Clone the repository**
    ```bash
    git clone https://github.com/SvenPfiffner/TorchViewer.git
    cd TorchViewer
    ```

2.  **Run the startup script**
    This script sets up the Python virtual environment, installs dependencies, and launches both the backend and frontend servers.
    ```bash
    ./start.sh
    ```

3.  **Open in Browser**
    Navigate to `http://localhost:5173` to start visualizing!

## 📝 Code Requirements

To ensure TorchViewer can correctly visualize your model, your code must meet the following criteria:

1.  **`model` Variable**: You must instantiate your model and assign it to a variable named `model`.
    *   *Required for*: Graph extraction.
2.  **`example_input` Variable**: You must create a dummy input tensor and assign it to a variable named `example_input`.
    *   *Required for*: Shape inference (displaying input/output dimensions).

### Example Template
```python
import torch
import torch.nn as nn

# 1. Define your model class
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))

# 2. Instantiate the model as 'model'
model = MyModel()

# 3. Define example input for shape inference
example_input = torch.randn(1, 1, 28, 28)
```

## 📖 Usage

1.  **Paste your Code**: Copy the template above or your own code into the editor.
2.  **Visualize**: Click the **Visualize** button.
3.  **Explore**:
    *   Scroll to zoom, drag to pan.
    *   Click on nodes to view detailed parameters.
    *   Observe the data flow and tensor shapes at each step.

## 🛠️ Tech Stack

*   **Frontend**: React, Vite, TypeScript, TailwindCSS, React Flow, Lucide Icons.
*   **Backend**: FastAPI, PyTorch (`torch.fx`).

