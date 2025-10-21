# Micrograd Pro

**Micrograd Pro** is an enhanced, educational re-implementation of [Karpathy's micrograd](https://github.com/karpathy/micrograd), designed for teaching, experimentation, and research. It expands the original with extra modules, modern activations, research paper references, visualizations and mathematical explanations throughout interactive notebooks.

## Features

- **Scalar Engine**  
  Core `Value` class for automatic differentiation (grads).

- **Extra Activations**  
  Includes GELU, LeakyReLU, Tanh, ReLU, and more.

- **Simple Neural Network Modules**  
  - `Layer`, `Linear`, `Sequential`
  - `LayerNorm`, `Dropout`

- **Notebooks**  
  - **`pro.ipynb`**: Main content here, step-by-step construction of Micrograd Pro core, with explanations and references to research papers.
  - **`flow.ipynb`**: Visualizes gradient flow, computation graphs, loss surfaces, and includes a small Iris classification example.

- **Mathematical Explanations**  
  Inline math and derivations in notebooks with simplifications.

- **Research Paper References**  
  Techniques and modules are linked to relevant papers for deeper study.


## Example Visualizations

![Dropout Viz](https://raw.githubusercontent.com/0lekz/micrograd_pro/main/pics/flow/output6.png)
![GELU Example](https://raw.githubusercontent.com/0lekz/micrograd_pro/main/pics/flow/output11.png)
![Comp Graph Example](https://raw.githubusercontent.com/0lekz/micrograd_pro/main/pics/flow/output1.png)
 
## Project Structure

```
micrograd_pro/
│
├── core/                # Core engine and neural network modules
│   ├── engine.py        # Value class
│   ├── nn.py            # Layers, activations, etc.
│
├── building_pro/        # Notebooks and demos
│   ├── pro.ipynb        # Step-by-step build & explanations
│   ├── flow.ipynb       # Gradient flow, visualizations, Iris demo
│
├── pics/                # Example images from notebooks
│   ├── flow/            # Images from flow notebook
|   |   └── ...
│   ├── gradient_flow_example.png
│   ├── loss_surface_3d.png
│   ├── iris_decision_boundary.png
|   └── ...
|
├── scratch              # original micrograd notebooks
│   ├── micrograd_scratch.ipynb
│   └── micrograd_scratch_clean.ipynb
│
├── tasks/               # More external materials, neetcode problems, handwritten notes
│
├── README.md
├── resources.md.        # List of most references
└── ...
```


## Getting Started

1. **Clone the repo**
    ```bash
    git clone https://github.com/0lekz/micrograd_pro.git
    cd micrograd_pro
    ```

2. **Install dependencies**
    - Most notebooks use: `numpy`, `matplotlib`, `seaborn`, `networkx`, `scipy`, `sklearn`
    - For animations: `matplotlib.animation`, `IPython.display`
    - For advanced visuals: `graphviz` (optional)

3. **Run the notebooks**
    - Open `building_pro/pro.ipynb` for a step-by-step build and explanations.
    - Open `building_pro/flow.ipynb` for visualizations and practical demos.

## Notebooks Overview

- **`pro.ipynb`**  
  - Builds the autodiff engine from scratch.
  - Explains each step with math and references.
  - Shows how to extend with new activations and layers.

- **`flow.ipynb`**  
  - Visualizes forward and backward passes.
  - Shows gradient flow through computation graphs.
  - Plots loss surfaces and gradient descent trajectories.
  - Includes a small Iris classification demo.

## Tasks Folder

The `tasks/` folder will contain:
- Useful external materials (e.g., neetcode problems and solutions)
- Handwritten notes
- Additional learning resources

## References

- [micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)
- [Layer Normalization (Ba et al., 2016)](https://arxiv.org/abs/1607.06450)
- [GELU Activation (Hendrycks & Gimpel, 2016)](https://arxiv.org/abs/1606.08415)
- [Dropout (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)

more references in: `resources.md`

## License

MIT License

## Contact

For questions or suggestions, open an issue or reach out via GitHub.

**Micrograd Pro** is a learning tool — experiment, hack, and enjoy exploring neural networks from the inside out!
