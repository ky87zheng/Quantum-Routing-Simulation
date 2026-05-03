# Scalable Graph State Generation in Quantum Networks

This repository contains the numerical simulation source code supporting the findings in the manuscript **"Scalable Graph State Generation with $O(1)$ Local Feedforward in Quantum Networks"**. 

The code maps our proposed $O(1)$ local feedforward routing protocol onto a dual-species trapped-ion platform, evaluating resource scalability, temporal feasibility, noise tolerance, and erasure conversion performance.

## Prerequisites

This project is built on Python and relies on Qiskit for quantum simulation. 
To ensure reproducibility, we recommend using a virtual environment (e.g., `venv` or `conda`).

**Note on Python Version:** This project is compatible with Python 3.9 - 3.11. 
*If you are using Python 3.12+, please note that older versions of Qiskit (e.g., 0.44.x) may have compatibility issues. Using a Python 3.9-3.11 environment is highly recommended.*

To install the required dependencies, please run:

`pip install -r requirements.txt`

## How to Run

Execute the main script from your terminal:

`python protocol-simulation.py`

Upon execution, an interactive prompt will appear, allowing you to select which simulation module to run. These options correspond directly to the results and figures presented in the manuscript:

### Options Menu:

*   **[1] Standard Execution (Verify the correctness of the protocol)**
    Executes a noise-free simulation to verify the logical correctness of the graph state generation, fusion, and center migration processes. Generates detailed circuit diagrams and operation logs.
*   **[2] Resource Efficiency & Temporal Analysis (Fig. 4 & Fig. 5)**
    Simulates the topological scaling of the network. Generates the asymptotic resource cost convergence plot (Fig. 4a), the amortized $O(1)$ complexity gate density plot (Fig. 4b), and the rigorous temporal feasibility plot showing the safety margin against the coherence limit (Fig. 5). *Outputs raw data to an Excel file.*
*   **[3] Parameter Sweep Comparison (Fig. 6)**
    Evaluates the protocol's error accumulation under different noise biases (eta=1 isotropic vs. eta=50 Z-biased) to demonstrate linear scalability in realistic hardware environments. Generates Fig. 6.
*   **[4] Phase Transition Diagram (Fig. 7)**
    Performs a dense parameter sweep across varying network scales and readout error rates to identify the boundary of genuine multipartite entanglement. Generates the logarithmic contour map shown in Fig. 7.
*   **[5] Erasure Extension Analysis (Fig. 8)**
    Simulates the application of heralded erasure conversion. Compares standard readout against a 90% conversion efficiency model, demonstrating significant improvements in the operational network diameter. Generates Fig. 8.

## Output Structure

Running the simulations will automatically create the following subdirectories in your working folder to store the outputs:
*   `/analysis_plots/`: High-resolution figures (.png) matching the manuscript.
*   `/results/`: Raw numerical data (.xlsx or .csv) containing comprehensive statistics for all simulated configurations.
*   `/detailed_logs/`: Step-by-step classical classical tracking of operations and topological modifications.
*   `/circuit_diagrams/`: Text-based representations of the generated quantum circuits.

## Note on Phenomenological Noise Modeling
For large-scale network simulations (e.g., m=24 nodes), explicitly tracking the classical control flow of dynamic retries for erasure conversion at the density-matrix level is computationally prohibitive. Thus, in Option [5], we employ a rigorous phenomenological model where heralded retries effectively filter out errors at the cost of additional resources, effectively suppressing the measurement error to e_eff = e_m * (1 - eta_conv), as detailed in Section III.E of the manuscript.
