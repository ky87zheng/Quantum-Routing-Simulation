# Reproducibility Package: Relay–Terminal Star-Graph-State Generation

This repository contains the Qiskit simulation code and numerical analysis used for the manuscript:

**Scalable Star-Graph-State Generation with Local Feedforward in Relay–Terminal Quantum Networks**

The main script is:

```text
protocol-simulation.py
```

It generates the numerical results for Figs. 4–7 of the manuscript.

## Repository contents

Recommended layout:

```text
.
├── README.md
├── requirements.txt
├── protocol-simulation.py
├── LICENSE
├── .gitignore
├── figure4.png
├── figure5.png
├── figure6.png
├── figure7.png
└── results/
    └── *.csv
```

The script automatically creates the `results/`, `analysis_plots/`,
`circuit_diagrams/`, and `detailed_logs/` directories when needed.

## Software requirements

Python 3.9 or later is recommended.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The code uses NumPy, pandas, Matplotlib, Qiskit, and Qiskit Aer.

## Main numerical parameters

The default settings are:

```text
SHOTS = 3000
BASE_SEED = 12664

T_1Q    = 10 us
T_2Q    = 100 us
T_meas  = 150 us
T_proc  = 50 us

L_link   = 50 km
L_access = 10 km
c_fiber  = 2e5 km/s

single-qubit gate error = 1e-4
two-qubit gate error    = 2e-3
measurement error       = 5e-3
```

The link-generation and memory sweeps are:

```text
T_gen = 0, 1, 5, 10 ms
T2    = 1, 10, 100, 1000 ms
```

Elementary Bell pairs are treated as ideal input resources in the simulator.
The final basis transformations and measurements used to evaluate the
two-setting star-state witness are treated as ideal diagnostic operations.

## Running the calculations

Run:

```bash
python protocol-simulation.py
```

The program provides the following options:

```text
1. Ideal protocol check
2. Resource utilization (Fig. 4)
3. Timing and link preparation (Fig. 5)
4. Noise analysis (Fig. 6)
5. Memory-coherence sensitivity (Fig. 7)
6. Run Figs. 4-7
7. Model summary
```

Option 6 reproduces all four numerical figures.

## Figure summary

- **Fig. 4** — Bell-link and local two-qubit-gate utilization.
- **Fig. 5** — post-link serial timing and synchronized elementary-link preparation.
- **Fig. 6** — two-setting star-state fidelity lower bound under circuit noise and link waiting.
- **Fig. 7** — finite-memory sensitivity for different values of \(T_2\).

For a network with `m` relay nodes and `n` terminal nodes per relay,

```text
N = mn
S = m(n + 1)
N_Bell = S - 1
N_2Q   = S - 2
```

With the representative timing parameters used in the manuscript,

```text
T_quantum   = 0.42 m + 0.10 n - 0.10 ms
T_classical = 0.60 m - 0.40 ms
T_serial    = 1.02 m + 0.10 n - 0.50 ms
```

The synchronized link-preparation baseline uses independent exponential
link-generation times with mean `T_gen`. The memory analysis uses

```text
Q = N [T_serial + T_gen (H_Slink - 1)]
```

and the multipartite-coherence factor `exp(-Q/T2)`.

## Output files

The script saves the generated figures as:

```text
figure4.png
figure5.png
figure6.png
figure7.png
```

Numerical data are written as timestamped CSV files in `results/`.

## License

This repository is released under the MIT License.
