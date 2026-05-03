import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import time
import os
import matplotlib.pyplot as plt
import pandas as pd
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, pauli_error
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class QuantumNetworkProtocol:
    """Quantum Network Protocol Framework with Noise Analysis"""

    def __init__(self, num_relays, peripherals_per_relay, connection_type='linear'):
        """
        Initialize quantum network

        Parameters:
        num_relays: Number of relay nodes
        peripherals_per_relay: Number of peripheral nodes per relay (can be integer or list)
        connection_type: Connection type ('linear' )
        """
        self.num_relays = num_relays
        self.connection_type = connection_type

        # Handle peripheral node configuration
        if isinstance(peripherals_per_relay, int):
            self.peripherals_list = [peripherals_per_relay] * num_relays
        else:
            self.peripherals_list = peripherals_per_relay

        # Automatically generate connection relationships
        if connection_type == 'linear':
            self.connections = [(i, i + 1) for i in range(num_relays - 1)]
        elif connection_type == 'star':
            # First node as center node
            self.connections = [(0, i) for i in range(1, num_relays)]
        else:
            raise ValueError("Unsupported connection type. Supported: 'linear' or 'star'")

        # Create relay configuration
        self.relay_config = {}
        for i in range(num_relays):
            # Calculate connection count for each relay node
            connection_count = sum(1 for conn in self.connections if i in conn)

            # Calculate required particles = peripheral nodes + connection count
            particles_needed = self.peripherals_list[i] + connection_count

            self.relay_config[i] = {
                'particles': particles_needed,
                'peripherals': self.peripherals_list[i]
            }

        # Initialize quantum backends
        self.backend = Aer.get_backend('qasm_simulator')
        self.noisy_backend = AerSimulator()
        self.circuit = None
        self.quantum_registers = {}
        self.classical_registers = {}
        self.measured_qubits = set()
        self.phase_counter = 0
        self.detailed_log = []
        self.current_measurements = {}
        self.control_particles = {}  # Store control particles for each relay node
        self.connection_particles = {}  # Store connection particles used by each relay node
        self.star_graphs = []  # Store star subgraph information

        # Calculate connection count for each relay node
        self.connection_counts = {}
        for relay_id in self.relay_config:
            self.connection_counts[relay_id] = sum(1 for conn in self.connections if relay_id in conn)

        # Noise analysis parameters
        self.noise_results = {}
        self.error_mitigation_results = {}

        # Create output directories
        os.makedirs("circuit_diagrams", exist_ok=True)
        os.makedirs("detailed_logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("analysis_plots", exist_ok=True)
        os.makedirs("noise_analysis", exist_ok=True)

    def create_noise_model(self, single_qubit_gate_error=0.0001,
                           two_qubit_gate_error=0.002,
                           measurement_error=0.005,
                           t1=1000.0,
                           t2=0.1,
                           thermal_relaxation=True,
                           bias_eta=50.0,
                           erasure_conversion_rate=0.0):

        noise_model = NoiseModel()

        # --- 1. Define biased Pauli error generator ---
        def get_biased_1q_error(p_total, eta):
            p_x = p_total / (eta + 2)
            p_y = p_x
            p_z = p_total - (p_x + p_y)
            p_i = 1 - p_total
            return pauli_error([('X', p_x), ('Y', p_y), ('Z', p_z), ('I', p_i)])

        def get_biased_2q_error(p_total, eta):
            p_per_qubit = 1 - np.sqrt(1 - p_total)
            err_1q = get_biased_1q_error(p_per_qubit, eta)
            return err_1q.tensor(err_1q)

        # --- 2. Construct base Pauli errors ---
        if bias_eta > 1.0:
            error_1q_gate = get_biased_1q_error(single_qubit_gate_error, bias_eta)
            error_2q_gate = get_biased_2q_error(two_qubit_gate_error, bias_eta)

            # Apply Erasure Conversion:
            # Converts a fraction of bit-flip errors into heralded losses, reducing effective residual errors
            effective_meas_error = measurement_error * (1.0 - erasure_conversion_rate)

            p_flip = effective_meas_error
            error_meas = pauli_error([('X', p_flip), ('I', 1 - p_flip)])
        else:
            effective_meas_error = measurement_error * (1.0 - erasure_conversion_rate)
            error_1q_gate = depolarizing_error(single_qubit_gate_error, 1)
            error_2q_gate = depolarizing_error(two_qubit_gate_error, 2)
            error_meas = depolarizing_error(effective_meas_error, 1)

        # --- 3. Superimpose thermal relaxation (T1/T2) errors ---
        single_qubit_gates = ['h', 'x', 'y', 'z', 's', 't']
        two_qubit_gates = ['cx', 'cz']

        if thermal_relaxation:
            gate_time_1q = 10e-6
            gate_time_2q = 100e-6
            meas_time = 150e-6

            if t2 < gate_time_2q:
                print(f"Warning: T2 ({t2}) is shorter than 2-qubit gate time.")

            thermal_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
            combined_error_1q = error_1q_gate.compose(thermal_1q)

            thermal_2q_q0 = thermal_relaxation_error(t1, t2, gate_time_2q)
            thermal_2q_q1 = thermal_relaxation_error(t1, t2, gate_time_2q)
            thermal_2q = thermal_2q_q0.tensor(thermal_2q_q1)
            combined_error_2q = error_2q_gate.compose(thermal_2q)

            thermal_meas = thermal_relaxation_error(t1, t2, meas_time)
            combined_error_meas = error_meas.compose(thermal_meas)

            noise_model.add_all_qubit_quantum_error(combined_error_1q, single_qubit_gates)
            noise_model.add_all_qubit_quantum_error(combined_error_2q, two_qubit_gates)
            noise_model.add_all_qubit_quantum_error(combined_error_meas, "measure")

        else:
            noise_model.add_all_qubit_quantum_error(error_1q_gate, single_qubit_gates)
            noise_model.add_all_qubit_quantum_error(error_2q_gate, two_qubit_gates)
            noise_model.add_all_qubit_quantum_error(error_meas, "measure")

        return noise_model

    def execute_with_noise(self, circuit, shots=2000,
                           single_qubit_gate_error=0.0001,
                           two_qubit_gate_error=0.002,
                           measurement_error_rate=0.005,
                           use_error_mitigation=False,
                           thermal_relaxation=True,
                           t1=1000, t2=1,
                           bias_eta=50,
                           erasure_conversion_rate=0.0):
        """
        Execute circuit with specified noise parameters including erasure conversion
        """
        # Create noise model
        noise_model = self.create_noise_model(
            single_qubit_gate_error=single_qubit_gate_error,
            two_qubit_gate_error=two_qubit_gate_error,
            measurement_error=measurement_error_rate,
            t1=t1, t2=t2,
            thermal_relaxation=thermal_relaxation,
            bias_eta=bias_eta,
            erasure_conversion_rate=erasure_conversion_rate
        )

        # Execute with noise
        job = execute(circuit, self.noisy_backend, shots=shots, noise_model=noise_model)
        result = job.result()
        noisy_counts = result.get_counts()

        return noisy_counts, None, noise_model

    def analyze_results(self, counts, center_index=None, leaf_nodes=None, shots=1000):
        """Analyze results and calculate fidelity"""
        print("=== Analyzing final state ===")
        print("Verifying if final state is star graph state")

        # If center particle and leaf nodes not provided, use default values
        if center_index is None or leaf_nodes is None:
            if not self.star_graphs:
                print("Warning: No star graph information, using default values")
                center_index = 0
                leaf_nodes = list(range(1, self.circuit.num_qubits))
            else:
                # Use center particle of first star graph as center node
                center_index = self.star_graphs[0]['center'].index

                # Collect all leaf nodes
                leaf_nodes = []
                for graph in self.star_graphs:
                    for leaf in graph['leaves']:
                        leaf_nodes.append(leaf.index)

        print(f"Center node: Particle {center_index}")
        print(f"Leaf nodes: {leaf_nodes}")

        correct_count = 0
        total_count = 0

        # Print first 5 states for debugging
        print("\nFirst 5 measurement results:")
        state_count = 0
        for state, count in counts.items():
            if state_count < 5:
                # Remove all spaces
                state_clean = state.replace(" ", "")

                # Take first n characters (n=number of qubits)
                total_qubits = self.circuit.num_qubits
                if len(state_clean) >= total_qubits:
                    final_str = state_clean[:total_qubits]
                else:
                    final_str = state_clean

                # Reverse string to correct little-endian
                final_reversed = final_str[::-1]

                print(f"Original state: {state}")
                print(f"Processed state: {final_reversed}")

                # Extract center node value
                if center_index < len(final_reversed):
                    center_value = final_reversed[center_index]
                    print(f"Center node value: {center_value}")
                else:
                    print("Center node value: Index out of range")
                    center_value = None

                # Extract leaf node values
                leaf_values = []
                for idx in leaf_nodes:
                    if idx < len(final_reversed):
                        leaf_values.append(final_reversed[idx])
                    else:
                        leaf_values.append('?')  # Index out of range

                print(f"Leaf node values: {leaf_values}")

                # Check consistency
                if center_value is not None and leaf_values and all(v == center_value for v in leaf_values):
                    print("✅ Center node and leaf nodes consistent")
                else:
                    print("❌ Center node and leaf nodes inconsistent")

                print(f"Count: {count}\n")
                state_count += 1

        # Calculate fidelity for all states
        for state, count in counts.items():
            # Remove all spaces
            state_clean = state.replace(" ", "")

            # Take first n characters (n=number of qubits)
            total_qubits = self.circuit.num_qubits
            if len(state_clean) >= total_qubits:
                final_str = state_clean[:total_qubits]
            else:
                final_str = state_clean

            # Reverse string to correct little-endian
            final_reversed = final_str[::-1]

            # Check consistency
            valid = True
            if center_index < len(final_reversed):
                center_value = final_reversed[center_index]

                for idx in leaf_nodes:
                    if idx >= len(final_reversed) or final_reversed[idx] != center_value:
                        valid = False
                        break
            else:
                valid = False

            if valid:
                correct_count += count
            total_count += count

        if total_count == 0:
            fidelity = 0.0
        else:
            fidelity = correct_count / total_count

        print(f"Verification completed: {correct_count}/{total_count} states have consistent measurement results")
        print(f"Fidelity: {fidelity:.4f}")

        # Save results
        result_filename = f"results/result_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(result_filename, 'w', encoding='utf-8') as f:
            f.write(f"Fidelity: {fidelity:.4f}\n")
            f.write(f"Center node: particle {center_index}\n")
            f.write(f"Leaf nodes: {leaf_nodes}\n")
            f.write(f"Total shots: {shots}\n")
            f.write(f"Correct states: {correct_count}\n")
            f.write(f"Total states: {total_count}\n")

        return fidelity

    def initialize_circuit(self):
        """Initialize quantum circuit"""
        # Calculate total quantum bits
        total_qubits = 0
        for relay_id, config in self.relay_config.items():
            total_qubits += config['particles'] + config['peripherals']

        # Create quantum register
        qr = QuantumRegister(total_qubits, 'node')

        # Calculate Phase1 classical register size
        phase1_size = 0
        for relay_id, config in self.relay_config.items():
            num_particles = config['particles']
            num_connections = self.connection_counts[relay_id]

            # Particles to measure = total particles - connection count - 1 (control particle)
            phase1_size += num_particles - num_connections

        # Create classical registers
        creg_phase1 = ClassicalRegister(phase1_size, 'phase1_meas')
        creg_phase2 = ClassicalRegister(self.num_relays, 'phase2_meas')
        creg_phase3 = ClassicalRegister(len(self.connections), 'phase3_meas')
        creg_final = ClassicalRegister(total_qubits, 'final_meas')

        self.circuit = QuantumCircuit(qr, creg_phase1, creg_phase2, creg_phase3, creg_final)

        # Allocate quantum registers
        start_index = 0
        for relay_id, config in self.relay_config.items():
            num_particles = config['particles']
            num_peripherals = config['peripherals']

            self.quantum_registers[relay_id] = {
                'particles': qr[start_index:start_index + num_particles],
                'peripherals': qr[start_index + num_particles:start_index + num_particles + num_peripherals]
            }
            start_index += num_particles + num_peripherals

        self.classical_registers = {
            'phase1': creg_phase1,
            'phase2': creg_phase2,
            'phase3': creg_phase3,
            'final': creg_final
        }

        self.save_circuit_diagram("00_Initial_Circuit")
        self.log_operation("Initialized quantum circuit")
        self.log_operation(f"Total qubits: {total_qubits}")
        self.log_operation(f"Relay configuration: {self.relay_config}")
        self.log_operation(f"Connections: {self.connections}")
        self.log_operation(f"Connection counts per relay: {self.connection_counts}")
        self.log_operation(f"Phase1 classical register size: {phase1_size}")
        self.log_operation("\nQuantum register allocation:")
        for relay_id, reg in self.quantum_registers.items():
            self.log_operation(f"Relay node {relay_id}:")
            self.log_operation(f"  Particles: {[q.index for q in reg['particles']]}")
            self.log_operation(f"  Peripheral nodes: {[q.index for q in reg['peripherals']]}")

        return self.circuit

    def update_connections_info(self):
        """Update connection information to reflect current star graph state"""
        self.connections_info = []

        for conn in self.connections:
            relay1, relay2 = conn

            # Find star graphs for both relays
            graph1 = None
            graph2 = None
            for graph in self.star_graphs:
                if graph['relay_id'] == relay1:
                    graph1 = graph
                if graph['relay_id'] == relay2:
                    graph2 = graph

            if graph1 is None or graph2 is None:
                continue

            # Find possible connection particles
            connection_found = False

            # 1. Find common leaf nodes
            common_leaves = set(graph1['leaves']) & set(graph2['leaves'])
            if common_leaves:
                particle = common_leaves.pop()
                self.connections_info.append({
                    'relays': (relay1, relay2),
                    'particles': (particle, particle)
                })
                connection_found = True

            # 2. If no common leaf nodes found, find connection particles
            if not connection_found:
                for leaf1 in graph1['leaves']:
                    for leaf2 in graph2['leaves']:
                        # Check if two particles are connected via Bell pair
                        if self.is_connected(leaf1, leaf2):
                            self.connections_info.append({
                                'relays': (relay1, relay2),
                                'particles': (leaf1, leaf2)
                            })
                            connection_found = True
                            break
                    if connection_found:
                        break

            # 3. If still not found, use original connection particles
            if not connection_found:
                # Find original connection particles
                for conn_info in self.original_connections_info:
                    if set(conn_info['relays']) == {relay1, relay2}:
                        self.connections_info.append(conn_info)
                        break

    def protocol1_create_subgraphs(self):
        """Protocol 1: Create star subgraphs"""
        self.log_operation("\n=== Phase 1: Create star subgraphs ===")

        # Step 1: Create Bell pairs for peripheral nodes
        self.log_operation("\nStep 1: Create Bell pairs for peripherals")
        for relay_id, config in self.relay_config.items():
            particles = self.quantum_registers[relay_id]['particles']
            peripherals = self.quantum_registers[relay_id]['peripherals']
            num_peripherals = config['peripherals']

            self.log_operation(f"\nCreating Bell pairs for relay {relay_id} with {num_peripherals} peripherals")

            for i in range(num_peripherals):
                self.log_operation(f"  Apply H to particle {particles[i].index}")
                self.circuit.h(particles[i])

                self.log_operation(f"  Apply CX: particle {particles[i].index} -> particle {peripherals[i].index}")
                self.circuit.cx(particles[i], peripherals[i])

                # Form 2-particle linear cluster state
                self.log_operation(f"  Apply H to particle {particles[i].index} to form linear cluster state")
                self.circuit.h(particles[i])

        # Step 2: Create Bell pairs for inter-relay connections
        self.log_operation("\nStep 2: Create Bell pairs for inter-relay connections")
        connection_index = {relay_id: config['peripherals'] for relay_id, config in self.relay_config.items()}

        # Initialize connection particle dictionary
        self.connection_particles = {relay_id: [] for relay_id in self.relay_config}

        # Store connection information - ensure correct initialization
        self.connections_info = []
        self.original_connections_info = []

        for conn in self.connections:
            relay1, relay2 = conn
            particles1 = self.quantum_registers[relay1]['particles']
            particles2 = self.quantum_registers[relay2]['particles']

            idx1 = connection_index[relay1]
            idx2 = connection_index[relay2]

            particle1 = particles1[idx1]
            particle2 = particles2[idx2]

            self.log_operation(f"\nCreating connection between {relay1} and {relay2}")
            self.log_operation(f"  Using particle {particle1.index} from relay {relay1}")
            self.log_operation(f"  Using particle {particle2.index} from relay {relay2}")

            # Store connection particles
            self.connection_particles[relay1].append(particle1)
            self.connection_particles[relay2].append(particle2)

            # Store connection information
            conn_info = {
                'relays': (relay1, relay2),
                'particles': (particle1, particle2)
            }
            self.connections_info.append(conn_info)

            # Create Bell pair
            self.log_operation(f"  Apply H to particle {particle1.index}")
            self.circuit.h(particle1)

            self.log_operation(f"  Apply CX: particle {particle1.index} -> particle {particle2.index}")
            self.circuit.cx(particle1, particle2)

            self.log_operation(f"  Apply H to particle {particle1.index} to form linear cluster state")
            self.circuit.h(particle1)

            connection_index[relay1] += 1
            connection_index[relay2] += 1

        # Save original connection information
        self.original_connections_info = self.connections_info.copy()

        # Step 3: Select control particles for each relay node and create star connections
        self.log_operation("\nStep 3: Create star connections with control particles")

        # Store control particles for each relay node
        self.control_particles = {}

        for relay_id, config in self.relay_config.items():
            particles = self.quantum_registers[relay_id]['particles']
            num_particles = config['particles']
            num_peripherals = config['peripherals']

            self.log_operation(f"\nCreating star connections for relay {relay_id}")

            # Select control particle
            if relay_id < self.num_relays - 1:  # Not last node
                if self.connection_particles[relay_id]:
                    control_particle = self.connection_particles[relay_id][-1]
                    self.log_operation(f"  Control particle (last connection particle): {control_particle.index}")
                else:
                    control_particle = particles[0]
                    self.log_operation(f"  Control particle (first particle): {control_particle.index}")
            else:  # Last node
                control_particle = None
                for i in range(num_peripherals):
                    if particles[i] not in self.connection_particles[relay_id]:
                        control_particle = particles[i]
                        break

                if control_particle is None:
                    control_particle = particles[0]
                    self.log_operation(f"  Control particle (first particle): {control_particle.index}")
                else:
                    self.log_operation(f"  Control particle (non-connection particle): {control_particle.index}")

            # Store control particle
            self.control_particles[relay_id] = control_particle

            # Connect control particle with all other non-connection particles
            for i in range(len(particles)):
                if particles[i] != control_particle:
                    if particles[i] not in self.connection_particles[relay_id]:
                        self.log_operation(
                            f"  Apply CZ: particle {control_particle.index} -> particle {particles[i].index}")
                        self.circuit.cz(control_particle, particles[i])

        # Step 4: Measure non-control particles and apply conditional corrections
        self.log_operation("\nStep 4: Measure non-control particles and apply corrections")

        # Create classical register index counter
        meas_index = 0

        for relay_id, config in self.relay_config.items():
            particles = self.quantum_registers[relay_id]['particles']
            peripherals = self.quantum_registers[relay_id]['peripherals']
            control_particle = self.control_particles[relay_id]

            self.log_operation(f"\nMeasuring non-control particles for relay {relay_id}")

            for i in range(len(particles)):
                particle = particles[i]

                if particle == control_particle or particle in self.connection_particles[relay_id]:
                    continue

                if meas_index < self.classical_registers['phase1'].size:
                    creg = self.classical_registers['phase1'][meas_index]
                    self.log_operation(f"  Apply H to particle {particle.index} (prepare for X-basis measurement)")
                    self.circuit.h(particle)

                    self.log_operation(f"  Measure particle {particle.index} (classical register phase1[{meas_index}])")
                    self.circuit.measure(particle, creg)
                    self.measured_qubits.add(particle.index)

                    if i < len(peripherals):
                        peripheral = peripherals[i]

                        self.log_operation(f"  Apply conditional correction to peripheral particle {peripheral.index}:")
                        self.log_operation(f"    If measurement == 0: apply H")
                        self.log_operation(f"    If measurement == 1: apply H and Z")

                        self.circuit.h(peripheral).c_if(creg, 0)
                        self.circuit.h(peripheral).c_if(creg, 1)
                        self.circuit.z(peripheral).c_if(creg, 1)

                    meas_index += 1
                else:
                    self.log_operation(f"Warning: No more classical registers available for measurement")
                    break

        self.save_circuit_diagram("01_Star_Subgraphs_Created")

        # Collect and return star subgraph information
        self.star_graphs = []
        for relay_id in range(self.num_relays):
            peripherals = self.quantum_registers[relay_id]['peripherals']
            control_particle = self.control_particles[relay_id]

            leaf_nodes = []
            for peripheral in peripherals:
                leaf_nodes.append(peripheral.index)

            for conn in self.connections:
                if conn[0] == relay_id:
                    relay2 = conn[1]
                    for particle in self.connection_particles[relay_id]:
                        if particle == control_particle:
                            for conn_particle in self.connection_particles[relay2]:
                                if self.is_connected(particle, conn_particle):
                                    leaf_nodes.append(conn_particle.index)

            star_graph = {
                'relay_id': relay_id,
                'center': control_particle,
                'leaves': [self.get_particle_by_index(idx) for idx in leaf_nodes]
            }
            self.star_graphs.append(star_graph)

            self.log_operation(f"\nRelay {relay_id} star graph:")
            self.log_operation(f"  Center particle: {control_particle.index}")
            self.log_operation(f"  Leaf particles: {leaf_nodes}")

        self.log_operation("\nConnection information after Protocol 1:")
        for conn_info in self.connections_info:
            self.log_operation(f"Connection {conn_info['relays']}:")
            self.log_operation(f"  Particle 1: {conn_info['particles'][0].index}")
            self.log_operation(f"  Particle 2: {conn_info['particles'][1].index}")

        self.log_operation("\nOriginal connection information after Protocol 1:")
        for conn_info in self.original_connections_info:
            self.log_operation(f"Connection {conn_info['relays']}:")
            self.log_operation(f"  Particle 1: {conn_info['particles'][0].index}")
            self.log_operation(f"  Particle 2: {conn_info['particles'][1].index}")

        return self.circuit, self.star_graphs

    def get_particle_by_index(self, index):
        """Get particle object by index"""
        for relay_id in self.quantum_registers:
            particles = self.quantum_registers[relay_id]['particles']
            peripherals = self.quantum_registers[relay_id]['peripherals']

            for particle in particles:
                if particle.index == index:
                    return particle

            for peripheral in peripherals:
                if peripheral.index == index:
                    return peripheral

        return None

    def protocol2_center_migration(self, center_particle, new_center_particle):
        """
        Protocol 2: Center migration

        Parameters:
        center_particle: Current center particle
        new_center_particle: New center particle (must be one of the leaf nodes)
        """
        self.log_operation("\n=== Phase 2: Center migration ===")
        self.log_operation(
            f"Migrating center from particle {center_particle.index} to particle {new_center_particle.index}")

        # Prepare X-basis measurement
        self.log_operation(f"  Apply H to particle {center_particle.index} (prepare for X-basis measurement)")
        self.circuit.h(center_particle)

        # Measure current center particle
        self.log_operation(f"  Measure particle {center_particle.index}")

        if not hasattr(self, 'phase2_meas_index'):
            self.phase2_meas_index = 0

        if self.phase2_meas_index < self.classical_registers['phase2'].size:
            creg = self.classical_registers['phase2'][self.phase2_meas_index]
            self.circuit.measure(center_particle, creg)
            self.measured_qubits.add(center_particle.index)
            self.phase2_meas_index += 1

            # Conditional correction
            self.log_operation(f"  Apply conditional correction to new center particle {new_center_particle.index}:")
            self.log_operation(f"    If measurement == 0: apply H")
            self.log_operation(f"    If measurement == 1: apply H and Z")

            self.circuit.h(new_center_particle).c_if(creg, 0)
            self.circuit.h(new_center_particle).c_if(creg, 1)
            self.circuit.z(new_center_particle).c_if(creg, 1)
        else:
            self.log_operation("Warning: No available classical register for measurement")

        # Update star graph information
        for graph in self.star_graphs:
            if center_particle.index == graph['center'].index:
                new_leaves = []
                for leaf in graph['leaves']:
                    # Retain leaves that are neither the new nor old center
                    if leaf.index != new_center_particle.index and leaf.index != center_particle.index:
                        new_leaves.append(leaf)

                graph['center'] = new_center_particle
                graph['leaves'] = new_leaves

                self.log_operation(f"Updated star graph for relay {graph['relay_id']}:")
                self.log_operation(f"  New center: {new_center_particle.index}")
                self.log_operation(f"  Leaves: {[leaf.index for leaf in new_leaves]}")
                break

        self.save_circuit_diagram("03_Center_Migration_Completed")
        self.update_connections_info()

        return self.circuit

    def protocol3_subgraph_fusion(self, center1, particle1, center2, particles_to_correct):
        """
        Protocol 3: Subgraph fusion

        Parameters:
        center1: Center particle of first star graph
        particle1: Connection particle of first star graph
        center2: Center particle of second star graph
        particles_to_correct: List of particles in second star graph that need correction
        """
        self.log_operation("\n=== Phase 3: Subgraph fusion ===")
        self.log_operation(f"Fusing star graphs with centers {center1.index} and {center2.index}")

        # Apply CZ gate
        self.log_operation(f"Apply CZ: particle {center2.index} -> particle {particle1.index}")
        self.circuit.cz(center2, particle1)

        # Prepare X-basis measurement
        self.log_operation(f"Apply H to particle {particle1.index} (prepare for X-basis measurement)")
        self.circuit.h(particle1)

        self.log_operation(f"Measure particle {particle1.index}")
        if not hasattr(self, 'phase3_meas_index'):
            self.phase3_meas_index = 0

        if self.phase3_meas_index < self.classical_registers['phase3'].size:
            creg = self.classical_registers['phase3'][self.phase3_meas_index]
            self.circuit.measure(particle1, creg)
            self.measured_qubits.add(particle1.index)
            self.phase3_meas_index += 1

            # Conditional correction
            self.log_operation(f"Apply conditional correction to particle {center2.index}:")
            self.log_operation(f"  If measurement == 0: apply H")
            self.log_operation(f"  If measurement == 1: apply H and Z")

            self.circuit.h(center2).c_if(creg, 0)
            self.circuit.h(center2).c_if(creg, 1)
            self.circuit.z(center2).c_if(creg, 1)

            self.log_operation(f"Apply conditional correction to peripheral nodes:")
            for node in particles_to_correct:
                self.log_operation(f"  Particle {node.index}: if measurement == 1, apply Z")
                self.circuit.z(node).c_if(creg, 1)
        else:
            self.log_operation("Warning: No available classical register for measurement")

        # Update star graphs
        graph1_index = None
        graph2_index = None
        for i, graph in enumerate(self.star_graphs):
            if center1.index == graph['center'].index:
                graph1_index = i
            if center2.index == graph['center'].index:
                graph2_index = i

        if graph1_index is not None and graph2_index is not None:
            leaf_nodes1 = [leaf for leaf in self.star_graphs[graph1_index]['leaves'] if leaf.index != particle1.index]
            leaf_nodes2 = [leaf for leaf in self.star_graphs[graph2_index]['leaves'] if leaf.index != center2.index]

            leaf_nodes1.append(center2)
            leaf_nodes1.extend(leaf_nodes2)

            self.star_graphs[graph1_index]['leaves'] = leaf_nodes1

            new_relay_id = f"{self.star_graphs[graph1_index]['relay_id']}-{self.star_graphs[graph2_index]['relay_id']}"
            self.star_graphs[graph1_index]['relay_id'] = new_relay_id

            self.star_graphs.pop(graph2_index)

            self.log_operation("\nAutomatically migrating center to second star graph's center particle")
            current_center = self.star_graphs[graph1_index]['center']
            new_center = center2

            self.log_operation(f"  Current center particle: {current_center.index}")
            self.log_operation(f"  Migrating to new center particle: {new_center.index}")

            self.protocol2_center_migration(current_center, new_center)

        self.save_circuit_diagram("04_Subgraph_Fusion_Completed")
        return self.circuit

    def protocol3_fuse_star_graphs(self, relay_id1, relay_id2):
        """
        Protocol 3: Fuse two star subgraphs
        """
        self.log_operation("\n=== Phase 3: Fuse star graphs ===")
        self.log_operation(f"Fusing star graphs of relay {relay_id1} and {relay_id2}")

        graph1 = None
        graph2 = None
        for graph in self.star_graphs:
            if graph['relay_id'] == relay_id1:
                graph1 = graph
            if graph['relay_id'] == relay_id2:
                graph2 = graph

        if graph1 is None or graph2 is None:
            self.log_operation(f"Error: Cannot find star graphs for relay nodes {relay_id1} or {relay_id2}")
            return self.circuit

        center1 = graph1['center']
        center2 = graph2['center']

        particle1 = None
        particle2 = None

        def get_original_relays(relay_id):
            if isinstance(relay_id, int):
                return [relay_id]
            return [int(x) for x in relay_id.split('-')]

        orig_relays1 = get_original_relays(relay_id1)
        orig_relays2 = get_original_relays(relay_id2)

        found_conn = None
        for conn in self.connections:
            if (conn[0] in orig_relays1 and conn[1] in orig_relays2) or \
                    (conn[0] in orig_relays2 and conn[1] in orig_relays1):
                found_conn = conn
                break

        if found_conn:
            for conn_info in self.original_connections_info:
                if set(conn_info['relays']) == set(found_conn):
                    particle1, particle2 = conn_info['particles']
                    break

        if particle1 is None:
            common_leaves = set(graph1['leaves']) & set(graph2['leaves'])
            if common_leaves:
                particle = common_leaves.pop()
                particle1 = particle
                particle2 = particle

        if particle1 is None:
            if found_conn:
                relay1, relay2 = found_conn
                if relay1 in self.connection_particles and relay2 in self.connection_particles:
                    particle1 = self.connection_particles[relay1][0]
                    particle2 = self.connection_particles[relay2][0]

        if particle1 is None or particle2 is None:
            self.log_operation(
                f"Error: Cannot find connection particles between relay nodes {relay_id1} and {relay_id2}")
            return self.circuit

        if particle2.index in [p.index for p in graph1['leaves']]:
            particle1 = particle2

        particles_to_correct = graph2['leaves']

        self.protocol3_subgraph_fusion(center1, particle1, center2, particles_to_correct)

        self.log_operation("\nProtocol 3 execution completed!")
        return self.circuit

    def is_connected(self, particle1, particle2):
        """Check if two particles are connected via Bell pair"""
        for instruction in self.circuit.data:
            if instruction.operation.name == 'cx':
                qubits = [q.index for q in instruction.qubits]
                if particle1.index in qubits and particle2.index in qubits:
                    return True
        return False

    def save_circuit_diagram(self, phase_name):
        """Save circuit diagram to file"""
        self.phase_counter += 1
        filename = f"circuit_diagrams/{self.phase_counter:02d}_{phase_name.replace(' ', '_')}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"===== {phase_name} =====\n")
            f.write(f"Number of qubits: {self.circuit.num_qubits}\n")
            f.write(f"Circuit depth: {self.circuit.depth()}\n")
            f.write(f"Total quantum gates: {self.circuit.size()}\n\n")

            gate_counts = self.circuit.count_ops()
            f.write("Gate type distribution:\n")
            for gate, count in gate_counts.items():
                f.write(f"  {gate}: {count}\n")

            f.write("\nCircuit structure:\n")
            try:
                f.write(
                    str(self.circuit.draw(output='text', filename=None, vertical_compression='high', line_length=120)))
            except Exception:
                f.write("Circuit visualization failed. Using basic representation.\n")
                f.write(str(self.circuit))

    def log_operation(self, operation):
        """Log operation to detailed log"""
        self.detailed_log.append(operation)
        print(operation)

    def execute(self, shots=1000):
        """Execute complete protocol"""
        self.initialize_circuit()
        self.protocol1_create_subgraphs()

        self.log_operation("\n=== Final measurement ===")
        for i in range(self.circuit.num_qubits):
            if i not in self.measured_qubits:
                self.log_operation(f"Measure particle {i} (classical register final[{i}])")
                self.circuit.measure(i, self.classical_registers['final'][i])

        self.save_circuit_diagram("02_Final_Measurement_Added")

        job = execute(self.circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()

        log_filename = f"detailed_logs/protocol_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_filename, 'w', encoding='utf-8') as f:
            for line in self.detailed_log:
                f.write(line + "\n")

        return counts, self.circuit, self.star_graphs

    def final_measurement(self):
        """Add final measurement, using Z-basis for center nodes and X-basis for leaf nodes"""
        self.log_operation("\n=== Final measurement (Updated after center migration) ===")

        if not self.star_graphs:
            print("Warning: No star graph information, using default measurement")
            for i in range(self.circuit.num_qubits):
                if i not in self.measured_qubits:
                    self.log_operation(f"Measure particle {i} (classical register final[{i}])")
                    self.circuit.measure(i, self.classical_registers['final'][i])
            return

        center_index = self.star_graphs[0]['center'].index
        leaf_nodes = []
        for graph in self.star_graphs:
            for leaf in graph['leaves']:
                leaf_nodes.append(leaf.index)

        self.log_operation(f"Updated center node: Particle {center_index}")
        self.log_operation(f"Updated leaf nodes: {leaf_nodes}")

        for i in range(self.circuit.num_qubits):
            if i not in self.measured_qubits:
                if i == center_index:
                    self.log_operation(f"Measure center particle {i} (Z-basis) (classical register final[{i}])")
                    self.circuit.measure(i, self.classical_registers['final'][i])
                elif i in leaf_nodes:
                    self.log_operation(f"Apply H gate to leaf node {i} (switch to X-basis)")
                    self.circuit.h(i)
                    self.log_operation(f"Measure leaf node {i} (X-basis) (classical register final[{i}])")
                    self.circuit.measure(i, self.classical_registers['final'][i])
                else:
                    self.log_operation(f"Measure particle {i} (Z-basis) (classical register final[{i}])")
                    self.circuit.measure(i, self.classical_registers['final'][i])

                self.measured_qubits.add(i)

    def calculate_graph_building_depth(self):
        """Calculate exact circuit depth for graph construction phase"""
        if not self.circuit:
            return 0

        try:
            graph_building_circuit = QuantumCircuit(
                self.circuit.qregs[0],
                *self.circuit.cregs
            )

            graph_building_gates = {'cz', 'h', 'z'}
            conditional_ops = set()

            cz_start_index = -1
            for i, instruction in enumerate(self.circuit.data):
                op_name = instruction.operation.name
                if op_name == 'cz':
                    cz_start_index = i
                    break

            if cz_start_index == -1:
                return 0

            measure_blocks = []
            current_block = []

            for i, instruction in enumerate(self.circuit.data):
                if instruction.operation.name == 'measure':
                    current_block.append(i)
                elif current_block:
                    measure_blocks.append(current_block)
                    current_block = []

            if current_block:
                measure_blocks.append(current_block)

            final_measure_start = len(self.circuit.data)
            if measure_blocks:
                largest_block = max(measure_blocks, key=len)
                if len(largest_block) > self.circuit.num_qubits * 0.5:
                    final_measure_start = largest_block[0]

            graph_ops_count = 0
            for i in range(cz_start_index, final_measure_start):
                instruction = self.circuit.data[i]
                op = instruction.operation
                op_name = op.name

                is_conditional = hasattr(op, 'condition') and op.condition

                if is_conditional:
                    graph_building_circuit.append(op, instruction.qubits, instruction.clbits)
                    conditional_ops.add(op_name)
                elif op_name in graph_building_gates:
                    graph_building_circuit.append(op, instruction.qubits, instruction.clbits)
                    graph_ops_count += 1

            depth = graph_building_circuit.depth()

            min_expected_depth = self.num_relays + min(self.peripherals_list) - 1
            max_expected_depth = self.circuit.depth() * 0.8

            if depth < min_expected_depth:
                depth = self._estimate_graph_building_depth()

            return depth

        except Exception as e:
            print(f"Warning: Error calculating graph building depth: {e}")
            return self._estimate_graph_building_depth()

    def _estimate_graph_building_depth(self):
        """Estimate depth of graph building phase"""
        base_depth = 0
        cz_depth = 1
        max_measurements = 0
        for relay_id, config in self.relay_config.items():
            num_particles = config['particles']
            num_connections = self.connection_counts[relay_id]
            measurements = num_particles - num_connections - 1
            max_measurements = max(max_measurements, measurements)

        measure_depth = max_measurements * 2

        fusion_depth = len(self.connections) * 3
        migration_depth = 3
        total_estimated_depth = cz_depth + measure_depth + fusion_depth + migration_depth
        adjusted_depth = total_estimated_depth * 0.6

        return max(1, int(adjusted_depth))

    def get_network_statistics(self):
        if not self.circuit:
            return {}

        total_bell_states = sum(self.peripherals_list) + len(self.connections)

        final_particle_count = 0
        if self.star_graphs:
            final_graph = self.star_graphs[0]
            final_particle_count = 1 + len(final_graph['leaves'])

        total_gates = self.circuit.count_ops()
        graph_pauli_x_measurements = self.circuit.num_qubits - final_particle_count
        try:
            graph_building_depth = self.calculate_graph_building_depth()
        except Exception as e:
            print(f"Warning: Failed to calculate graph building depth: {e}")
            graph_building_depth = 0

        stats = {
            'num_relays': self.num_relays,
            'peripherals_per_relay': self.peripherals_list[0] if isinstance(self.peripherals_list[0], int) else 'mixed',
            'total_qubits': self.circuit.num_qubits,
            'bell_states_count': total_bell_states,
            'final_particle_count': final_particle_count,
            'connection_type': self.connection_type,
            'final_center': self.star_graphs[0]['center'].index if self.star_graphs else None,
            'final_leaves_count': len(self.star_graphs[0]['leaves']) if self.star_graphs else 0,

            'total_gates': sum(total_gates.values()),
            'circuit_depth': self.circuit.depth(),
            'h_gates': total_gates.get('h', 0),
            'cx_gates': total_gates.get('cx', 0),
            'cz_gates': total_gates.get('cz', 0),
            'z_gates': total_gates.get('z', 0),
            'measure_gates': total_gates.get('measure', 0),

            'graph_building_depth': graph_building_depth,
            'graph_cz_gates': total_gates.get('cz', 0),
            'graph_pauli_x_measurements': graph_pauli_x_measurements,
            'graph_conditional_z': total_gates.get('z', 0),
            'graph_h_gates': total_bell_states + graph_pauli_x_measurements,
            'total_graph_gates': 0
        }

        stats['total_graph_gates'] = (
                stats['graph_cz_gates'] +
                stats['graph_pauli_x_measurements'] +
                stats['graph_conditional_z'] +
                stats['graph_h_gates']
        )

        return stats


def run_parameter_sweep_comparison_full():
    """
    [Section 3.3] Full Parameter Sweep Comparison:
    Loops over Relays, Peripherals, AND Noise Bias (eta).
    """
    print("=" * 60)
    print("Full Quantum Network Sweep (Relays x Peripherals x Eta)")
    print("=" * 60)

    # Physical Constants
    T2_COHERENCE = 1
    TIME_PER_HOP_CLASSICAL = 600e-6

    num_relays_range = range(3, 13)
    peripherals_range = [3, 4, 5, 6]
    shots = 2000
    noise_level = 0.0001

    scenarios = [
        {'eta': 1, 'label': 'Isotropic (η=1)', 'color': '#1f77b4'},
        {'eta': 50, 'label': 'Biased (η=50)', 'color': '#d62728'}
    ]

    results = []
    total_iters = len(scenarios) * len(num_relays_range) * len(peripherals_range)
    curr_iter = 0

    for scen in scenarios:
        eta = scen['eta']
        print(f"\n--- Scenario: {scen['label']} ---")

        for num_relays in num_relays_range:
            for peripherals in peripherals_range:
                curr_iter += 1
                print(f"\rProgress {curr_iter}/{total_iters}: {num_relays}R x {peripherals}P (eta={eta})...", end="")

                try:
                    network = QuantumNetworkProtocol(num_relays, peripherals, 'linear')
                    network.initialize_circuit()
                    network.protocol1_create_subgraphs()

                    curr_gid = 0
                    for next_rid in range(1, num_relays):
                        network.protocol3_fuse_star_graphs(curr_gid, next_rid)
                        curr_gid = f"{curr_gid}-{next_rid}"

                    if network.star_graphs:
                        fg = network.star_graphs[0]
                        if fg['leaves']:
                            network.protocol2_center_migration(fg['center'], fg['leaves'][0])

                    network.final_measurement()

                    if not network.star_graphs: continue

                    final_graph = network.star_graphs[0]
                    center_idx = final_graph['center'].index
                    leaf_nodes = [l.index for l in final_graph['leaves']]

                    counts_noisy, _, _ = network.execute_with_noise(
                        network.circuit, shots=shots,
                        single_qubit_gate_error=noise_level,
                        two_qubit_gate_error=noise_level * 20,
                        measurement_error_rate=noise_level * 50,
                        use_error_mitigation=False,
                        thermal_relaxation=True,
                        bias_eta=eta
                    )

                    fidelity_circuit = network.analyze_results(counts_noisy, center_idx, leaf_nodes, shots)

                    t_classical = (num_relays - 1) * TIME_PER_HOP_CLASSICAL
                    decoherence_factor = np.exp(-t_classical / T2_COHERENCE)
                    fidelity_final = fidelity_circuit * decoherence_factor

                    stats_entry = {
                        'num_relays': num_relays,
                        'peripherals': peripherals,
                        'fidelity': fidelity_final,
                        'network_size': num_relays * peripherals,
                        'eta': eta,
                        'label': scen['label'],
                        'color': scen['color']
                    }
                    results.append(stats_entry)

                except Exception as e:
                    pass

    print("\nSweep completed.")

    if results:
        plot_full_comparison(results, noise_level)

    return results, noise_level


def plot_full_comparison(results, noise_level):
    """
    Complex plotting:
    (a) Fidelity vs Relay Count.
    (b) Scaling Regression with formatted formula F = A - B * m * n.
    """
    if not results: return

    df = pd.DataFrame(results)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    colors = {3: '#1f77b4', 4: '#2ca02c', 5: '#ff7f0e', 6: '#9467bd'}
    markers = {3: 'o', 4: 's', 5: '^', 6: 'D'}
    linestyles = {1: '-', 50: '--'}
    epsilon_1Q = noise_level * 100
    epsilon_2Q = noise_level * 20 * 100
    epsilon_m = noise_level * 50 * 100

    for p in sorted(df['peripherals'].unique()):
        subset = df[(df['eta'] == 1) & (df['peripherals'] == p)].sort_values('num_relays')
        if not subset.empty:
            ax1.plot(subset['num_relays'], subset['fidelity'],
                     color=colors[p], linestyle=linestyles[1], marker=markers[p],
                     linewidth=2, markersize=7, label='_nolegend_')

    for p in sorted(df['peripherals'].unique()):
        subset = df[(df['eta'] == 50) & (df['peripherals'] == p)].sort_values('num_relays')
        if not subset.empty:
            ax1.plot(subset['num_relays'], subset['fidelity'],
                     color=colors[p], linestyle=linestyles[50], marker=markers[p],
                     linewidth=2, markersize=7, alpha=0.9, label='_nolegend_')

    all_relays = sorted(df['num_relays'].unique())
    ax1.set_xticks(all_relays)

    ax1.set_xlabel('Number of Relay Nodes (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Operational Fidelity', fontsize=12, fontweight='bold')
    ax1.set_title(
        f'(a) Protocol Performance vs Network Scale\n'
        f'(Solid: $\\eta=1$ (Isotropic), Dashed: $\\eta=50$ (Biased))\n'
        f'Error Rates: '
        f'$\\epsilon_{{1Q}}={epsilon_1Q:.2f}\\%$, '
        f'$\\epsilon_{{2Q}}={epsilon_2Q:.2f}\\%$, '
        f'$\\epsilon_m={epsilon_m:.2f}\\%$',
        fontsize=13,
        fontweight='bold',
        pad=12
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.05)

    legend_lines = [
        Line2D([0], [0], color='black', linestyle='-', lw=2, label='η=1 (Isotropic)'),
        Line2D([0], [0], color='black', linestyle='--', lw=2, label='η=50 (Biased)')
    ]
    leg1 = ax1.legend(handles=legend_lines, loc='upper right', title="Noise Type", fontsize=10, frameon=True)
    ax1.add_artist(leg1)

    legend_points = []
    for p in sorted(df['peripherals'].unique()):
        h = Line2D([0], [0], color=colors[p], marker=markers[p], linestyle='None',
                   markersize=8, label=f'{p} Peripherals')
        legend_points.append(h)

    ax1.legend(handles=legend_points, loc='lower left', title="Peripherals", fontsize=10, frameon=True)

    for eta in [1, 50]:
        subset = df[df['eta'] == eta]
        if len(subset) < 2: continue

        color = '#1f77b4' if eta == 1 else '#d62728'
        linestyle = '-' if eta == 1 else '--'

        X = subset['network_size'].values
        y = subset['fidelity'].values

        slope, intercept, r_value, _, _ = stats.linregress(X, y)
        r_squared = r_value ** 2

        ax2.scatter(X, y, color=color, alpha=0.4, s=30, zorder=5)

        x_fit = np.linspace(X.min(), X.max(), 100)
        y_fit = slope * x_fit + intercept

        slope_abs = abs(slope)
        formula_str = f"$F = {intercept:.3f} - {slope_abs:.4f} \\times m \\times n$"
        label_text = f"$\\eta={eta}$: {formula_str}\n$R^2 = {r_squared:.3f}$"
        ax2.plot(x_fit, y_fit, color=color, linewidth=2.5, linestyle=linestyle,
                 label=label_text, zorder=10)

    ax2.set_xlabel('Total Network Size ($m \\times n$)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Operational Fidelity $F$', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Linear Scaling of Fidelity\nwith Total Network Size', fontsize=14, fontweight='bold')

    ax2.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.95, loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.05)

    plt.tight_layout()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_filename = f"analysis_plots/combined_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_erasure_extension_analysis():
    """
    [Key Simulation for Paper]
    Demonstrates network diameter extension via Erasure Conversion.
    Compares Standard Readout vs. Erasure-Converted Readout.
    """
    print("=" * 60)
    print("Erasure Conversion Impact Analysis")
    print("=" * 60)

    T2_COHERENCE = 1.0
    TIME_PER_HOP = 600e-6

    relay_counts = range(4, 25, 2)
    peripherals = 4
    shots = 2000

    base_meas_error = 0.005

    scenarios = [
        {
            'label': 'Standard Readout (0.5% Error)',
            'erasure_rate': 0.0,
            'color': 'gray',
            'style': '--'
        },
        {
            'label': 'w/ Erasure Conversion (Conv=90%)',
            'erasure_rate': 0.90,
            'color': '#d62728',
            'style': '-'
        }
    ]

    results = []

    for m in relay_counts:
        t_wait = (m - 1) * TIME_PER_HOP
        decoherence_decay = np.exp(-t_wait / T2_COHERENCE)

        network = QuantumNetworkProtocol(m, peripherals, 'linear')
        network.initialize_circuit()
        network.protocol1_create_subgraphs()
        curr_gid = 0
        for next_rid in range(1, m):
            network.protocol3_fuse_star_graphs(curr_gid, next_rid)
            curr_gid = f"{curr_gid}-{next_rid}"
        if network.star_graphs:
            fg = network.star_graphs[0]
            if fg['leaves']:
                network.protocol2_center_migration(fg['center'], fg['leaves'][0])
        network.final_measurement()

        if not network.star_graphs: continue

        final_graph = network.star_graphs[0]
        center_idx = final_graph['center'].index
        leaf_idxs = [l.index for l in final_graph['leaves']]

        for scen in scenarios:
            print(f"\rSimulating: Relays={m}, Scenario={scen['label']}...", end="")

            counts, _, _ = network.execute_with_noise(
                network.circuit,
                shots=shots,
                single_qubit_gate_error=0.0001,
                two_qubit_gate_error=0.002,
                measurement_error_rate=base_meas_error,
                thermal_relaxation=True,
                bias_eta=50,
                erasure_conversion_rate=scen['erasure_rate']
            )

            fid_circuit = network.analyze_results(counts, center_idx, leaf_idxs, shots)
            fid_total = fid_circuit * decoherence_decay

            results.append({
                'num_relays': m,
                'fidelity': fid_total,
                'scenario': scen['label'],
                'color': scen['color'],
                'style': scen['style']
            })

    print("\nErasure analysis completed.")
    plot_erasure_extension(results)
    return results


def plot_erasure_extension(results):
    """
    Plots the extension of network diameter enabled by erasure conversion.
    (Upgraded visualization for PRA publication)
    """
    if not results: return
    df = pd.DataFrame(results)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6.5))

    scenarios = df['scenario'].unique()

    xlims = (df['num_relays'].min() - 1, df['num_relays'].max() + 1)
    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=2.5, label='Entanglement Limit (F=0.5)')

    ax.fill_between(xlims, 0.4, 0.5, color='gray', alpha=0.1)
    ax.text(xlims[0] + 0.5, 0.45, "Loss of Genuine Multipartite Entanglement ($F < 0.5$)",
            color='dimgray', fontsize=12, style='italic', fontweight='bold')

    for label in scenarios:
        subset = df[df['scenario'] == label].sort_values('num_relays')
        style = subset['style'].iloc[0]
        color = subset['color'].iloc[0]

        ax.plot(subset['num_relays'], subset['fidelity'],
                marker='o', linestyle=style, color=color, linewidth=2.5, markersize=8,
                label=label, zorder=4)

    for label in scenarios:
        subset = df[df['scenario'] == label].sort_values('num_relays')
        x = subset['num_relays'].values
        y = subset['fidelity'].values
        color = subset['color'].iloc[0]

        for i in range(len(y) - 1):
            if y[i] >= 0.5 > y[i + 1]:
                m_max = int(x[i])
                f_max = y[i]

                ax.annotate(
                    f'Max Scale\n$m_{{max}} = {m_max}$',
                    xy=(m_max, f_max),
                    xytext=(m_max + 1.5, f_max + 0.08),
                    arrowprops=dict(facecolor=color, shrink=0.05, width=1.5, headwidth=8, alpha=0.8),
                    fontsize=11, color=color, fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=color, alpha=0.9),
                    zorder=5
                )
                break

    red_subset = df[df['scenario'].str.contains('Erasure')].sort_values('num_relays')
    if not red_subset.empty and red_subset['fidelity'].min() > 0.5:
        max_m = red_subset['num_relays'].max()
        min_f = red_subset['fidelity'].min()

        ax.annotate(
            f'Remains Operable\n($F={min_f:.2f}$ at $m={max_m}$)',
            xy=(max_m, min_f),
            xytext=(max_m - 3.5, min_f - 0.08),
            arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1.5, headwidth=8, alpha=0.8),
            fontsize=11, color='#d62728', fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d62728', alpha=0.9),
            zorder=5
        )

    ax.set_xlim(xlims)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel('Network Diameter (Number of Relays $m$)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Operational Fidelity', fontsize=13, fontweight='bold')
    ax.set_title('Network Scalability Enhancement via Erasure Conversion', fontsize=15, fontweight='bold')

    ax.legend(fontsize=11, frameon=True, framealpha=0.95, loc='upper right')

    plt.tight_layout()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plt.savefig(f"analysis_plots/erasure_extension_upgraded_{timestamp}.png", dpi=300)
    plt.show()


def run_resource_efficiency_analysis():
    """
    [Section 3.1 & 3.2] Resource Efficiency & Rigorous Temporal Feasibility Analysis
    (Single Trial Mode - Fixes KeyError)
    """
    print("=" * 60)
    print("Quantum Network Resource Analysis (Single Trial, Data Integrity Mode)")
    print("=" * 60)

    HW_PARAMS = {
        'T_1q': 10e-6, 'T_2q': 100e-6, 'T_meas': 150e-6, 'T_proc': 50e-6,
        'L_link': 50.0, 'L_access': 10.0, 'c_fiber': 2e5, 'T_coherence': 1.0
    }

    num_relays_range = range(3, 13)
    peripherals_range = [3, 4, 5, 6]

    results = []
    total_configs = len(num_relays_range) * len(peripherals_range)
    current_config = 0

    for num_relays in num_relays_range:
        for peripherals in peripherals_range:
            current_config += 1
            print(f"\rProgress: {current_config}/{total_configs} | Network: {num_relays}x{peripherals}", end="")

            try:
                network = QuantumNetworkProtocol(num_relays, peripherals, 'linear')
                network.initialize_circuit()
                network.protocol1_create_subgraphs()

                curr_gid = 0
                for next_rid in range(1, num_relays):
                    network.protocol3_fuse_star_graphs(curr_gid, next_rid)
                    curr_gid = f"{curr_gid}-{next_rid}"

                if network.star_graphs:
                    fg = network.star_graphs[0]
                    if fg['leaves']:
                        network.protocol2_center_migration(fg['center'], fg['leaves'][0])

                stats = network.get_network_statistics()

                particles = stats.get('final_particle_count', 0)
                gates = stats.get('total_graph_gates', 0)
                bell_pairs = stats.get('bell_states_count', 0)

                if particles > 0:
                    stats['ecpq'] = bell_pairs / particles
                    stats['gate_density'] = gates / particles
                    stats['efficiency_ratio'] = particles / gates
                else:
                    stats['ecpq'] = 0
                    stats['gate_density'] = 0
                    stats['efficiency_ratio'] = 0

                t_1q, t_2q, t_meas = HW_PARAMS['T_1q'], HW_PARAMS['T_2q'], HW_PARAMS['T_meas']
                t_proc = HW_PARAMS['T_proc']
                t_lat_backbone = (HW_PARAMS['L_link'] / HW_PARAMS['c_fiber']) + t_proc
                t_lat_access = (HW_PARAMS['L_access'] / HW_PARAMS['c_fiber']) + t_proc

                t_phase1 = (t_2q + t_meas + t_1q) + t_lat_access
                t_step_fusion = t_2q + t_meas + t_lat_backbone + t_1q
                t_step_migration = t_meas + t_lat_backbone + t_1q
                t_phase2_serial = (num_relays - 1) * (t_step_fusion + t_step_migration)
                t_phase3 = t_meas + t_lat_access + t_1q

                stats['time_total_s'] = t_phase1 + t_phase2_serial + t_phase3

                t_quantum_pure = num_relays * (t_2q + t_meas + t_1q) + (num_relays - 1) * (
                        t_2q + 2 * t_meas + 2 * t_1q) + (t_meas + t_1q)
                stats['time_quantum_s'] = t_quantum_pure
                stats['time_classical_s'] = stats['time_total_s'] - t_quantum_pure

                stats['params'] = HW_PARAMS

                results.append(stats)

            except Exception as e:
                pass

    print("\nSimulations completed.")
    return results, HW_PARAMS


def plot_top_tier_resource_analysis(results):
    if not results: return None
    df = pd.DataFrame(results)

    if not os.path.exists('analysis_plots'):
        os.makedirs('analysis_plots')

    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    peripherals = sorted(df['peripherals_per_relay'].unique())
    colors = sns.color_palette("viridis", len(peripherals))

    for i, n in enumerate(peripherals):
        subset = df[df['peripherals_per_relay'] == n].sort_values('num_relays')
        ax1.plot(subset['num_relays'], subset['ecpq'], 'o-',
                 color=colors[i], markersize=6, alpha=0.9, label=f'$n={n}$')

    ax1.set_xlabel('Network Scale (Number of Relay Nodes $m$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Entanglement Cost per Qubit', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Asymptotic Convergence of Resource Cost', fontsize=13)
    ax1.legend(title='Peripherals')
    ax1.grid(True, linestyle=':', alpha=0.6)

    def precise_density_model(X, c_base, c_overhead, c_correction):
        n, m = X
        return c_base + (c_overhead / n) - (c_correction / (m * n))

    n_data = df['peripherals_per_relay'].values
    m_data = df['num_relays'].values
    y_data = df['gate_density'].values

    try:
        popt, _ = curve_fit(precise_density_model, (n_data, m_data), y_data,
                            bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]))
        c_base, c_overhead, c_correction = popt

        fit_msg = f"Fit: $\\rho \\approx {c_base:.1f} + {c_overhead:.1f}/n - {c_correction:.1f}/mn$"
        fit_success = True
    except:
        fit_success = False
        fit_msg = "Model Fit Failed"

    for i, n in enumerate(peripherals):
        subset = df[df['peripherals_per_relay'] == n].sort_values('num_relays')
        m_vals = subset['num_relays']

        ax2.plot(m_vals, subset['gate_density'], 's-',
                 color=colors[i], markersize=6, alpha=0.9, label=f'$n={n}$')

        if fit_success:
            m_smooth = np.linspace(m_vals.min(), m_vals.max(), 50)
            y_theory = precise_density_model((n, m_smooth), c_base, c_overhead, c_correction)
            ax2.plot(m_smooth, y_theory, '--', color=colors[i], linewidth=1.5, alpha=0.6)

    ax2.set_xlabel('Network Scale (Number of Relay Nodes $m$)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Gate Density (Ops / Final Qubit)', fontsize=12, fontweight='bold')

    ax2.set_title(f'(b) Amortized Complexity ({fit_msg})', fontsize=12)
    ax2.legend(title='Peripherals')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_plots/top_tier_resource_{timestamp}_corrected.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename


def plot_temporal_feasibility(results, hw_params):
    """
    [Section 3.2] Temporal Feasibility Analysis (Log Scale)
    Shows breakdown of Quantum vs Classical time against Coherence Limit.
    Updated: Adds explicit percentage annotation for Classical Dominance.
    """
    if not results: return None
    df = pd.DataFrame(results)
    if 'peripherals_per_relay' in df.columns:
        target_n = 6 if 6 in df['peripherals_per_relay'].values else df['peripherals_per_relay'].unique()[0]
        df = df[df['peripherals_per_relay'] == target_n].sort_values('num_relays')

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 7))

    x = df['num_relays']
    t_total = df['time_total_s']
    t_classical = df['time_classical_s']
    t2_limit = hw_params['T_coherence']

    ratios = (t_classical / t_total) * 100
    avg_ratio = ratios.mean()

    ax.plot(x, t_total, 'D-', color='navy', linewidth=2.5, markersize=8, label='Total Execution Time ($T_{exec}$)')
    ax.plot(x, t_classical, 'o--', color='forestgreen', linewidth=2, alpha=0.8, label='Classical Comm. & Processing')

    ax.axhline(y=t2_limit, color='crimson', linestyle='-', linewidth=2.5)
    ax.text(x.min(), t2_limit * 1.15, f"  Ion Trap Coherence Limit ($T_2 = {t2_limit}s$)",
            color='crimson', fontweight='bold', va='bottom', fontsize=11)

    ax.fill_between(x, 1e-6, t2_limit, color='limegreen', alpha=0.1, label='Feasible Operation Region')
    mid_idx = len(x) // 2
    mid_x = x.iloc[mid_idx]
    mid_y = t_classical.iloc[mid_idx]

    ax.annotate(
        f"Classical Dominance: {avg_ratio:.1f}%\n(Fiber Latency Limited)",
        xy=(mid_x, mid_y),
        xytext=(mid_x, mid_y * 0.15),
        arrowprops=dict(facecolor='forestgreen', shrink=0.05, alpha=0.6, width=1.5, headwidth=8),
        fontsize=11, color='darkgreen', fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="forestgreen", alpha=0.9)
    )

    ax.set_yscale('log')
    ax.set_xlabel('Network Scale (Number of Relay Nodes $m$)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Temporal Feasibility: Protocol Latency vs. Hardware Physics', fontsize=16, fontweight='bold')

    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.9, loc='lower left')

    max_time = t_total.max()
    margin = t2_limit / max_time
    ax.text(x.max(), t2_limit * 0.5, f"Safety Margin $\\approx {margin:.0f}\\times$  ",
            color='navy', fontweight='bold', va='top', ha='right', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="navy", alpha=0.8))

    plt.tight_layout()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_plots/temporal_feasibility_rigorous_{timestamp}.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    return filename


def run_phase_transition_analysis():
    """
    [Section 3.4] Phase Transition Diagram with T2 Correction
    """
    print("=" * 60)
    print("Phase Transition Analysis (Physically Corrected)")
    print("=" * 60)

    T2_COHERENCE = 1.0
    TIME_CLASSICAL_LATENCY = 600e-6

    relay_counts = list(range(3, 13))
    fixed_peripherals = 4
    meas_error_levels = np.logspace(-3, -1, 50)
    shots = 2000

    results = []
    total_steps = len(relay_counts) * len(meas_error_levels)
    step = 0

    for m in relay_counts:
        t_dead_wait = (m - 1) * TIME_CLASSICAL_LATENCY
        decoherence_factor = np.exp(-t_dead_wait / T2_COHERENCE)

        network_template = QuantumNetworkProtocol(m, fixed_peripherals, 'linear')
        network_template.initialize_circuit()
        network_template.protocol1_create_subgraphs()
        curr_gid = 0
        for next_rid in range(1, m):
            network_template.protocol3_fuse_star_graphs(curr_gid, next_rid)
            curr_gid = f"{curr_gid}-{next_rid}"
        if network_template.star_graphs:
            fg = network_template.star_graphs[0]
            if fg['leaves']:
                network_template.protocol2_center_migration(fg['center'], fg['leaves'][0])
        network_template.final_measurement()

        if not network_template.star_graphs: continue

        final_graph = network_template.star_graphs[0]
        center_idx = final_graph['center'].index
        leaf_idxs = [l.index for l in final_graph['leaves']]
        circuit = network_template.circuit

        for err_meas in meas_error_levels:
            step += 1
            print(f"\rScanning: M={m}, Err={err_meas:.4f} ({step}/{total_steps})", end="")

            try:
                counts, _, _ = network_template.execute_with_noise(
                    circuit, shots=shots,
                    single_qubit_gate_error=0.0001,
                    two_qubit_gate_error=0.002,
                    measurement_error_rate=err_meas,
                    use_error_mitigation=False,
                    thermal_relaxation=True,
                    bias_eta=50.0
                )

                fid_circuit = network_template.analyze_results(counts, center_idx, leaf_idxs, shots)
                fid_final = fid_circuit * decoherence_factor

                results.append({
                    'num_relays': m,
                    'noise_level': err_meas,
                    'fidelity': fid_final
                })
            except Exception:
                pass

    print("\nPhase transition scan completed.")
    return results


def plot_phase_transition_diagram(results):
    if not results: return None
    df = pd.DataFrame(results)

    pivot_table = df.pivot_table(values='fidelity', index='noise_level', columns='num_relays')

    X = pivot_table.columns.values
    Y = pivot_table.index.values
    Z = pivot_table.values
    X_grid, Y_grid = np.meshgrid(X, Y)

    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(10, 8))

    cp = ax.contourf(X_grid, Y_grid, Z, levels=20, cmap='RdYlBu', alpha=0.9, vmin=0, vmax=1)
    cbar = fig.colorbar(cp, label='Operational Fidelity')
    ax.set_yscale('log')
    levels = [0.5, 0.8]
    cs = ax.contour(X_grid, Y_grid, Z, levels=levels, colors=['red', 'black'], linewidths=2.5, linestyles=['-', '--'])
    ax.clabel(cs, inline=True, fontsize=12, fmt='F=%.1f', colors='blue')
    current_tech_limit = 0.005
    ax.axhline(y=current_tech_limit, color='lime', linestyle=':', linewidth=3, alpha=0.8)
    if Y.max() > 0:
        y_text_pos = current_tech_limit * 1.5 if current_tech_limit * 1.5 < Y.max() else current_tech_limit * 0.9
    else:
        y_text_pos = current_tech_limit * 1.5

    ax.text(X[0] + 0.1, y_text_pos,
            "Current Ion Trap Readout $\\sim 0.5\\%$",
            color='lime', fontweight='bold', fontsize=10, ha='left')

    ax.set_xlabel('Network Scale (Number of Relay Nodes)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Measurement Readout Error Rate (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('Operational Phase Diagram: Scalability vs. Dominant Noise Source', fontsize=14, fontweight='bold')

    from matplotlib.ticker import LogLocator, FuncFormatter
    formatter = FuncFormatter(lambda y, _: '{:.2%}'.format(y))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

    plt.tight_layout()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"analysis_plots/phase_transition_log_{timestamp}_meas_dominant.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    return filename


def save_results_to_excel(results, filename_prefix="resource_efficiency"):
    if not results:
        print("No results to save.")
        return None
    df = pd.DataFrame(results)
    df['conversion_efficiency'] = df['final_particle_count'] / df['bell_states_count']
    df['gates_per_particle'] = df['total_graph_gates'] / df['final_particle_count']
    df['cz_gate_ratio'] = df['graph_cz_gates'] / df['total_graph_gates']
    df['other_gates'] = df['total_graph_gates'] - df['graph_cz_gates']

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    excel_filename = f"results/{filename_prefix}_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All Configurations', index=False)
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Configurations',
                'Average Efficiency (Particles/Gate)',
                'Average CZ Gate Ratio',
                'Average Circuit Depth',
                'Best Efficiency Configuration',
                'Worst Efficiency Configuration',
                'Most Resource-Efficient Configuration'
            ],
            'Value': [
                len(df),
                df['efficiency_ratio'].mean(),
                df['cz_gate_ratio'].mean(),
                df['graph_building_depth'].mean(),
                f"{df.loc[df['efficiency_ratio'].idxmax(), 'num_relays']}×{df.loc[df['efficiency_ratio'].idxmax(), 'peripherals_per_relay']}",
                f"{df.loc[df['efficiency_ratio'].idxmin(), 'num_relays']}×{df.loc[df['efficiency_ratio'].idxmin(), 'peripherals_per_relay']}",
                f"{df.loc[df['gates_per_particle'].idxmin(), 'num_relays']}×{df.loc[df['gates_per_particle'].idxmin(), 'peripherals_per_relay']}"
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)

        relay_summary = df.groupby('num_relays').agg({
            'total_graph_gates': ['mean', 'std', 'min', 'max'],
            'efficiency_ratio': ['mean', 'std', 'min', 'max'],
            'graph_building_depth': ['mean', 'std', 'min', 'max'],
            'final_particle_count': ['mean', 'std', 'min', 'max']
        }).round(3)
        relay_summary.to_excel(writer, sheet_name='By Relay Count')

        peripheral_summary = df.groupby('peripherals_per_relay').agg({
            'total_graph_gates': ['mean', 'std', 'min', 'max'],
            'efficiency_ratio': ['mean', 'std', 'min', 'max'],
            'graph_building_depth': ['mean', 'std', 'min', 'max'],
            'final_particle_count': ['mean', 'std', 'min', 'max']
        }).round(3)
        peripheral_summary.to_excel(writer, sheet_name='By Peripheral Count')

        target_particle_count = 12
        target_configs = df[df['final_particle_count'] == target_particle_count]
        if not target_configs.empty:
            target_configs_sorted = target_configs.sort_values('efficiency_ratio', ascending=False)
            target_configs_sorted.to_excel(writer, sheet_name=f'Target {target_particle_count} Particles', index=False)

        resource_breakdown = df[
            ['num_relays', 'peripherals_per_relay', 'graph_cz_gates', 'other_gates', 'total_graph_gates']].copy()
        resource_breakdown['cz_percentage'] = (
                resource_breakdown['graph_cz_gates'] / resource_breakdown['total_graph_gates'] * 100).round(1)
        resource_breakdown['other_percentage'] = (
                resource_breakdown['other_gates'] / resource_breakdown['total_graph_gates'] * 100).round(1)
        resource_breakdown.to_excel(writer, sheet_name='Resource Breakdown', index=False)

    print(f"Results saved to Excel file: {excel_filename}")
    return excel_filename


if __name__ == "__main__":
    # Choose simulation type
    print("Choose simulation type:")
    print("1. Standard Execution (Verify the correctness of the protocol)")
    print("2. Resource Efficiency & Temporal Analysis (Fig. 4 & Fig. 5)")
    print("3. Parameter Sweep Comparison (Fig. 6)")
    print("4. Phase Transition Diagram (Fig. 7)")
    print("5. Erasure Extension Analysis (Fig. 8)")

    choice = input("Enter choice (1-5): ").strip()

    if choice == "1":
        # Single simulation with comprehensive noise analysis
        # Original simulation (no noise analysis)
        print("=" * 60)
        print("Quantum Network Protocol Simulation (Original)")
        print("=" * 60)
        # Configuration parameters
        num_relays = 5
        peripherals_per_relay = 5
        connection_type = 'linear'

        # Create quantum network
        network = QuantumNetworkProtocol(
            num_relays=num_relays,
            peripherals_per_relay=peripherals_per_relay,
            connection_type=connection_type
        )

        # Initialize circuit
        network.initialize_circuit()

        # Execute Protocol 1
        circuit, star_graphs = network.protocol1_create_subgraphs()

        print("\nProtocol 1 execution completed!")
        print(f"Circuit depth: {circuit.depth()}")
        print(f"Total quantum gates: {circuit.size()}")

        # Print star subgraph information
        print("\nStar subgraph information:")
        for graph in network.star_graphs:
            print(f"Relay node {graph['relay_id']}:")
            print(f"  Center particle: {graph['center'].index}")
            print(f"  Leaf nodes: {[p.index for p in graph['leaves']]}")

        # Execute Protocol 3: Sequentially fuse all star graphs
        print(f"\n=== Executing Protocol 3: Sequentially fusing {num_relays} star subgraphs ===")

        # Dynamically fuse all star graphs
        current_graph_id = 0  # Start from first star graph

        for next_relay_id in range(1, num_relays):
            print(f"\n--- Fusing {next_relay_id + 1}th star graph (Relay node {next_relay_id}) ---")

            # Execute fusion
            network.protocol3_fuse_star_graphs(current_graph_id, next_relay_id)

            # Update current graph identifier
            current_graph_id = f"{current_graph_id}-{next_relay_id}"

            # Print fused star graph information
            print(f"\nFused star graph information ({next_relay_id + 1} nodes fused):")
            for graph in network.star_graphs:
                print(f"Relay node {graph['relay_id']}:")
                print(f"  Center particle: {graph['center'].index}")
                print(f"  Leaf nodes: {[p.index for p in graph['leaves']]}")

        print(f"\n=== Executing Protocol 2: Center Migration ===")
        if network.star_graphs:
            final_graph = network.star_graphs[0]
            current_center = final_graph['center']
            leaves = final_graph['leaves']

            if leaves:
                # Select the first leaf node as the new center
                new_center = leaves[0]
                print(f"Migrating center from particle {current_center.index} to particle {new_center.index}")

                # Execution Center Migration
                network.protocol2_center_migration(current_center, new_center)

                # Update star chart information
                final_graph = network.star_graphs[0]
                print(f"New center particle: {final_graph['center'].index}")
                print(f"Leaf nodes after migration: {[p.index for p in final_graph['leaves']]}")
            else:
                print("Warning: No leaf nodes available for center migration")
        else:
            print("Error: No star graph available for center migration")

        # Add final measurement
        network.final_measurement()
        # Execute simulation
        shots = 1000
        job = execute(network.circuit, network.backend, shots=shots)
        result = job.result()
        counts = result.get_counts()

        # Verify final star graph
        print("\n=== Verifying final star graph ===")
        if network.star_graphs:
            final_graph = network.star_graphs[0]
            center_index = final_graph['center'].index
            leaf_nodes = [leaf.index for leaf in final_graph['leaves']]

            print(f"Verifying final star graph (all {num_relays} relay nodes fused):")
            print(f"  Center particle: {center_index}")
            print(f"  Leaf nodes: {leaf_nodes}")
            print(f"  Number of leaf nodes: {len(leaf_nodes)}")

            # Analyze results
            fidelity = network.analyze_results(
                counts=counts,
                center_index=center_index,
                leaf_nodes=leaf_nodes,
                shots=shots
            )
            print(f"Final fidelity: {fidelity:.4f}")
        else:
            print("Error: No star graph information available")

        # Save detailed log
        log_filename = f"detailed_logs/protocol_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(f"Quantum Network Protocol Simulation Results\n")
            f.write(f"Number of relay nodes: {num_relays}\n")
            f.write(f"Peripheral nodes per relay: {peripherals_per_relay}\n")
            f.write(f"Connection type: {connection_type}\n")
            f.write(f"Total qubits: {network.circuit.num_qubits}\n")
            f.write(f"Circuit depth: {network.circuit.depth()}\n")
            f.write(f"Total quantum gates: {network.circuit.size()}\n")
            f.write(f"Simulation shots: {shots}\n")
            if network.star_graphs:
                f.write(f"Final fidelity: {fidelity:.4f}\n")
            f.write("\nDetailed operation log:\n")
            for line in network.detailed_log:
                f.write(line + "\n")

        print(f"\nProtocol verification completed!")
        print(f"Number of relay nodes: {num_relays}")
        print(f"Peripheral nodes per relay: {peripherals_per_relay}")
        print(f"Total qubits: {network.circuit.num_qubits}")
        print(f"Circuit depth: {network.circuit.depth()}")
        print(f"Total quantum gates: {network.circuit.size()}")
        if network.star_graphs:
            print(f"Final fidelity: {fidelity:.4f}")
        print(f"Log saved: {log_filename}")


    elif choice == "2":
        results, hw_params = run_resource_efficiency_analysis()

        if results:
            save_results_to_excel(results, "top_tier_resource_data")
            print("\nGenerating Resource Plots (Fig.4 in paper)...")
            plot_top_tier_resource_analysis(results)
            print("\nGenerating Temporal Plots (Fig.5 in paper)...")
            plot_temporal_feasibility(results, hw_params)

    elif choice == "3":
        print("=" * 60)
        print("Quantum Network Parameter Sweep - Enhanced Analysis")
        print("=" * 60)
        results, noise_level = run_parameter_sweep_comparison_full()

    elif choice == "4":
        results = run_phase_transition_analysis()
        if results:
            df = pd.DataFrame(results)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            df.to_csv(f"results/phase_transition_data_{timestamp}.csv", index=False)
            print("\nGenerating Phase Transition Diagram (Section 3.4)...")
            plot_phase_transition_diagram(results)

    elif choice == "5":
        run_erasure_extension_analysis()

    else:
        print("Invalid choice. Please enter 1, 2, 3,4 or 5.")
