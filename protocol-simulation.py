# Simulation script for the numerical results in Figs. 4-7.

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import math
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, pauli_error


SHOTS = 3000
BASE_SEED = 12664
VERBOSE = False
SAVE_CIRCUITS = False

HW_PARAMS = {
    "T_1q": 10e-6,
    "T_2q": 100e-6,
    "T_meas": 150e-6,
    "T_proc": 50e-6,
    "L_link": 50.0,
    "L_access": 10.0,
    "c_fiber": 2e5,
    "T2_ref": 1.0,
}

NOISE_PARAMS = {
    "single_qubit_gate_error": 1e-4,
    "two_qubit_gate_error": 2e-3,
    "measurement_error_rate": 5e-3,
}

TGEN_SWEEP = [0.0, 1e-3, 5e-3, 10e-3]
T2_SWEEP = [1e-3, 10e-3, 100e-3, 1.0]

for _d in ("results", "analysis_plots", "circuit_diagrams", "detailed_logs"):
    os.makedirs(_d, exist_ok=True)


def qidx(q):
    if hasattr(q, "index"):
        return q.index
    return q._index


class QuantumNetworkProtocol:
    def __init__(self, num_relays, peripherals_per_relay, connection_type="linear"):
        self.num_relays = num_relays
        self.connection_type = connection_type

        if isinstance(peripherals_per_relay, int):
            self.peripherals_list = [peripherals_per_relay] * num_relays
        else:
            self.peripherals_list = list(peripherals_per_relay)

        if connection_type == "linear":
            self.connections = [(i, i + 1) for i in range(num_relays - 1)]
        elif connection_type == "star":
            self.connections = [(0, i) for i in range(1, num_relays)]
        else:
            raise ValueError("Unsupported connection type. Supported: 'linear' or 'star'")

        self.relay_config = {}
        for i in range(num_relays):
            connection_count = sum(1 for conn in self.connections if i in conn)
            self.relay_config[i] = {
                "particles": self.peripherals_list[i] + connection_count,
                "peripherals": self.peripherals_list[i],
            }

        self.connection_counts = {
            relay_id: sum(1 for conn in self.connections if relay_id in conn)
            for relay_id in self.relay_config
        }

        self.backend = Aer.get_backend("qasm_simulator")
        self.noisy_backend = AerSimulator(method="stabilizer")

        self.circuit = None
        self.quantum_registers = {}
        self.classical_registers = {}
        self.measured_qubits = set()

        self.phase_counter = 0
        self.detailed_log = []

        self.control_particles = {}
        self.connection_particles = {}
        self.connections_info = []
        self.original_connections_info = []
        self.star_graphs = []

        self.phase2_meas_index = 0
        self.phase3_meas_index = 0

    def log_operation(self, text):
        self.detailed_log.append(text)
        if VERBOSE:
            print(text)

    def save_circuit_diagram(self, phase_name):
        if not SAVE_CIRCUITS or self.circuit is None:
            return
        self.phase_counter += 1
        filename = (
            f"circuit_diagrams/{self.phase_counter:02d}_"
            f"{phase_name.replace(' ', '_')}.txt"
        )
        with open(filename, "w", encoding="utf-8") as f:
            f.write(str(self.circuit.draw(output="text", fold=120)))

    def get_particle_by_index(self, index):
        for relay_id in self.quantum_registers:
            for q in self.quantum_registers[relay_id]["particles"]:
                if qidx(q) == index:
                    return q
            for q in self.quantum_registers[relay_id]["peripherals"]:
                if qidx(q) == index:
                    return q
        return None

    def is_connected(self, particle1, particle2):
        i1, i2 = qidx(particle1), qidx(particle2)
        for instruction in self.circuit.data:
            if instruction.operation.name == "cx":
                ids = [qidx(q) for q in instruction.qubits]
                if i1 in ids and i2 in ids:
                    return True
        return False

    def initialize_circuit(self):
        total_qubits = sum(
            config["particles"] + config["peripherals"]
            for config in self.relay_config.values()
        )
        qr = QuantumRegister(total_qubits, "node")

        phase1_size = 0
        for relay_id, config in self.relay_config.items():
            phase1_size += config["particles"] - self.connection_counts[relay_id]

        creg_phase1 = ClassicalRegister(phase1_size, "phase1_meas")
        creg_phase2 = ClassicalRegister(self.num_relays, "phase2_meas")
        creg_phase3 = ClassicalRegister(len(self.connections), "phase3_meas")
        creg_final = ClassicalRegister(total_qubits, "final_meas")

        self.circuit = QuantumCircuit(
            qr, creg_phase1, creg_phase2, creg_phase3, creg_final
        )

        start_index = 0
        for relay_id, config in self.relay_config.items():
            npart = config["particles"]
            nper = config["peripherals"]
            self.quantum_registers[relay_id] = {
                "particles": qr[start_index:start_index + npart],
                "peripherals": qr[start_index + npart:start_index + npart + nper],
            }
            start_index += npart + nper

        self.classical_registers = {
            "phase1": creg_phase1,
            "phase2": creg_phase2,
            "phase3": creg_phase3,
            "final": creg_final,
        }
        return self.circuit

    def protocol1_create_subgraphs(self):
        self.log_operation("=== Protocol 1: create local star subgraphs ===")

        for relay_id, config in self.relay_config.items():
            particles = self.quantum_registers[relay_id]["particles"]
            peripherals = self.quantum_registers[relay_id]["peripherals"]
            for i in range(config["peripherals"]):
                self.circuit.h(particles[i])
                self.circuit.cx(particles[i], peripherals[i])
                self.circuit.h(particles[i])

        connection_index = {
            relay_id: config["peripherals"]
            for relay_id, config in self.relay_config.items()
        }

        self.connection_particles = {
            relay_id: [] for relay_id in self.relay_config
        }
        self.connections_info = []
        self.original_connections_info = []

        for relay1, relay2 in self.connections:
            particles1 = self.quantum_registers[relay1]["particles"]
            particles2 = self.quantum_registers[relay2]["particles"]

            particle1 = particles1[connection_index[relay1]]
            particle2 = particles2[connection_index[relay2]]

            self.connection_particles[relay1].append(particle1)
            self.connection_particles[relay2].append(particle2)

            self.connections_info.append({
                "relays": (relay1, relay2),
                "particles": (particle1, particle2),
            })

            self.circuit.h(particle1)
            self.circuit.cx(particle1, particle2)
            self.circuit.h(particle1)

            connection_index[relay1] += 1
            connection_index[relay2] += 1

        self.original_connections_info = self.connections_info.copy()

        self.control_particles = {}

        for relay_id, config in self.relay_config.items():
            particles = self.quantum_registers[relay_id]["particles"]
            nper = config["peripherals"]

            if relay_id < self.num_relays - 1:
                control_particle = (
                    self.connection_particles[relay_id][-1]
                    if self.connection_particles[relay_id]
                    else particles[0]
                )
            else:
                control_particle = None
                for i in range(nper):
                    if particles[i] not in self.connection_particles[relay_id]:
                        control_particle = particles[i]
                        break
                if control_particle is None:
                    control_particle = particles[0]

            self.control_particles[relay_id] = control_particle

            for particle in particles:
                if (
                    particle != control_particle
                    and particle not in self.connection_particles[relay_id]
                ):
                    self.circuit.cz(control_particle, particle)

        meas_index = 0

        for relay_id, config in self.relay_config.items():
            particles = self.quantum_registers[relay_id]["particles"]
            peripherals = self.quantum_registers[relay_id]["peripherals"]
            control_particle = self.control_particles[relay_id]

            for i, particle in enumerate(particles):
                if (
                    particle == control_particle
                    or particle in self.connection_particles[relay_id]
                ):
                    continue

                creg = self.classical_registers["phase1"][meas_index]
                self.circuit.h(particle)
                self.circuit.measure(particle, creg)
                self.measured_qubits.add(qidx(particle))

                if i < len(peripherals):
                    peripheral = peripherals[i]
                    self.circuit.h(peripheral)
                    self.circuit.z(peripheral).c_if(creg, 1)

                meas_index += 1

        self.star_graphs = []
        for relay_id in range(self.num_relays):
            peripherals = self.quantum_registers[relay_id]["peripherals"]
            control_particle = self.control_particles[relay_id]
            leaf_nodes = list(peripherals)

            for conn in self.connections:
                if conn[0] == relay_id:
                    relay2 = conn[1]
                    for particle in self.connection_particles[relay_id]:
                        if particle == control_particle:
                            for conn_particle in self.connection_particles[relay2]:
                                if self.is_connected(particle, conn_particle):
                                    leaf_nodes.append(conn_particle)

            self.star_graphs.append({
                "relay_id": relay_id,
                "center": control_particle,
                "leaves": leaf_nodes,
            })

        self.save_circuit_diagram("01_Star_Subgraphs_Created")
        return self.circuit, self.star_graphs

    def protocol2_center_migration(self, center_particle, new_center_particle):
        self.log_operation("=== Protocol 2: center migration ===")
        self.circuit.h(center_particle)

        if self.phase2_meas_index >= self.classical_registers["phase2"].size:
            raise RuntimeError("No available phase2 classical bit.")

        creg = self.classical_registers["phase2"][self.phase2_meas_index]
        self.phase2_meas_index += 1

        self.circuit.measure(center_particle, creg)
        self.measured_qubits.add(qidx(center_particle))

        self.circuit.h(new_center_particle)
        self.circuit.z(new_center_particle).c_if(creg, 1)

        for graph in self.star_graphs:
            if qidx(center_particle) == qidx(graph["center"]):
                graph["leaves"] = [
                    leaf for leaf in graph["leaves"]
                    if qidx(leaf) not in {
                        qidx(new_center_particle), qidx(center_particle)
                    }
                ]
                graph["center"] = new_center_particle
                break

        self.save_circuit_diagram("02_Center_Migration")
        return self.circuit

    def protocol3_subgraph_fusion(
        self, center1, particle1, center2, particles_to_correct
    ):
        self.log_operation("=== Protocol 3: star fusion ===")

        self.circuit.cz(center2, particle1)
        self.circuit.h(particle1)

        if self.phase3_meas_index >= self.classical_registers["phase3"].size:
            raise RuntimeError("No available phase3 classical bit.")

        creg = self.classical_registers["phase3"][self.phase3_meas_index]
        self.phase3_meas_index += 1

        self.circuit.measure(particle1, creg)
        self.measured_qubits.add(qidx(particle1))

        self.circuit.h(center2)
        self.circuit.z(center2).c_if(creg, 1)
        for node in particles_to_correct:
            self.circuit.z(node).c_if(creg, 1)

        graph1_index = None
        graph2_index = None

        for i, graph in enumerate(self.star_graphs):
            if qidx(center1) == qidx(graph["center"]):
                graph1_index = i
            if qidx(center2) == qidx(graph["center"]):
                graph2_index = i

        if graph1_index is None or graph2_index is None:
            raise RuntimeError("Cannot locate the two star graphs for fusion.")

        leaf_nodes1 = [
            leaf for leaf in self.star_graphs[graph1_index]["leaves"]
            if qidx(leaf) != qidx(particle1)
        ]
        leaf_nodes2 = [
            leaf for leaf in self.star_graphs[graph2_index]["leaves"]
            if qidx(leaf) != qidx(center2)
        ]

        leaf_nodes1.append(center2)
        leaf_nodes1.extend(leaf_nodes2)
        self.star_graphs[graph1_index]["leaves"] = leaf_nodes1

        self.star_graphs[graph1_index]["relay_id"] = (
            f"{self.star_graphs[graph1_index]['relay_id']}-"
            f"{self.star_graphs[graph2_index]['relay_id']}"
        )

        self.star_graphs.pop(graph2_index)

        current_center = self.star_graphs[graph1_index]["center"]
        self.protocol2_center_migration(current_center, center2)

        self.save_circuit_diagram("03_Subgraph_Fusion")
        return self.circuit

    def protocol3_fuse_star_graphs(self, relay_id1, relay_id2):
        graph1 = None
        graph2 = None

        for graph in self.star_graphs:
            if graph["relay_id"] == relay_id1:
                graph1 = graph
            if graph["relay_id"] == relay_id2:
                graph2 = graph

        if graph1 is None or graph2 is None:
            raise RuntimeError(
                f"Cannot find star graphs for relay IDs {relay_id1}, {relay_id2}."
            )

        center1 = graph1["center"]
        center2 = graph2["center"]

        def original_relays(relay_id):
            if isinstance(relay_id, int):
                return [relay_id]
            return [int(x) for x in relay_id.split("-")]

        orig1 = original_relays(relay_id1)
        orig2 = original_relays(relay_id2)

        found_conn = None
        for conn in self.connections:
            if (
                (conn[0] in orig1 and conn[1] in orig2)
                or (conn[0] in orig2 and conn[1] in orig1)
            ):
                found_conn = conn
                break

        particle1 = None
        particle2 = None

        if found_conn is not None:
            for conn_info in self.original_connections_info:
                if set(conn_info["relays"]) == set(found_conn):
                    particle1, particle2 = conn_info["particles"]
                    break

        if particle1 is None:
            common = set(graph1["leaves"]) & set(graph2["leaves"])
            if common:
                particle1 = common.pop()
                particle2 = particle1

        if particle1 is None and found_conn is not None:
            r1, r2 = found_conn
            if (
                r1 in self.connection_particles
                and r2 in self.connection_particles
            ):
                particle1 = self.connection_particles[r1][0]
                particle2 = self.connection_particles[r2][0]

        if particle1 is None or particle2 is None:
            raise RuntimeError(
                f"Cannot find connection particles between {relay_id1} and {relay_id2}."
            )

        if qidx(particle2) in [qidx(p) for p in graph1["leaves"]]:
            particle1 = particle2

        self.protocol3_subgraph_fusion(
            center1, particle1, center2, list(graph2["leaves"])
        )
        return self.circuit

    def get_network_statistics(self):
        if self.circuit is None:
            return {}

        ops = self.circuit.count_ops()
        final_particle_count = (
            1 + len(self.star_graphs[0]["leaves"])
            if self.star_graphs else 0
        )

        return {
            "num_relays": self.num_relays,
            "peripherals_per_relay": self.peripherals_list[0],
            "bell_pairs": sum(self.peripherals_list) + len(self.connections),
            "local_two_qubit_gates": int(ops.get("cz", 0)),
            "final_particle_count": final_particle_count,
            "circuit_depth": self.circuit.depth(),
        }


def build_serial_network(m, n):
    net = QuantumNetworkProtocol(m, n, "linear")
    net.initialize_circuit()
    net.protocol1_create_subgraphs()

    current_graph_id = 0
    for next_relay_id in range(1, m):
        net.protocol3_fuse_star_graphs(current_graph_id, next_relay_id)
        current_graph_id = f"{current_graph_id}-{next_relay_id}"

    if not net.star_graphs:
        raise RuntimeError("No final star graph was produced.")

    final_graph = net.star_graphs[0]
    if not final_graph["leaves"]:
        raise RuntimeError("No terminal leaf is available for final center migration.")

    net.protocol2_center_migration(
        final_graph["center"], final_graph["leaves"][0]
    )

    final_graph = net.star_graphs[0]
    center = qidx(final_graph["center"])
    leaves = [qidx(q) for q in final_graph["leaves"]]

    output = [center] + leaves
    if len(set(output)) != m * n:
        raise RuntimeError(
            f"Output bookkeeping mismatch: expected {m*n}, got {len(set(output))}."
        )

    return net, center, leaves


def _empty_like(qc):
    return QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)


def _append_data(dst, data):
    for inst in data:
        dst.append(inst.operation, inst.qubits, inst.clbits)


def _split_before_first_protocol_cz(qc):
    data = list(qc.data)
    first_cz = next(
        (i for i, inst in enumerate(data) if inst.operation.name == "cz"),
        len(data),
    )

    prefix = _empty_like(qc)
    suffix = _empty_like(qc)
    _append_data(prefix, data[:first_cz])
    _append_data(suffix, data[first_cz:])
    return prefix, suffix


def apply_protocol_noise(
    protocol_circuit,
    single_qubit_gate_error=None,
    two_qubit_gate_error=None,
    measurement_error_rate=None,
):
    if single_qubit_gate_error is None:
        single_qubit_gate_error = NOISE_PARAMS["single_qubit_gate_error"]
    if two_qubit_gate_error is None:
        two_qubit_gate_error = NOISE_PARAMS["two_qubit_gate_error"]
    if measurement_error_rate is None:
        measurement_error_rate = NOISE_PARAMS["measurement_error_rate"]

    prefix, suffix = _split_before_first_protocol_cz(protocol_circuit)

    err_1q = depolarizing_error(single_qubit_gate_error, 1).to_instruction()
    err_2q = depolarizing_error(two_qubit_gate_error, 2).to_instruction()
    err_meas = pauli_error([
        ("X", measurement_error_rate),
        ("I", 1.0 - measurement_error_rate),
    ]).to_instruction()

    noisy = _empty_like(protocol_circuit)
    _append_data(noisy, prefix.data)

    for inst in suffix.data:
        name = inst.operation.name

        if name == "measure":
            noisy.append(err_meas, [inst.qubits[0]], [])

        noisy.append(inst.operation, inst.qubits, inst.clbits)

        if name == "h":
            noisy.append(err_1q, [inst.qubits[0]], [])
        elif name == "cz":
            noisy.append(err_2q, list(inst.qubits), [])

    return noisy


def make_ideal_witness_readout(protocol_circuit, center, leaves, setting):
    qc = protocol_circuit.copy()
    final_reg = next(reg for reg in qc.cregs if reg.name == "final_meas")
    output = [center] + leaves

    if setting == "pop":
        for idx in leaves:
            qc.h(qc.qubits[idx])
    elif setting == "coh":
        qc.h(qc.qubits[center])
    else:
        raise ValueError("setting must be 'pop' or 'coh'")

    for idx in output:
        qc.measure(qc.qubits[idx], final_reg[idx])

    return qc


def _final_register_bits(state, total_qubits):
    group = state.split()[0]
    return group.zfill(total_qubits)[::-1]


def population_term(counts, total_qubits, output_indices):
    good = 0
    total = 0

    for state, count in counts.items():
        bits = _final_register_bits(state, total_qubits)
        vals = [bits[i] for i in output_indices]
        if vals and all(v == vals[0] for v in vals):
            good += count
        total += count

    return good / total if total else 0.0


def coherence_term(counts, total_qubits, output_indices):
    value = 0.0
    total = 0

    for state, count in counts.items():
        bits = _final_register_bits(state, total_qubits)
        parity = sum(int(bits[i]) for i in output_indices) % 2
        value += (1.0 if parity == 0 else -1.0) * count
        total += count

    return value / total if total else 0.0


def fidelity_lower_bound(p_star, c_star):
    return float(np.clip(p_star + 0.5 * c_star - 0.5, 0.0, 1.0))


def simulate_observables(m, n, shots=SHOTS, noisy=True):
    net, center, leaves = build_serial_network(m, n)
    protocol_circuit = net.circuit.copy()

    if noisy:
        protocol_circuit = apply_protocol_noise(protocol_circuit)

    pop_circuit = make_ideal_witness_readout(
        protocol_circuit, center, leaves, "pop"
    )
    coh_circuit = make_ideal_witness_readout(
        protocol_circuit, center, leaves, "coh"
    )

    seed_base = BASE_SEED + 100 * int(m) + int(n)

    if noisy:
        counts_pop = net.noisy_backend.run(
            pop_circuit, shots=shots, seed_simulator=seed_base
        ).result().get_counts()

        counts_coh = net.noisy_backend.run(
            coh_circuit, shots=shots, seed_simulator=seed_base + 1
        ).result().get_counts()
    else:
        counts_pop = execute(
            pop_circuit, net.backend, shots=shots, seed_simulator=seed_base
        ).result().get_counts()

        counts_coh = execute(
            coh_circuit, net.backend, shots=shots, seed_simulator=seed_base + 1
        ).result().get_counts()

    output = [center] + leaves
    p_star = population_term(counts_pop, pop_circuit.num_qubits, output)
    c_star = coherence_term(counts_coh, coh_circuit.num_qubits, output)

    return {
        "p_star": p_star,
        "c_star": c_star,
        "F_lb_circuit": fidelity_lower_bound(p_star, c_star),
        "N": len(output),
        "S_links": net.get_network_statistics()["bell_pairs"],
    }


def serial_timing(m, n, params=None):
    if params is None:
        params = HW_PARAMS

    t_backbone = params["L_link"] / params["c_fiber"] + params["T_proc"]
    t_access = params["L_access"] / params["c_fiber"] + params["T_proc"]

    t_init_q = (
        n * params["T_2q"]
        + params["T_meas"]
        + params["T_1q"]
    )
    t_fusion_q = (
        params["T_2q"]
        + params["T_meas"]
        + params["T_1q"]
    )
    t_migration_q = params["T_meas"] + params["T_1q"]
    t_final_q = params["T_meas"] + params["T_1q"]

    t_quantum = (
        t_init_q
        + (m - 1) * (t_fusion_q + t_migration_q)
        + t_final_q
    )
    t_classical = 2 * t_access + 2 * (m - 1) * t_backbone

    return {
        "quantum": t_quantum,
        "classical": t_classical,
        "total": t_quantum + t_classical,
    }


def harmonic_number(k):
    return sum(1.0 / j for j in range(1, int(k) + 1))


def expected_link_preparation_time(S_links, T_gen):
    if T_gen <= 0:
        return 0.0
    return T_gen * harmonic_number(S_links)


def expected_wait_before_start(S_links, T_gen):
    if T_gen <= 0:
        return 0.0
    return T_gen * (harmonic_number(S_links) - 1.0)


def memory_exposure(m, n, T_gen=0.0):
    N = m * n
    S_links = m * n + (m - 1)
    tau_mem = (
        serial_timing(m, n)["total"]
        + expected_wait_before_start(S_links, T_gen)
    )
    return N * tau_mem


def apply_memory_dephasing(p_star, c_star, Q, T2):
    if T2 <= 0:
        return fidelity_lower_bound(p_star, 0.0)
    c_mem = c_star * math.exp(-Q / T2)
    return fidelity_lower_bound(p_star, c_mem)


def run_ideal_verification(m=5, n=5, shots=2000):
    print("=" * 72)
    print("Ideal protocol verification")
    print("=" * 72)

    obs = simulate_observables(m, n, shots=shots, noisy=False)
    print(f"m={m}, n={n}, N={obs['N']}")
    print(f"P_star       = {obs['p_star']:.6f}")
    print(f"C_star       = {obs['c_star']:.6f}")
    print(f"F_LB         = {obs['F_lb_circuit']:.6f}")

    if obs["F_lb_circuit"] < 0.99:
        print("WARNING: ideal verification below 0.99.")

    return obs


def run_resource_utilization_analysis():
    print("=" * 72)
    print("Fig. 4: Resource Utilization")
    print("=" * 72)

    rows = []
    max_bell_dev = 0
    max_twoq_dev = 0

    for m in range(3, 13):
        for n in (3, 4, 5, 6):
            N = m * n
            S = m * (n + 1)
            N_bell = S - 1
            N_2q = S - 2

            net, _, _ = build_serial_network(m, n)
            st = net.get_network_statistics()

            max_bell_dev = max(max_bell_dev, abs(st["bell_pairs"] - N_bell))
            max_twoq_dev = max(
                max_twoq_dev,
                abs(st["local_two_qubit_gates"] - N_2q),
            )

            rows.append({
                "m": m,
                "n": n,
                "N": N,
                "S": S,
                "N_Bell": N_bell,
                "N_2Q": N_2q,
                "C_ent": N_bell / N,
                "C_2Q": N_2q / N,
            })

    df = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/fig4_resource_utilization_{stamp}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    markers = {3: "o", 4: "s", 5: "^", 6: "D"}

    for n in (3, 4, 5, 6):
        sub = df[df["n"] == n].sort_values("m")
        ax1.plot(
            sub["m"], sub["C_ent"],
            marker=markers[n], linewidth=2, label=rf"$n={n}$"
        )
        ax2.plot(
            sub["m"], sub["C_2Q"],
            marker=markers[n], linewidth=2, label=rf"$n={n}$"
        )

    ax1.set_xlabel(r"Number of relay nodes $m$")
    ax1.set_ylabel(r"Bell-link cost per delivered qubit $\mathcal{C}_{\rm ent}$")
    ax1.set_title("(a) Entanglement-resource utilization")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.set_xlabel(r"Number of relay nodes $m$")
    ax2.set_ylabel(r"Two-qubit-gate cost per delivered qubit $\mathcal{C}_{2Q}$")
    ax2.set_title("(b) Local two-qubit-gate utilization")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figure4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Max Bell-count deviation from S-1: {max_bell_dev}")
    print(f"Max 2Q-count deviation from S-2:   {max_twoq_dev}")
    print("Saved: figure4.png")
    return df


def run_temporal_feasibility_analysis(n=4):
    print("=" * 72)
    print("Fig. 5: Temporal Feasibility and Stochastic Link Preparation")
    print("=" * 72)

    mgrid = np.arange(3, 13)

    rows = []
    for m in mgrid:
        timing = serial_timing(int(m), n)
        S_links = int(m) * n + (int(m) - 1)
        rows.append({
            "m": int(m),
            "n": n,
            "N": int(m) * n,
            "S_links": S_links,
            "T_quantum_ms": 1e3 * timing["quantum"],
            "T_classical_ms": 1e3 * timing["classical"],
            "T_serial_ms": 1e3 * timing["total"],
            "H_S_links": harmonic_number(S_links),
        })

    df = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/fig5_temporal_link_{stamp}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax1.plot(
        df["m"], df["T_serial_ms"],
        "o-", linewidth=2, label=r"Post-link $T_{\rm serial}$"
    )
    ax1.plot(
        df["m"], df["T_classical_ms"],
        "s--", linewidth=1.8, label="Classical component"
    )
    ax1.axhline(
        1e3 * HW_PARAMS["T2_ref"],
        linestyle=":", linewidth=1.8, label=r"$T_2=1$ s reference"
    )
    ax1.set_yscale("log")
    ax1.set_xlabel(r"Number of relay nodes $m$")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(rf"(a) Post-link temporal scaling ($n={n}$)")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8)

    ax2.plot(
        df["N"], df["H_S_links"], "o-", linewidth=2
    )
    ax2.set_xlabel(r"Delivered terminal qubits $N=mn$")
    ax2.set_ylabel(
        r"$\mathbb{E}[T_{\rm prep}]/T_{\rm gen}=H_{S_{\rm link}}$"
    )
    ax2.set_title("(b) Synchronized link-preparation baseline")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("figure5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Timing formula with Table-I parameters:")
    print("  T_quantum   = 0.42 m + 0.10 n - 0.10 ms")
    print("  T_classical = 0.60 m - 0.40 ms")
    print("  T_serial    = 1.02 m + 0.10 n - 0.50 ms")
    print("Saved: figure5.png")
    return df


def run_noise_analysis(shots=SHOTS):
    print("=" * 72)
    print("Fig. 6: Noise Analysis")
    print("=" * 72)
    print("Panel (a): circuit noise only.")
    print("Panel (b): T2=1 s and synchronized link waiting.")
    print("Final diagnostic basis rotations/readout are ideal.")
    print("=" * 72)

    rows = []

    configs = [
        (m, n)
        for n in (3, 4, 5, 6)
        for m in range(3, 13)
    ]

    for step, (m, n) in enumerate(configs, 1):
        print(
            f"\rCircuit simulation {step}/{len(configs)}: m={m}, n={n}",
            end="",
        )

        obs = simulate_observables(m, n, shots=shots, noisy=True)
        rows.append({
            "m": m,
            "n": n,
            "N": m * n,
            "P_star": obs["p_star"],
            "C_star": obs["c_star"],
            "F_lb_circuit": obs["F_lb_circuit"],
        })

    print()
    df = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/fig6_circuit_noise_{stamp}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    markers = {3: "o", 4: "s", 5: "^", 6: "D"}

    for n in (3, 4, 5, 6):
        sub = df[df["n"] == n].sort_values("N")
        ax1.plot(
            sub["N"], sub["F_lb_circuit"],
            marker=markers[n], linewidth=2, label=rf"$n={n}$"
        )

    ax1.axhline(
        0.5, linestyle=":", linewidth=1.8,
        label="Sufficient GME-certification threshold"
    )
    ax1.set_xlabel(r"Network size $N=mn$")
    ax1.set_ylabel(r"Star-state fidelity lower bound $F_{\rm LB}$")
    ax1.set_title("(a) Circuit-noise scaling")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    n = 4
    sub = df[df["n"] == n].sort_values("m")
    waiting_rows = []

    for T_gen in TGEN_SWEEP:
        vals = []
        for _, row in sub.iterrows():
            m = int(row["m"])
            Q = memory_exposure(m, n, T_gen=T_gen)
            F = apply_memory_dephasing(
                row["P_star"], row["C_star"], Q, HW_PARAMS["T2_ref"]
            )
            vals.append(F)
            waiting_rows.append({
                "m": m,
                "n": n,
                "N": int(row["N"]),
                "T_gen_ms": 1e3 * T_gen,
                "T2_ms": 1e3 * HW_PARAMS["T2_ref"],
                "Q_s": Q,
                "P_star": row["P_star"],
                "C_star": row["C_star"],
                "F_lb": F,
            })

        label = (
            "No additional waiting"
            if T_gen == 0
            else rf"$T_{{\rm gen}}={1e3*T_gen:g}$ ms"
        )
        ax2.plot(sub["N"], vals, "o-", linewidth=2, label=label)

    ax2.axhline(
        0.5, linestyle=":", linewidth=1.8,
        label="Sufficient GME-certification threshold"
    )
    ax2.set_xlabel(r"Network size $N=mn$")
    ax2.set_ylabel(r"Star-state fidelity lower bound $F_{\rm LB}$")
    ax2.set_title(r"(b) Link-waiting sensitivity ($n=4$, $T_2=1$ s)")
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figure6.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(waiting_rows).to_csv(
        f"results/fig6_link_waiting_{stamp}.csv", index=False
    )

    print("Saved: figure6.png")
    return df


def run_memory_coherence_sensitivity(shots=SHOTS):
    print("=" * 72)
    print("Fig. 7: Memory-Coherence Sensitivity")
    print("=" * 72)
    print("Fixed: n=4, no additional stochastic link waiting.")
    print("=" * 72)

    n = 4
    rows = []

    for step, m in enumerate(range(3, 13), 1):
        print(f"\rCircuit simulation {step}/10: m={m}, n={n}", end="")

        obs = simulate_observables(m, n, shots=shots, noisy=True)
        Q = memory_exposure(m, n, T_gen=0.0)

        for T2 in T2_SWEEP:
            rows.append({
                "m": m,
                "n": n,
                "N": m * n,
                "T2_ms": 1e3 * T2,
                "Q_s": Q,
                "Q_over_T2": Q / T2,
                "P_star": obs["p_star"],
                "C_star": obs["c_star"],
                "F_lb_circuit": obs["F_lb_circuit"],
                "F_lb": apply_memory_dephasing(
                    obs["p_star"], obs["c_star"], Q, T2
                ),
            })

    print()
    df = pd.DataFrame(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results/fig7_t2_sensitivity_{stamp}.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    specs = {
        1.0: ("o", "--"),
        10.0: ("s", "-"),
        100.0: ("^", "-"),
        1000.0: ("D", "-"),
    }

    for T2_ms in (1.0, 10.0, 100.0, 1000.0):
        sub = df[np.isclose(df["T2_ms"], T2_ms)].sort_values("N")
        marker, ls = specs[T2_ms]
        kwargs = {
            "marker": marker,
            "linestyle": ls,
            "linewidth": 2,
            "label": rf"$T_2={T2_ms:g}$ ms",
        }
        if T2_ms == 1.0:
            kwargs["markerfacecolor"] = "none"
            kwargs["markeredgewidth"] = 1.8
            kwargs["zorder"] = 6

        ax1.plot(sub["N"], sub["F_lb"], **kwargs)

    ax1.axhline(
        0.5, linestyle=":", linewidth=1.8,
        label="Sufficient GME-certification threshold"
    )
    ax1.set_xlabel(r"Network size $N=mn$")
    ax1.set_ylabel(r"Star-state fidelity lower bound $F_{\rm LB}$")
    ax1.set_title("(a) Finite-memory sensitivity ($n=4$; no additional waiting)")
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    for T2_ms, marker, ls in ((1.0, "o", "--"), (10.0, "s", "-")):
        sub = df[np.isclose(df["T2_ms"], T2_ms)].sort_values("N")
        kwargs = {
            "marker": marker,
            "linestyle": ls,
            "linewidth": 2,
            "label": rf"$T_2={T2_ms:g}$ ms",
        }
        if T2_ms == 1.0:
            kwargs["markerfacecolor"] = "none"
            kwargs["markeredgewidth"] = 1.8

        ax2.plot(sub["N"], sub["Q_over_T2"], **kwargs)

    ax2.axhline(
        1.0, linestyle=":", linewidth=1.5, label=r"$Q/T_2=1$"
    )
    ax2.set_xlabel(r"Network size $N=mn$")
    ax2.set_ylabel(r"Memory-dephasing exponent $Q/T_2$")
    ax2.set_title("(b) Short-coherence dephasing scale")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("figure7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved: figure7.png")
    return df


def print_model_summary():
    print("\nModel summary")
    print("-" * 72)
    print("Protocol 1-3: relay-terminal star-state construction")
    print("Fig.4: resource utilization")
    print("  C_ent = (S-1)/N = 1 + 1/n - 1/(mn)")
    print("  C_2Q  = (S-2)/N = 1 + 1/n - 2/(mn)")
    print("Fig.5: temporal feasibility + harmonic link baseline")
    print("Fig.6: two-setting star-state fidelity lower bound")
    print("Fig.7: finite-T2 sensitivity at 1, 10, 100, 1000 ms")
    print(f"Shots: {SHOTS}; seed: {BASE_SEED}")
    print("Memory: C_star -> C_star exp(-Q/T2)")
    print("Q = N[T_serial + T_gen(H_Slink-1)]")
    print("T_gen=0 is used only as a no-additional-waiting baseline.")
    print("Not modeled: Bell infidelity, purification, asynchronous optimization")


if __name__ == "__main__":
    print("Select calculation:")
    print("1. Ideal protocol check")
    print("2. Resource utilization (Fig. 4)")
    print("3. Timing and link preparation (Fig. 5)")
    print("4. Noise analysis (Fig. 6)")
    print("5. Memory-coherence sensitivity (Fig. 7)")
    print("6. Run Figs. 4-7")
    print("7. Model summary")

    choice = input("Enter choice (1-7): ").strip()

    if choice == "1":
        run_ideal_verification()
    elif choice == "2":
        run_resource_utilization_analysis()
    elif choice == "3":
        run_temporal_feasibility_analysis()
    elif choice == "4":
        run_noise_analysis()
    elif choice == "5":
        run_memory_coherence_sensitivity()
    elif choice == "6":
        run_resource_utilization_analysis()
        run_temporal_feasibility_analysis()
        run_noise_analysis()
        run_memory_coherence_sensitivity()
    elif choice == "7":
        print_model_summary()
    else:
        print("Invalid choice.")