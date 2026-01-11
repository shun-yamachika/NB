"""
Generate RB data for WLS-MN analysis
200 samples per m, m = 1 to 40
"""
import numpy as np
import netsquid as ns
from netsquid import pydynaa
from netsquid.qubits.ketstates import b00
from netsquid.qubits.qubitapi import QFormalism
from netsquid.nodes.connections import DirectConnection
from netsquid.components.qmemory import QuantumMemory
from netsquid.components.qsource import QSource
from netsquid.components.qchannel import QuantumChannel
from netsquid.nodes.node import Node
from netsquid.protocols import Protocol

from netsquid.components import (T1T2NoiseModel, QuantumChannel, ClassicalChannel,
                                  FibreLossModel, FibreDelayModel, DepolarNoiseModel,
                                  DephaseNoiseModel)
from netsquid.components.qprocessor import *
from netsquid.components.instructions import *
from netsquid.components.qprogram import *
from netsquid.util.simstats import *
from netsquid.qubits import ketstates as ks
from netsquid.qubits.state_sampler import StateSampler

from n_node_RB import MultiNodeRB
from teleportationnoisemodel import TeleportationNoiseModel

import pickle as pk

print("="*80)
print("Generating RB Data: 200 samples per m, m=1-40")
print("="*80)

# Set T1 and T2 times for the nodes
T1 = 10**9  # one second
T2 = 12 * 10**6  # 12 milliseconds
gate_time = 39*10**(3)  # 39 microseconds

# Set bright state population fraction
alpha = 0.95

ns.set_qstate_formalism(QFormalism.DM)

n_nodes = 2

# Parameters
min_bounces = 1
max_bounces = 40
n_samples = 200

print(f"\nExperiment parameters:")
print(f"  Nodes: {n_nodes}")
print(f"  Bounce range: {min_bounces} to {max_bounces}")
print(f"  Samples per bounce: {n_samples}")
print(f"  Total simulations: {(max_bounces - min_bounces + 1) * n_samples}")

# Run simulation
print(f"\nStarting simulation...")
print(f"This may take several minutes...")

ns.sim_reset()

nodes = [Node(f"{i}") for i in range(n_nodes)]

# Define teleportation quantum channel connections
qchannelsUP = {i: QuantumChannel(f"TelChannel{i}{i+1}", length=40,
                                  models={"delay_model": FibreDelayModel(),
                                         "quantum_noise_model": TeleportationNoiseModel(alpha=alpha)})
               for i in range(n_nodes)}
qchannelsDOWN = {i: QuantumChannel(f"TelChannel{i+1}{i}", length=40,
                                    models={"delay_model": FibreDelayModel(),
                                           "quantum_noise_model": TeleportationNoiseModel(alpha=alpha)})
                 for i in range(n_nodes)}

# Define quantum processors with T1T2 noise
qproc0 = QuantumProcessor("QPD-0", num_positions=2,
                          mem_noise_models=[T1T2NoiseModel(T1=T1, T2=T2)]*2)

qprocs = {i: QuantumProcessor(f"QPD-{i}", num_positions=1,
                              mem_noise_models=[T1T2NoiseModel(T1=T1, T2=T2)])
          for i in range(1, n_nodes)}

# Add quantum processors to the nodes
nodes[0].add_subcomponent(qproc0)
for i in range(1, n_nodes):
    nodes[i].add_subcomponent(qprocs[i])

# Add a source of states to node A_0
nodes[0].add_subcomponent(QSource("QSource", status=2))

# Add quantum channel connecting A_i and A_i+1
for i in range(n_nodes-1):
    nodes[i].connect_to(nodes[i+1],
                       DirectConnection(f"quantum_connect{i}{i+1}",
                                       qchannelsUP[i], qchannelsDOWN[i]),
                       label=f"quantum{i}{i+1}")

# Set up protocol instance
two_node_RB = MultiNodeRB(n_nodes=n_nodes,
                         min_bounces=min_bounces,
                         max_bounces=max_bounces,
                         n_samples=n_samples)

# Assign nodes to the RB protocol
for i in range(n_nodes):
    two_node_RB.nodes.append(nodes[i])

# Assign Clifford instructions to the processors
for instr in two_node_RB.cliffords:
    for i in range(n_nodes):
        two_node_RB.nodes[i].qmemory.add_instruction(
            instr, duration=gate_time, topology=[0],
            q_noise_model=T1T2NoiseModel(T1=T1, T2=T2))

# Trigger the start of the protocol
two_node_RB.start()
god = pydynaa.Entity()
god._schedule_now(two_node_RB.evtype_trigger)

# Start the simulation run
print(f"\nRunning NetSquid simulation...")
ns.sim_run()
print(f"Simulation complete!")

# Extract fidelity data and save
output_file = 'AB_decay_200samples_main.pickle'
with open(output_file, 'wb') as f:
    pk.dump({"decay data": two_node_RB.get_fidelity(),
             "endpoints": [min_bounces, max_bounces]}, f)

print(f"\n" + "="*80)
print(f"Data saved to: {output_file}")
print(f"="*80)

# Quick summary
fidelity_data = two_node_RB.get_fidelity()
fid_dict = fidelity_data[1]

print(f"\nData summary:")
print(f"  Bounce lengths: {min_bounces} to {max_bounces}")
print(f"  Number of bounce lengths: {len(fid_dict)}")
print(f"  Samples per bounce length: {n_samples}")

# Sample statistics
print(f"\n  Sample statistics:")
for m in [1, 10, 20, 30, 40]:
    if m in fid_dict:
        data_m = fid_dict[m]
        print(f"    m={m:2d}: mean={np.mean(data_m):.6f}, std={np.std(data_m, ddof=1):.6f}")

print(f"\n" + "="*80)
print("Data generation complete")
print("="*80)
