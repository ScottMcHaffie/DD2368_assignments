import torch
import math

def get_device(gpu_no = 0):
    return torch.device(f'cuda:{gpu_no}' if torch.cuda.is_available() else 'cpu')

class QuantumCircuit:
    def __init__(self, n_qubits: int, state_vector = None, device = 'cuda', gpu_no = 0):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        
        # Handle device specification
        if isinstance(device, torch.device):
            self.device = device
        elif device == 'cuda':
            self.device = get_device(gpu_no)
        else:
            self.device = torch.device(device)

        if state_vector is None:
            state_vector = torch.zeros(self.dim, device = self.device, dtype = torch.cfloat)
            state_vector[0] = 1
            self.state_vector = state_vector.reshape(-1, 1)
        else:
            if state_vector.shape[0] == self.dim:
                # Ensure state_vector is on the correct device
                self.state_vector = state_vector.to(self.device, dtype = torch.cfloat)
            else:
                raise ValueError(f'State vector size must match 2^{n_qubits}.')

        # Identity and Pauli gates with proper device
        self.I = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device = self.device, dtype = torch.cfloat)
        self.Pauli_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device = self.device, dtype = torch.cfloat)
        self.Pauli_Y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], device = self.device, dtype = torch.cfloat)
        self.Pauli_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], device = self.device, dtype = torch.cfloat)

        # Projectors
        self.proj_0  = torch.tensor([[1.0, 0.0], [0.0, 0.0]], device = self.device, dtype = torch.cfloat)
        self.proj_1  = torch.tensor([[0.0, 0.0], [0.0, 1.0]], device = self.device, dtype = torch.cfloat)

        # Hadamard gate
        self.H = (1 / torch.sqrt(torch.tensor(2.0, device=self.device))) * torch.tensor([[1.0, 1.0], [1.0, -1.0]], device = self.device, dtype = torch.cfloat)

    def single_qubits(self, target: int, gate: torch.Tensor):
        if target < 0 or self.n_qubits <= target:
            raise ValueError('Target out of bounds')
        else:
            # Initialize with proper device
            single_gate = torch.tensor(1.0, device = self.device, dtype = torch.cfloat)
            for index in range(self.n_qubits):
                if index == target:
                    single_gate = torch.kron(single_gate, gate)
                else:
                    single_gate = torch.kron(single_gate, self.I)
            self.state_vector = torch.matmul(single_gate, self.state_vector).to(self.device)
        return self.state_vector

    def controlled_gate(self, control: int, target: int, gate: torch.Tensor):
        if control < 0 or self.n_qubits <= control:
            raise ValueError(f'Control qubits {control} is out of bounds (must be 0 ≤ qubit < {self.n_qubits}).')
        elif target < 0 or self.n_qubits <= target:
            raise ValueError(f'Target qubit {control} is out of bounds (must be 0 ≤ qubit < {self.n_qubits}).')
        elif control == target:
            raise ValueError(f'Control and target must differ.')
        else:
            # Initialize with proper device
            control_gate_0 = torch.tensor(1.0, device = self.device, dtype = torch.cfloat)
            control_gate_1 = torch.tensor(1.0, device = self.device, dtype = torch.cfloat)
    
            for index in range(self.n_qubits):
                if index == control:
                    control_gate_0 = torch.kron(control_gate_0, self.proj_0)
                    control_gate_1 = torch.kron(control_gate_1, self.proj_1)
                elif index == target:
                    control_gate_0 = torch.kron(control_gate_0, self.I)
                    control_gate_1 = torch.kron(control_gate_1, gate)
                else:
                    control_gate_0 = torch.kron(control_gate_0, self.I)
                    control_gate_1 = torch.kron(control_gate_1, self.I)
            
            control_gate = control_gate_0 + control_gate_1
            self.state_vector = torch.matmul(control_gate, self.state_vector)
            return self.state_vector

    def XGate(self, target: int):
        self.single_qubits(target, self.Pauli_X)

    def YGate(self, target: int):
        self.single_qubits(target, self.Pauli_Y)

    def ZGate(self, target: int):
        self.single_qubits(target, self.Pauli_Z)

    def HGate(self, target: int):
        self.single_qubits(target, self.H)

    def Rx(self, target: int, theta):
        # Ensure theta is a tensor on the correct device
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device)
        else:
            theta = theta.to(self.device)
            
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        Rx_matrix = torch.stack([
            torch.stack([cos, -1.0j * sin]),
            torch.stack([-1.0j * sin, cos])], dim=0).to(self.device)
        self.single_qubits(target, Rx_matrix)

    def Ry(self, target: int, theta):
        # Ensure theta is a tensor on the correct device
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device)
        else:
            theta = theta.to(self.device)
            
        cos = torch.cos(theta / 2)
        sin = torch.sin(theta / 2)
        Ry_matrix = torch.stack([
            torch.stack([cos, -sin]),
            torch.stack([sin, cos])], dim=0).to(self.device)
        self.single_qubits(target, Ry_matrix)

    def Rz(self, target: int, theta):
        # Ensure theta is a tensor on the correct device
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta, device=self.device)
        else:
            theta = theta.to(self.device)
            
        e1 = torch.exp(-0.5j * theta)
        e2 = torch.exp(0.5j * theta)
        Rz_mat = torch.tensor([[e1, 0.0],
                           [0.0, e2]],
                           dtype = torch.cfloat,
                           device = self.device)
        self.single_qubits(target, Rz_mat)

    def Ry_layer(self, angs: torch.Tensor):
        angs = angs.to(self.device, dtype=torch.cfloat)
        cos = torch.cos(angs[0] / 2)
        sin = torch.sin(angs[0] / 2)
        rot = torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])

        for index in range(1, len(angs)):
            cos = torch.cos(angs[index] / 2)
            sin = torch.sin(angs[index] / 2)
            rot = torch.kron(rot, torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])]))
        self.state_vector = torch.matmul(rot, self.state_vector)
        return self.state_vector

    def Rz_layer(self, angs: torch.Tensor):
        angs = angs.to(self.device)
        exp_ang = torch.exp(1.0j * angs[0])
        zero = torch.tensor(0.0, dtype = torch.cfloat, device = self.device)
        one = torch.tensor(1.0, dtype = torch.cfloat, device = self.device)
        rot = torch.stack([torch.stack([one, zero]), torch.stack([zero, exp_ang])])
       
        for index in range(1, len(angs)):
            exp_ang = torch.exp(1.0j * angs[index])
            rot = torch.kron(rot, torch.stack([torch.stack([one, zero]), torch.stack([zero, exp_ang])]))
        self.state_vector = torch.matmul(rot, self.state_vector)
        return self.state_vector

    def cx(self, control: int, target: int):
        self.controlled_gate(control, target, self.Pauli_X)

    def cz(self, control: int, target: int):
        self.controlled_gate(control, target, self.Pauli_Z)

    def cx_linear_layer(self):
        self.controlled_gate(self.n_qubits - 2, self.n_qubits - 1, self.Pauli_X)
        for index in range(self.n_qubits - 3, -1, -1):
            self.controlled_gate(index, index + 1, self.Pauli_X)

    def cz_linear_layer(self):
        self.controlled_gate(self.n_qubits - 2, self.n_qubits - 1, self.Pauli_Z)
        for index in range(self.n_qubits - 3, -1, -1):
            self.controlled_gate(index, index + 1, self.Pauli_Z)
    
    def probabilities(self):
        return self.state_vector.conj() * self.state_vector