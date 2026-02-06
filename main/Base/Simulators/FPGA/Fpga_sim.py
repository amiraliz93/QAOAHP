import typing
import numpy as np
import serial
import time
import struct
from ...qaoa_simulator_base import Sim_Base, CostsType, ParamType, TermsType


class FpgaDriver:
    """
    Purpose: Low-level UART serial interface to NTU FPGA hardware

    UART Protocol Implementation: Communicates with FPGA state machine
      via serial commands (opcode-based)

    Based on **NEW_smachine.sv and qaoa_system.sv** protocol
    AIM: low-level interface  that communicate with FPGA' binary protocol via UART serial port

    Data Flow: 
    PC <--> UART <--> FPGA State Machine <--> BRAM <--> QAOA System
        Writes data → internal registers (rT, rA, rB) → BRAM banks
        Reads BRAM → internal register (rU) → Python PC

    BRAM Organization:
        6 banks for state real/imag, costs, parameters
    Timing: Uses delays between operations (no handshaking - fire-and-forget protocol)

    Key Methods:
        connect() - Establish UART connection
        load_data() - Transfer problem (costs, initial state, γ/β params) to FPGA memory
        execute() - Send RUN command to start QAOA computation
        read_result() - Retrieve final statevector from FPGA
    """
    
 
    # Operation codes from NEW_smachine.sv
    OP_NONE = 0
    OP_SEND1T = 1      # Send 1 byte from PC --> rT
    OP_SEND8T = 2      # Send 8 bytes from PC --> rT
    OP_MOV_T2A = 3     # Move rT to rA
    OP_MOV_T2B = 4     # Move rT to rB
    OP_MOV_A2U = 5     # Move rA to rU (for output)
    OP_MOV_A2B = 6     # Move rA to rB
    OP_MOV_Info2U = 7  # Move Info to rU (Send firmware version info)
    OP_FETCH1U = 60    # Fetch 1 byte from rU --> PC
    OP_FETCH8U = 61    # Fetch 8 bytes (64-bit) from rU --> PC
    OP_INC_A = 84      # Increment rA by 1 --> rA = rA +1
    OP_WRITE_T2RAM = 111  # Write rT to BRAM at address rA ---  rT to BRAM[rA]
    OP_READ_RAM2U = 112   # Read BRAM at address rA to rU -- BRAM[rA] → rU
    OP_SEND_CMD = 118     # Send command to qaoa_system

    # QAOA system commands (from qaoa_system.sv)
    qa_INIT = 4   # Initialize QAOA system
    qa_WAIT = 1   # Wait state
    qa_RUN = 2    # Run QAOA layer
    
    # BRAM bank identifiers (bits 56-59 in address)
    BRAM_STATE_REAL = 0x0000000000000000      # BRAM[0]: State vector real part
    BRAM_STATE_IMAG = 0x2000000000000000      # BRAM[1]: State vector imaginary part
    BRAM_COST_FUNC = 0x4000000000000000       # BRAM[2]: Cost function
    BRAM_PARAMS = 0x6000000000000000          # BRAM[5]: Parameters (cos β, sin β, γ)

    #new code
    # Aim: 1- Computing intermediate values on FPGA 2- Preprocessing data before BRAM writes 
    # 3- Testing arithmetic units  4- Future optimizations ( computing cos/sin on FPGA instead of Python)
    OP_ADD_B2A = 80   # rA = rA +rB (64bit fixed, 2cycles)
    OP_MUL_B2A = 81   # rA = rA * rB (24bit fixed, 8 cycles)
    OP_ADDFP_B2A = 82 # rA = rA + rB (64bit float, 27 cycles)
    OP_ADDFP_B2A = 83 # rA = rA * rB (64bit float, 24 cycles)

    # Arithmetic operation latencies (used in state machine timing)
    # Aim: 1)These define how many clock cycles each arithmetic operation takes. -2)Status polling - Knowing when operations complete
    FP64_ADD_LATENCY = 27 # For example if  choose OP_ADDFP_B2A, must wait 27 cycles before next operation
    FP64_MUL_LATENCY = 24
    FIX64_ADD_LATENCY = 2
    FIX64_MUL_LATENCY = 8


    def __init__(self, port="COM3", baudrate=115200, timeout=1):
        """
        Initialize FPGA driver with serial port settings
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed (default 115200)
            timeout: Read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False
        self.version = None

    def connect(self):
        """Connect to FPGA via UART and verify version"""
        print(f"Connecting to FPGA on {self.port}...")
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            
            # Clear buffers
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            time.sleep(0.1)
            
            # Check version
            self.ser.write(bytes([self.OP_MOV_Info2U, self.OP_FETCH8U]))
            time.sleep(0.01)

            version_bytes = self.ser.read(8)
            if len(version_bytes) == 8:
                self.version = version_bytes.decode('ascii', errors='ignore').strip('\x00')
            
                if "NTUSMv" in self.version:
                    self.connected = True
                    print(f"✓ FPGA connected: {self.version}")
                    return True
                else:
                    print(f"✗ FPGA version check failed: {self.version}")
                    self.ser.close()
                    return False
            else:
                print(f"✗ FPGA version read failed, got {len(version_bytes)} bytes")
                self.ser.close()
                return False
        except serial.SerialException as e:
            print(f" serial error: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            self.connected = False
            return False

    def _send_opcode(self, opcode):
        """Send single opcode byte"""
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        self.ser.write(bytes([opcode]))
        time.sleep(1e-5)  # Prevent UART buffer overflow
        
        # in order to send pre-formatted bytes 
    
    def _send_byte(self, value):
        """Send single byte (64 bit integer in little-endian format)"""
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        self.ser.write(bytes([value]))
        time.sleep(1e-5)

    def _send_fp64(self, value):
        """Send 64-bit float in little-endian format"""
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        data = struct.pack('<d', value)
        self.ser.write(data)
        time.sleep(1e-5)

    def _send_int64(self, value):
        """Send 64-bit integer in little-endian format"""
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        data = value.to_bytes(8, byteorder='little', signed=False)
        self.ser.write(data)
        time.sleep(1e-5)

    def _fetch_fp64(self):
        """Fetch 64-bit float from FPGA"""
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        data = self.ser.read(8)
        if len(data) != 8:
            raise RuntimeError(f"Expected 8 bytes, got {len(data)}")
        return struct.unpack('<d', data)[0]

    def _write_bram_fp64(self, bram_bank, addr, value):
        """
        Write a 64-bit float to BRAM
        
        Args:
            bram_bank: BRAM bank selector (BRAM_STATE_REAL, etc.)
            addr: Address within BRAM (0 to 2^13-1)
            value: Float64 value to write
        """
        full_addr = bram_bank | addr
        
        # Set address in rA
        self._send_opcode(self.OP_SEND8T)
        self._send_int64(full_addr)
        self._send_opcode(self.OP_MOV_T2A)
        
        # Write value
        self._send_opcode(self.OP_SEND8T)
        self._send_fp64(value)
        self._send_opcode(self.OP_WRITE_T2RAM)

    def _read_bram_fp64(self, bram_bank, addr):
        """
        Read a 64-bit float from BRAM
        
        Args:
            bram_bank: BRAM bank selector
            addr: Address within BRAM
            
        Returns:
            Float64 value
        """
        full_addr = bram_bank | addr
        
        # Set address in rA
        self._send_opcode(self.OP_SEND8T)
        self._send_int64(full_addr)
        self._send_opcode(self.OP_MOV_T2A)
        
        # Read to rU and fetch
        self._send_opcode(self.OP_READ_RAM2U)
        self._send_opcode(self.OP_FETCH8U)
        
        return self._fetch_fp64()

    def load_data(self, diag_hamiltonian, initial_state, gammas, betas):
        """
        Load QAOA problem data to FPGA
        
        Args:
            diag_hamiltonian: Cost diagonal array (length 2^n_qubits)
            initial_state: Initial quantum state (complex128, length 2^n_qubits)
            gammas: Gamma parameters for each layer
            betas: Beta parameters for each layer
        """
        if not self.connected:
            raise RuntimeError("Not connected to FPGA")
        
        print("Loading data to FPGA...")
        n_states = len(initial_state)
        n_layers = len(gammas)
        
        try:
            # 1. Initialize QAOA system
            print("  Initializing QAOA system...")
            self._send_opcode(self.OP_SEND1T)
            self._send_byte(self.qa_INIT)
            self._send_opcode(self.OP_SEND_CMD)
            time.sleep(0.01)
            
            # 2. Write cost function to BRAM[2]
            print(f"  Writing {n_states} cost values to BRAM[2]...")
            for i in range(n_states):
                self._write_bram_fp64(self.BRAM_COST_FUNC, i, float(diag_hamiltonian[i]))
            
            # 3. Write initial state to BRAM[0] (real) and BRAM[1] (imaginary)
            print(f"  Writing {n_states} state amplitudes to BRAM[0,1]...")
            for i in range(n_states):
                self._write_bram_fp64(self.BRAM_STATE_REAL, i, initial_state[i].real)
                self._write_bram_fp64(self.BRAM_STATE_IMAG, i, initial_state[i].imag)
            
            # 4. Write parameters to BRAM[5]
            # Format: [cos(β₀), sin(β₀), γ₀, cos(β₁), sin(β₁), γ₁, ...]
            print(f"  Writing {n_layers} layer parameters to BRAM[5]...")
            for layer in range(n_layers):
                base_addr = layer * 3
                self._write_bram_fp64(self.BRAM_PARAMS, base_addr + 0, np.cos(betas[layer]))
                self._write_bram_fp64(self.BRAM_PARAMS, base_addr + 1, np.sin(betas[layer]))
                self._write_bram_fp64(self.BRAM_PARAMS, base_addr + 2, gammas[layer])
            
            print("✓ Data loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False

    def execute(self, n_layers):
        """
        Execute QAOA simulation on FPGA
        
        Args:
            n_layers: Number of QAOA layers to execute
        """
        if not self.connected:
            raise RuntimeError("Not connected to FPGA")
        
        print(f"Executing {n_layers} QAOA layers on FPGA...")
        
        try:
            # Send RUN command to start QAOA execution
            self._send_opcode(self.OP_SEND1T)
            self._send_byte(self.qa_RUN)
            self._send_opcode(self.OP_SEND_CMD)
            
            # Wait for execution (estimate: ~1ms per layer per state)
            # You may need to adjust this based on your FPGA clock speed
            # Or implement status polling
            time.sleep(0.1 * n_layers)
            
            print("✓ QAOA execution complete")
            return True
            
        except Exception as e:
            print(f"✗ Execution error: {e}")
            return False

    def read_result(self, n_states):
        """
        Read result statevector from FPGA
        
        Args:
            n_states: Number of states (2^n_qubits)
            
        Returns:
            Complex numpy array with final statevector
        """
        if not self.connected:
            print("Warning: Not connected, returning uniform state")
            return np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)
        
        print(f"Reading {n_states} state amplitudes from FPGA...")
        
        try:
            result = np.zeros(n_states, dtype=np.complex128)
            
            # Read from BRAM[0] (real) and BRAM[1] (imaginary)
            for i in range(n_states):
                real_part = self._read_bram_fp64(self.BRAM_STATE_REAL, i)
                imag_part = self._read_bram_fp64(self.BRAM_STATE_IMAG, i)
                result[i] = complex(real_part, imag_part)
            
            print(f"✓ Read {n_states} amplitudes")
            return result
            
        except Exception as e:
            print(f"✗ Error reading result: {e}")
            # Return default uniform state
            return np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)

    def disconnect(self):
        """Disconnect from FPGA"""
        if self.connected and self.ser:
            print("Disconnecting from FPGA...")
            try:
                # Send WAIT command before closing
                self._send_opcode(self.OP_SEND1T)
                self._send_byte(self.qa_WAIT)
                self._send_opcode(self.OP_SEND_CMD)
                time.sleep(0.01)
                self.ser.close()
                print("✓ Disconnected")
            except:
                pass
        self.connected = False

class FPGASimulator(Sim_Base):
    """
    FPGA-based QAOA Simulator for NTU Hardware
    AIM: 
        High-level QAOA simulator interface to Sim_Base abstract interface, making FPGA hardware appear like other backednd
    ---------------------------------------------
    How is work:
        Initialization: Validates qubit count (≤13 for 8192 states), creates FpgaDriver instance
        Problem Setup: Converts problem terms/costs into diagonal Hamiltonian array
        Simulation: Orchestrates hardware via driver:
            - Connects to FPGA
            - Loads cost diagonal, initial state, γ/β parameters
            - Executes QAOA layers
            - Reads final statevector
    ---------------------------------------------
    API Compatibility:
        Provides standard methods (get_expectation, get_overlap, get_probabilities)
          
    Note:
        Designed for NTU FPGA with specific UART protocol and BRAM layout
    ---------------------------------------------
    Key Methods:
        simulate_qaoa(gammas, betas) - Main entry point, returns final statevector
        get_expectation() - Computes ⟨ψ|H_C|ψ⟩ from result
        _diag_from_terms() - Converts problem terms to cost diagonal

    """
    
    
    _hc_diag: np.ndarray

    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
        fpga_config: dict | None = None,
    ) -> None:
        """
        Initialize FPGA-based QAOA simulator
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits (max 13 for NTU FPGA, supporting 2^13=8192 states)
        costs : CostsType | None
            Precomputed cost values
        terms : TermsType | None
            Hamiltonian terms
        fpga_config : dict, optional
            Configuration: {'port': 'COM3', 'baudrate': 115200, 'max_qubits': 13}
        """
        # Initialize base class
        super().__init__(n_qubits=n_qubits, costs=costs, terms=terms)
        
        # FPGA configuration
        self.fpga_config = fpga_config or {}
        default_config = {
            'port': 'COM3',
            'baudrate': 115200,
            'max_qubits': 13,  # NTU FPGA supports 2^13 = 8192 states
        }
        for key, value in default_config.items():
            if key not in self.fpga_config:
                self.fpga_config[key] = value
        
        # Validate qubit count
        if n_qubits > self.fpga_config['max_qubits']:
            raise ValueError(
                f"Number of qubits ({n_qubits}) exceeds FPGA maximum "
                f"({self.fpga_config['max_qubits']})"
            )
        
        # Initialize driver
        self.fpga = FpgaDriver(
            port=self.fpga_config['port'],
            baudrate=self.fpga_config['baudrate']
        )
        self.connected = False

    def _diag_from_terms(self, terms: TermsType) -> np.ndarray:
        """
        Compute the diagonal of the cost Hamiltonian from problem terms.
        
        Parameters
        ----------
        terms : TermsType
            List of Hamiltonian terms (coefficient, [qubit_indices])
            
        Returns
        -------
        np.ndarray
            Diagonal elements of the cost Hamiltonian
            """
        costs = np.zeros(self.n_states)
        for coeff, qubits in terms:
            for i in range(self.n_states):
                binary = [(i >> j) & 1 for j in range(self.n_qubits)]
                spins = [1 - 2 * binary[j] for j in qubits]
                costs[i] += coeff * np.prod(spins)
        return costs

    def _diag_from_costs(self, costs: CostsType) -> np.ndarray:
        """
        Process provided cost array for FPGA computation.
        Parameters
        ----------
        costs : CostsType
            Array of cost values for each computational basis state       
        Returns
        -------
        np.ndarray
            Processed cost array suitable for FPGA
        """
        return np.array(costs)

    def get_cost_diagonal(self) -> np.ndarray:
        """
        Return the diagonal of the cost Hamiltonian as a numpy array.
        Returns
        np.ndarray
            Diagonal elements of the cost Hamiltonian
        """
        return np.array(self._hc_diag)


    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Simulate QAOA on FPGA hardware
        
        Parameters
        ----------
        gammas : ParamType
            Phase separation parameters
        betas : ParamType
            Mixing parameters
        sv0 : np.ndarray | None
            Initial state (default: uniform superposition)
            
        Returns
        -------
        np.ndarray
            Final statevector
        """
        gammas_np = np.asarray(gammas)
        betas_np = np.asarray(betas)
        
        if len(gammas_np) != len(betas_np):
            raise ValueError(f"Parameter count mismatch: {len(gammas_np)} gammas vs {len(betas_np)} betas")
        
        # Initialize state
        if sv0 is None:
            sv0 = np.ones(self.n_states, dtype=np.complex128) / np.sqrt(self.n_states)
        
        n_layers = len(gammas_np)
        
        try:
            # Connect to FPGA
            if not self.connected:
                self.connected = self.fpga.connect()
                if not self.connected:
                    raise RuntimeError("Failed to connect to FPGA")
            
            # Load data
            success = self.fpga.load_data(self._hc_diag, sv0, gammas_np, betas_np)
            if not success:
                raise RuntimeError("Failed to load data to FPGA")
            
            # Execute
            success = self.fpga.execute(n_layers)
            if not success:
                raise RuntimeError("FPGA execution failed")
            
            # Read result
            result = self.fpga.read_result(self.n_states)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"FPGA simulation error: {str(e)}")
        
        finally:
            # Keep connection open for multiple calls
            pass

    def get_expectation(self, result, costs: typing.Any = None, optimization_type="min", **kwargs) -> float:
        """
        Calculate expectation value of the cost Hamiltonian from the final state.
        
        Parameters
        ----------
        result : np.ndarray
            Final state vector from QAOA simulation
        costs : typing.Any, optional
            Cost values (defaults to internal diagonal)
        optimization_type : str, optional
            Type of optimization problem ("min" or "max"), by default "min"
            
        Returns
        -------
        float
            Expectation value (negated for maximization problems)
        """
        if costs is None:
            costs = self._hc_diag
        probs = np.abs(result) ** 2
        expectation = np.sum(costs * probs)
        return -expectation if optimization_type == "max" else expectation

    def get_overlap(
        self, result, costs: CostsType | None = None, 
        indices: np.ndarray | typing.Sequence[int] | None = None,
        optimization_type="min", **kwargs
    ) -> float:
        """Calculate overlap with optimal states"""
        if indices is None:
            if costs is None:
                costs = self._hc_diag
            optimal_value = np.max(costs) if optimization_type == "max" else np.min(costs)
            optimal_indices = np.where(np.isclose(costs, optimal_value))[0]
        else:
            optimal_indices = indices
        
        overlap = 0.0
        for idx in optimal_indices:
            overlap += np.abs(result[idx]) ** 2
        return overlap

    def get_statevector(self, result: typing.Any, **kwargs) -> np.ndarray:
        """
        Return the statevector as a numpy array.
        
        Parameters
        ----------
        result : typing.Any
            Result from simulate_qaoa
            
        Returns
        -------
        np.ndarray
            State vector
        """
        return np.array(result)

    def get_probabilities(self, result: typing.Any, **kwargs) -> np.ndarray:
        """Return probabilities"""
        return np.abs(result) ** 2

    def __del__(self):
        """Cleanup: disconnect on deletion"""
        if self.connected:
            self.fpga.disconnect()
    


