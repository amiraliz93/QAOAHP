import typing
import numpy as np
import serial # UART communication to FPGA
import time
import struct # convert Python numbers into bytes
from ...qaoa_simulator_base import Sim_Base, CostsType, ParamType, TermsType
from ...precomputation.numpy_vectorized import precompute_vectorized_cpu_parallel
from ....parameter_utils import generate_mixer_sincos_fpga

# the format of the data into the FPGa must match as: signed int64 little-endian Q3.61 fixed-point
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

    FIX_P = 64    # total bits
    FIX_N = 61    # fractional bits  -> Q3.61, range [-4, 4)

    # Operation codes from NEW_smachine.sv
    OP_NONE = 0  # 1 bytes
    OP_NONE8 = 0  # 8 byets
    OP_SEND1T = 1      # Send 1 byte from PC --> rT
    OP_SEND8T = 2      # Send 64-bit address as integer from PC --> rT
    OP_MOV_T2A = 3     # Move address from temp register (rT) to address
    OP_MOV_T2B = 4     # Move rT to rB
    OP_MOV_A2U = 5     # Move rA to rU (for output)
    OP_MOV_A2B = 6     # Move rA to rB
    OP_MOV_Info2U = 7
    OP_MOV_S2U = 8  
    OP_MOV_T2P = 9
    OP_FETCH1U = 60    # Fetch 1 byte from rU --> PC
    OP_FETCH8U = 61 
    OP_ADD_B2A = 80   # rA = rA +rB (64bit fixed, 2cycles)
    OP_MUL_B2A = 81   # rA = rA * rB (24bit fixed, 8 cycles)
    OP_ADDFP_B2A = 82 # rA = rA + rB (64bit float, 27 cycles)
    OP_MULFP_B2A = 83 # rA = rA * rB (64bit float, 24 cycles)
    OP_INC_A = 84      # Increment rA by 1 --> rA = rA +1
    OP_WRITE_T2RAM = 111  # Write rT to BRAM at address rA ---  rT to BRAM[rA]
    OP_READ_RAM2U = 112   # Read BRAM at address rA to rU -- BRAM[rA] → rU 
    OP_SEND_CMD = 118     # see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv
    OP_WRITE_T2_AG = 119
    # add gen_register selector
    AG_SET_t_L2Addr = 0; AG_SET_t_L2PipeGC=1; AG_SET_tb_B2GenCost=2
    AG_SET_t_L2Pipe=3; AG_SET_nPLayer=4;    AG_SET_L1Qbit=5
    AG_SET_AddrMask=6; AG_SET_t_B2GenCost=7; AG_SET_tb_B2Mixer=8
    AG_SET_t_L2Compute=9
    HOST_WAIT = 254

    # QAOA system commands (from qaoa_system.sv)
    qa_INIT = 16   # Initialize QAOA system
    qa_WAIT = 1   # Wait state
    qa_RUN = 2    # Run QAOA layer
    

    # BRAM address
    #BRAM_CONFIG     = 0x4000000000000000  # Config registers (NQ, NS, Np)
    BRAM_PARAMS = 0x0800_0000_0000_0000      # Parameters cos(β), sin(β), γ per layer
                  
    BRAM_STATE_REAL = 0x1000_0000_0000_0000      # BRAM[0]: Initial state real components 
    BRAM_STATE_IMAG = 0x2000_0000_0000_0000      # BRAM[1]: Initial state imaginary components (2^n values)
    BRAM_COST_FUNC  = 0x0400_0000_0000_0000       # BRAM[2]: Diagonal cost values _hc_diag (2^n values)
    #BRAM_CONFIG_STEP = 0x0100000000000000      # Address step for config registers




# helper function
    def __init__(self, fpga_config: dict , timeout=1):
        
        """
        Initialize FPGA driver with serial port settings
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
            baudrate: Communication speed (default 115200)
            timeout: Read timeout in seconds
        """
        port = fpga_config.get("port")
        baudrate = fpga_config.get("baudrate", 115200) 
        timeout = fpga_config.get("timeout", 1) 

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
                port= self.port,
                baudrate= self.baudrate,
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
        
# Higher level data formation
    def _write_bytes(self, data: bytes):
        """Write raw bytes to UART with the shared connection check."""
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        self.ser.write(data)
        time.sleep(2e-4)

    def _send_opcode(self, opcode):
        """Send single opcode byte"""
        self._write_bytes(bytes([opcode]))
        
        # in order to send pre-formatted bytes 
                       
    def _convert_byte(self, scaled, bytes, signed=True):
        return scaled.to_bytes(bytes, "little", signed=signed)
    
    def _float_to_fixed(self, v , f_bit):
        v = np.rint(np.asarray(v, dtype=np.float64) * (1 << f_bit)).astype(object)
        min_val = -(1 << (self.FIX_P -1))
        max_val =  (1 <<(self.FIX_P - 1))
        if v < min_val | v > max_val:
            raise ValueError("fixed-point overflow")
        return [int(x) for x in v]
    

    def _send_byte(self, value):
        """Send single byte (64 bit integer in little-endian format)"""
        self._write_bytes(bytes([value]))

    def _send_int64(self, value):
        """Send 64-bit integer in little-endian format"""
        data = int(value).to_bytes(8, byteorder='little', signed=False)
        self._write_bytes(data)

    def _send_fixed(self, v, name="value", sign=True):
        """Convert float -> Q3.61 -> 8 bytes two's-complement LE, send over UART."""
        self._write_bytes(self._convert_byte(v, 8, signed=sign))

    def _compute_timing(self, NQ, Np):
        LP_BRAM_A, LP_BRAM_D, LP_GEN_COST = 2, 1, 2
        LP_MIXER_IN, LP_MIXER_OUT = 1, 1
        L_BRAM_R = L_BRAM_W = LP_BRAM_A + LP_BRAM_D + 2
        N3 = 1+10+2+1+1+1
        gcN0, gcN1 = 10, 170                       # test_mul / CORDIC latencies — VERIFY vs HDL
        # I update this based on new method need to check again 
        gcPipe = 1 + gcN0 + 1 + gcN0 + 1 + 1 + gcN1 + 1   # in, mul1, fx, mul2, slicer, cordic_in, cordic, out
        Lc = 1 + gcPipe + 1 + L_BRAM_R + 2*LP_GEN_COST
        Lm = 1 + N3 + 1 + L_BRAM_R + L_BRAM_W + 1 + LP_MIXER_IN + LP_MIXER_OUT
        LInit = 24
        NS = 1 << NQ
        LPipe = max(NS, Lm + NS//2 + NS%2)
        DVTc = Lc // LPipe + 1
        tGenCost = DVTc*LPipe - Lc
        if tGenCost < LInit: tGenCost += LPipe
        tbGenCost = LPipe*(NQ+1) - Lc
        t_Mixer = LPipe
        if tbGenCost < LInit:
            t_Mixer = LPipe + LInit - tbGenCost; tbGenCost = LInit
        t_Compute = t_Mixer + LPipe*NQ + Lm
        mask = (1 << (NQ-1)) - 1 if NQ >= 1 else 0
        return {
            self.AG_SET_t_L2Addr:    NS-2,      self.AG_SET_t_L2Pipe:   LPipe-2,
            self.AG_SET_t_L2PipeGC:  Lc-2,      self.AG_SET_tb_B2GenCost: tbGenCost-2,
            self.AG_SET_t_B2GenCost: tGenCost-2, self.AG_SET_nPLayer:    Np,
            self.AG_SET_L1Qbit:      NQ-1,      self.AG_SET_AddrMask:   mask,
            self.AG_SET_tb_B2Mixer:  t_Mixer-2, self.AG_SET_t_L2Compute: t_Compute,
        }

    def _program_addr_gen(self, NQ, Np):
        for sel, val in self._compute_timing(NQ, Np).items():
            self._send_opcode(self.OP_SEND1T); self._send_byte(sel)
            self._send_opcode(self.OP_MOV_T2A)
            self._send_opcode(self.OP_SEND8T); self._send_int64(val & ((1<<64)-1))
            self._send_opcode(self.OP_WRITE_T2_AG)


    def _fetch_fx64(self):
        if not self.connected or self.ser is None:
            raise RuntimeError("Not connected to FPGA")
        d = self.ser.read(8)
        if len(d) != 8:
            raise RuntimeError(f"Expected 8 bytes from FPGA, got {len(d)}")
        v = int.from_bytes(d, "little", signed=True)
        return v / float(1 << self.FIX_N) # the two's-complement adjustment for signed Q3.61
    
    def _wait_for_fpga(self, timeout=1000):
        if self.ser is None:
            raise RuntimeError("Serial connection not established")
        for _ in range(timeout):
            self._send_opcode(self.OP_MOV_S2U)
            self._send_opcode(self.OP_FETCH1U)
            
            dr = self.ser.read(1)
            if dr == bytes([self.qa_WAIT]): return True
            time.sleep(0.001)
        return False

    def load_data(self, diag_hamiltonian, sv0_real, sv0_imag, gammas, cosb, sinb):
        """
        Load data to FPGA
        Args:
            diag_hamiltonian: Cost diagonal array (length 2^n_qubits)
            initial_state: Initial quantum state (complex128, length 2^n_qubits)
            gammas: Gamma parameters for each layer
            betas: Beta parameters for each layer
        """
        if not self.connected:
            raise RuntimeError("Not connected to FPGA")        
        print("Loading data to FPGA...")
        n_states = len(diag_hamiltonian)
        p = len(gammas)
        n_qubits = int(np.log2(n_states)) # numbert of qubits

        if len(sv0_real) != n_states:
            raise ValueError("sv0_real length does not match diag_hamiltonian")
        if len(sv0_imag) != n_states:
            raise ValueError("sv0_real length does not match diag_hamiltonian")
        if len(cosb) != p or len(sinb) != p:
            raise ValueError("cosb/sinb length must match number of gamma layers")
        try:
            # 1. Initialize QAOA system
            # --- Phase 1: Reset cycle 
            print("  Phase 1: Reset cycle...")
            self._send_opcode(self.OP_SEND1T); self._send_byte(self.qa_WAIT)
            self._send_opcode(self.OP_SEND_CMD)

            # 2) Program addr_gen timing registers
            self._program_addr_gen(n_qubits, p)

        # 3) Params: p+1 triples, redundant first cos/sin, trailing gamma=-1
            print("  Phase 3: Load parameters...")
            cosb_w = [float(cosb[0])] + [float(c) for c in cosb]
            sinb_w = [float(sinb[0])] + [float(s) for s in sinb]
            gam_w  = [float(g) for g in gammas] + [-1.0]
            # convert into fixed point
            fix_gamma_w = self._float_to_fixed(gam_w, f_bit=58) # Q6.58
            fix_cosb_w = self._float_to_fixed(cosb_w, f_bit=61) # Q3.61
            fix_sinb_w = self._float_to_fixed(sinb_w, f_bit=61) # Q3.61
            fix_sv0_r =  self._float_to_fixed(sv0_real, f_bit=61) # Q3.61
            fix_sv0_i = self._float_to_fixed(sv0_imag, f_bit=61) # Q3.61
            fix_H = self._float_to_fixed(diag_hamiltonian, f_bit=59) # Q5.59

            self._send_opcode(self.OP_SEND8T); self._send_int64(self.BRAM_PARAMS) # send address for parameter block
            self._send_opcode(self.OP_MOV_T2A)
            for p_L in range (p +1): # send actual parameters
                for value in (fix_cosb_w[p_L], fix_sinb_w[p_L], fix_gamma_w[p_L]):
                    self._send_opcode(self.OP_SEND8T); self._send_fixed(value)
                    self._send_opcode(self.OP_WRITE_T2RAM); self._send_opcode(self.OP_INC_A)

          # 4) State real / 5) State imag / 6) Cost values
            print("  Phase 4: Load state and cost data...")
            for address, value in ((self.BRAM_STATE_REAL, fix_sv0_r), 
                                   (self.BRAM_STATE_IMAG, fix_sv0_i),
                                   (self.BRAM_COST_FUNC,  fix_H)):
                self._send_opcode(self.OP_SEND8T); self._send_int64(address)
                self._send_opcode(self.OP_MOV_T2A)
                for i in range(n_states):
                    self._send_opcode(self.OP_SEND8T); self._send_fixed(int(value[i]))
                    self._send_opcode(self.OP_WRITE_T2RAM); self._send_opcode(self.OP_INC_A)
            print("✓ Data loaded to FPGA")
            return True
        
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
        
    def execute(self, p):
            if not self.connected:
                raise RuntimeError("Not connected to FPGA")
            try:
                self._send_opcode(self.OP_SEND1T); self._send_byte(self.qa_RUN)
                self._send_opcode(self.OP_SEND_CMD)
                done = self._wait_for_fpga(timeout=1000)

                if not done:
                    print("✗ FPGA did not signal completion")
                    return False
                
                self._send_opcode(self.OP_SEND1T); self._send_byte(self.qa_WAIT)
                self._send_opcode(self.OP_SEND_CMD)
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
            
            # Read real parts sequentially (set address once, then INC)
            self._send_opcode(self.OP_SEND8T)
            self._send_int64(self.BRAM_STATE_REAL)
            self._send_opcode(self.OP_MOV_T2A)
            for i in range(n_states):
                self._send_opcode(self.OP_READ_RAM2U)
                self._send_opcode(self.OP_FETCH8U)
                result[i] = self._fetch_fx64()  # real part only for now
                self._send_opcode(self.OP_INC_A)
            
            # Read imaginary parts sequentially
            self._send_opcode(self.OP_SEND8T)
            self._send_int64(self.BRAM_STATE_IMAG)
            self._send_opcode(self.OP_MOV_T2A)
            for i in range(n_states):
                self._send_opcode(self.OP_READ_RAM2U)
                self._send_opcode(self.OP_FETCH8U)
                imag_part = self._fetch_fx64()
                result[i] = complex(result[i].real, imag_part)
                self._send_opcode(self.OP_INC_A)
            
            print(f"✓ Read {n_states} amplitudes")
            return result
            
        except Exception as e:
            print(f"✗ Error reading result: {e}")
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
        if fpga_config is None:
            raise ValueError("fpga_config is not provided. Must be a dictionar with port, baudrate, max_qubits")
        self.fpga_config = fpga_config
        
        # Validate qubit count
        if n_qubits > self.fpga_config['max_qubits']:
            raise ValueError(f"Number of qubits ({n_qubits}) exceeds FPGA maximum {self.fpga_config['max_qubits']} "   )
        
        # Initialize driver - create instance of FPGA Driver
        self.fpga = FpgaDriver(fpga_config=self.fpga_config)
        self.connected = False

    def _diag_from_terms(self, terms: TermsType) -> np.ndarray:
        """
        np.ndarray
            Diagonal elements of the cost Hamiltonian
            """
        a = precompute_vectorized_cpu_parallel(terms, 0.0, self.n_qubits)
        return a
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
    
    def _scale_H_gamma(self, H, gammas):
        H = np.asarray(H, dtype=np.float64) # H in range ±10
        gammas = np.asarray(gammas, dtype=np.float64)

        S = np.max(np.abs(H))
        if S == 0:
            return H.copy(), gammas.copy(), 1.0
        H_scaled = H / np.sqrt(2.0 * S)
        gamma_scaled = gammas * np.sqrt(S) / (np.pi * np.sqrt(2.0))

        return H_scaled, gamma_scaled, S
 
    
    def simulate_qaoa(
        self,
        gammas: ParamType,
        betas: ParamType,
        sv0: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:

#       # chekc len of gamma and beta
        if len(gammas) != len(betas):
            raise ValueError(f"Parameter mismatch: {len(gammas)} gammas vs {len(betas)} betas")
        
        # Initialize state - 
        # Important Note: new gamma must be in format fixed signed Q6.58 and new_H must be Q5.59 . scaled by 2⁵⁸ and 2⁵⁹ respectively to send in FPGA. 
        p = len(gammas)
        if sv0 is None:
            sv0 = np.ones(self.n_states, dtype=np.complex128) / np.sqrt(self.n_states)
        assert sv0 is not None, "sv0 must not be None after initialization"
        sv0_re = []
        sv0_im = []
        for i in range(self.n_states):
            sv0_re.append(sv0[i].real)
            sv0_im.append(sv0[i].imag)

        cosb_0, sinb_0 = generate_mixer_sincos_fpga(betas, p)
        assert len(cosb_0) == p and len(sinb_0) == p, "cosb and sinb must have length p"

        # normalise _hc_diag to fit sign Q5.59 and  new_gamma is in Q6.58]
        H_scaled , gamma_scaled, S = self._scale_H_gamma(self._hc_diag, gammas) # H_send in [-1,1]

     # sv0 , gamma, bet, and hc_diag are avalilable here in list of array format
        try:
            # Connect to FPGA
            if not self.connected:
                self.connected = self.fpga.connect()
                if not self.connected:
                    raise RuntimeError("Failed to connect to FPGA")
            # Load data
            success = self.fpga.load_data(H_scaled, sv0_re, sv0_im, gamma_scaled, cosb_0, sinb_0)
            if not success:
                raise RuntimeError("Failed to load data to FPGA")            
            # Execute
            success = self.fpga.execute(p)
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
    


