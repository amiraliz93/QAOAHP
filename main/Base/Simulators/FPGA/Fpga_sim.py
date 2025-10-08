import typing
import numpy as np
from typing import Sequence
import serial
import time
from ..FPGA import Sim_Base
from Sim_Base import CostsType, ParamType, TermsType
# Mock FPGA interface (to be replaced with actual FPGA driver)
class FpgaDriver:

    def __init__(self, port="COM3", baudrate=115200, timeout=1):
        """
    Initialise FPGA driver with serial port setting
    
    This class interfaces with FPGA hardware to accelerate QAOA simulations by offloading
    the quantum state evolution to specialized hardware.

    Args:
    port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
    baudrate: communication speed (must match FPGA settigs)
    timeout: Read timeout in seconds
    """ 
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False
        
    def connect(self):
        """ connect to FPGA via UART"""
        print("Connecting to FPGA board...")
        try:
            self.ser = serial.Serial(
                port = self.port,
                baudrate = self.baudrate,
                bytesize = serial.EIGHTBITS,
                parity= serial.PARITY_NONE,
                stopbits= serial.STOPBITS_ONE,
                timeout = self.timeout
            )

            # clear any existing data in the buffers
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        
            # simple handshake to verify connection - do this directly
            # instead of using _send_command which checks self.connected
            cmd_bytes = ("Hellow FPGA\n").encode('ascii')
            self.ser.write(cmd_bytes)
            self.ser.flush()
            
            # Read response directly too
            response = self.ser.readline().decode('ascii').strip()
            
            if response == "READY":
                self.connected = True  # Now set this flag
                print(f"FPGA connected and ready on {self.port}")
                return True
            else:
                print(f"FPGA connection failed, got {response}")
                self.ser.close()
                self.connected = False
                return False
            
        except Exception as e:
            print(f"Error connection is {e}")
            self.connected = False
            return False


    def _read_response(self):
        """Read response from FPGA"""
        if not self.connected:
            raise RuntimeError("Not connected to FPGA")
            
        response = self.ser.readline().decode('ascii').strip()
        return response

    def _send_command(self, cmd):
        """send command string to FPGA board"""
        if not self.connected:
            raise RuntimeError("Not connected to FPGA")
        
        cmd_bytes = (cmd + '\n').encode('ascii')
        self.ser.write(cmd_bytes)
        self.ser.flush()

        # wait for akcnowledgement
        response = self._read_response()
        if response != "ACK":
            raise RuntimeError(f"FPGA did not acknowledge command '{cmd}', got '{response}'")
        
    def _send_data(self, data_array):
        """send data to fpga board as binary data
        
        First send the lenght aas 4-byte int 
         then send the array data as binary """
        
        if not self.connected: 
            raise RuntimeError("Not connected to the fpga (_send_data function)")
        
        # convert array to binary
        data_bytes = data_array.tobytes()

        # send data lenght first 
        lenght = len(data_bytes)
        lenght_bytes = lenght.to_bytes(4, byteorder="little")
        self.ser.write(lenght_bytes)

        # send actual data
        self.ser.write(data_bytes)
        self.ser.flush()


        # wait for acknowledgment
        response = self._read_response()
        if response != "ACK":
            raise RuntimeError(f"FPGA data send error {response}")

    def _read_data(self, expected_size):
        """Read binary data into numpy array"""
        if not self.connected:
            raise RuntimeError("Not connected to FPGA")
            
        # Read binary data
        data_bytes = self.ser.read(expected_size * 8)  # 8 bytes per complex number (2 floats)
        
        if len(data_bytes) != expected_size * 8:
            raise RuntimeError(f"Expected {expected_size * 8} bytes but got {len(data_bytes)}")
            
        # Convert to numpy array
        return np.frombuffer(data_bytes, dtype=np.complex64)
    

    def load_data(self, diag_hamiltonian, initial_state, theta):
        """ send problem data to fpga
        
        Args:
          diag_hamiltonian: Cost diagonal array
          initial_state: Initial quantum state 2**N size
          """
        if not self.connected:
            return False


        try:
            #send command to prepare data
            self._send_command("LOAD_DATA")
            response = self._read_response()
            if response != "READY":
                raise RuntimeError(f"FPGA not ready : {response}")
            
            # send the diogonal hailtonian
            self._send_data(diag_hamiltonian.astype(np.complex64))

            # send the initial state
            self._send_data(initial_state.astype(np.complex64))
            # send theta parameters
            self._send_data(theta.astype(np.float64))

            print("Data loaded to FPGA")
            return True
        except Exception as e:
            print(f"Error loading data to fpga is {e}")
  
        
    def execute(self):
        print("Executing QAOA simulation on FPGA...")
        if not self.connected:
            return False
      
        
        try:
            # send excecute command
            self._send_command("EXCECUTE")

            # waite for completion
            response = self._read_response()
            if response != "COMPLETE":
                raise RuntimeError(f"fpga excecution error: {response}")
            print(" QAOA simulation complete on FPGA")
            return True
        
        except Exception as e:
            print(f"Error during FPGA excution: {e}")
            return False
          

    def read_result(self, n_states):
        """read result which is updated statevectore from fpga

        Args:
            n_states (_type_): number of qubits

        Returns:
            numpy array with final statevector
        """
        print(f"Reading {n_states} state amplitudes from FPGA...")
        if not self.connected:
            return np.ones(n_states, dtype=np.complex64) / np.sqrt(n_states)
        
        try:
            #send read comand
            self._send_command("read_data")
            response = self._read_response()
            if response != "SENDING":
                raise RuntimeError(f" fpga result read error: {response}")
            # read binary data
            result = self._read_data(n_states)
            print(f" read {n_states} amplitute from fpga")
            return result
        
        except Exception as e:
            print(f"Error reading result from FPGA: {str(e)}")
            # Return a default result in case of error
            return np.ones(n_states, dtype=np.complex64) / np.sqrt(n_states)


    def disconnect(self):
        print("Disconnecting from FPGA board")
        if self.connected and self.ser:
            try:
                self._send_command("GOODBYE")
                self.ser.close()
            except:
                pass
        self.connected = False
        print("DInsconnected from FPGA bpard")
        return True

class FPGASimulator(SimulationBase):

    _hc_diag: np.ndarray

    def __init__(
        self,
        n_qubits: int,
        costs: CostsType | None = None,
        terms: TermsType | None = None,
        fpga_config: dict | None = None,
    ) -> None:
        """
        Initialize the FPGA-based QAOA simulator.
        
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the system
        costs : CostsType | None
            Precomputed cost function values for all basis states
        terms : TermsType | None
            List of Hamiltonian terms as (coefficient, [qubit_indices])
        fpga_config : dict, optional
            Configuration parameters for FPGA hardware:
            - 'device_id': FPGA device identifier
            - 'max_qubits': Maximum supported qubits
            - 'bitstream_path': Path to FPGA bitstream file
        """
        # Initialize base class first
        super().__init__(n_qubits=n_qubits, costs=costs, terms=terms)
        
        # Store FPGA configuration
        self.fpga_config = fpga_config or {}
        
        # Default configuration parameters
        default_config = {
            'device_id': 0,
            'max_qubits': 16,  # Typical max qubits for FPGA
            'bitstream_path': './bitstream/qaoa_default.bit',
        }
        
        # Apply defaults for missing configuration
        for key, value in default_config.items():
            if key not in self.fpga_config:
                self.fpga_config[key] = value
        
        # Validate qubit count against FPGA limitations
        if n_qubits > self.fpga_config['max_qubits']:
            raise ValueError(
                f"Number of qubits ({n_qubits}) exceeds maximum supported by FPGA "
                f"({self.fpga_config['max_qubits']})"
            )
        
        # Initialize FPGA driver
        self.fpga = FpgaDriver()
        self.connected = False
    
    def _connect_to_fpga(self):
        """Connect to the FPGA board if not already connected."""
        if not self.connected:
            self.connected = self.fpga.connect()
            if not self.connected:
                raise RuntimeError("Failed to connect to FPGA board")
    
    def _disconnect_from_fpga(self):
        """Disconnect from the FPGA board if connected."""
        if self.connected:
            self.fpga.disconnect()
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
        # Calculate the diagonal elements on the CPU
        costs = np.zeros(self.n_states)
        for coeff, qubits in terms:
            for i in range(self.n_states):
                # Convert i to binary representation
                binary = [(i >> j) & 1 for j in range(self.n_qubits)]
                # Convert 0/1 to +1/-1
                spins = [1 - 2 * binary[j] for j in qubits]
                # Calculate term value
                term_val = coeff * np.prod(spins)
                costs[i] += term_val
                
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
        # Simply convert to numpy array for FPGA compatibility
        return np.array(costs)
    
    def get_cost_diagonal(self) -> np.ndarray:
        """
        Return the diagonal of the cost Hamiltonian as a numpy array.
        
        Returns
        -------
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
        Simulate QAOA circuit using FPGA hardware.
        
        This method:
        1. Prepares initial state vector
        2. Formats parameters for FPGA
        3. Loads data to FPGA
        4. Executes QAOA simulation on FPGA
        5. Retrieves and returns result
        
        Parameters
        ----------
        gammas : ParamType
            Parameters for phase separation operators
        betas : ParamType
            Parameters for mixing operators
        sv0 : np.ndarray | None, optional
            Initial state vector, by default None (uniform superposition)
            
        Returns
        -------
        np.ndarray
            Final state vector after QAOA evolution
        """
        # Convert parameters to numpy arrays
        gammas_np = np.asarray(gammas)
        betas_np = np.asarray(betas)
        
        # Validate parameters
        if len(gammas_np) != len(betas_np):
            raise ValueError(f"Mismatch in parameter counts: {len(gammas_np)} gammas vs {len(betas_np)} betas")
        
        # Initialize state vector if not provided
        if sv0 is None:
            # Start with uniform superposition |+⟩^⊗n
            sv0 = np.ones(self.n_states, dtype=np.complex128) / np.sqrt(self.n_states)
        
        # Combine gammas and betas into theta for FPGA
        # Format: [gamma_0, beta_0, gamma_1, beta_1, ...]
        p = len(gammas_np)  # Number of QAOA layers
        theta = np.zeros(2 * p)
        theta[0::2] = gammas_np
        theta[1::2] = betas_np
        
        try:
            # Connect to FPGA
            self._connect_to_fpga()
            
            # Load data to FPGA
            success = self.fpga.load_data(
                diag_hamiltonian=self._hc_diag,
                initial_state=sv0,
                theta=theta
            )
            if not success:
                raise RuntimeError("Failed to load data to FPGA")
            
            # Execute QAOA simulation on FPGA
            success = self.fpga.execute()
            if not success:
                raise RuntimeError("FPGA execution failed")
            
            # Read result from FPGA
            result = self.fpga.read_result(self.n_states)
            
            return result
        
        except Exception as e:
            # Handle FPGA-specific errors
            raise RuntimeError(f"FPGA simulation error: {str(e)}")
        
        finally:
            # Ensure FPGA is disconnected properly
            self._disconnect_from_fpga()
    
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
        # Use provided costs or internal diagonal
        if costs is None:
            costs = self._hc_diag
            
        # Calculate probabilities of each computational basis state
        probs = np.abs(result) ** 2
        
        # Calculate expectation value
        expectation = np.sum(costs * probs)
        
        # Return negative expectation for maximization problems
        if optimization_type == "max":
            return -expectation
        return expectation
    
    def get_overlap(
        self, result, costs: CostsType | None = None, indices: np.ndarray | Sequence[int] | None = None, 
        optimization_type="min", **kwargs
    ) -> float:
        """
        Calculate overlap between the result and optimal states.
        
        Parameters
        ----------
        result : np.ndarray
            Final state vector from QAOA simulation
        costs : CostsType, optional
            Cost function values (defaults to internal diagonal)
        indices : np.ndarray | Sequence[int], optional
            Indices of optimal states (determined from costs if not provided)
        optimization_type : str, optional
            Type of optimization problem ("min" or "max"), by default "min"
            
        Returns
        -------
        float
            Overlap with optimal states (probability sum)
        """
        # Determine optimal states if indices not provided
        if indices is None:
            # Use provided costs or internal diagonal
            if costs is None:
                costs = self._hc_diag
                
            # Find optimal indices based on optimization type
            if optimization_type == "max":
                optimal_value = np.max(costs)
                optimal_indices = np.where(np.isclose(costs, optimal_value))[0]
            else:
                optimal_value = np.min(costs)
                optimal_indices = np.where(np.isclose(costs, optimal_value))[0]
        else:
            optimal_indices = indices
            
        # Calculate total probability of being in an optimal state
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
        # In this implementation, result is already the statevector
        return np.array(result)
    
    def get_probabilities(self, result: typing.Any, **kwargs) -> np.ndarray:
        """
        Return the probabilities of each computational basis state.
        
        Parameters
        ----------
        result : typing.Any
            Result from simulate_qaoa
            
        Returns
        -------
        np.ndarray
            Array of probabilities
        """
        # Calculate probabilities from statevector
        return np.abs(result) ** 2