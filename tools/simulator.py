"""
Simulation Tool Integration

Provides interfaces to various SystemVerilog/Verilog simulators
including Icarus Verilog, ModelSim, and Verilator.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import logging
import time
import os


@dataclass
class SimulationConfig:
    """Configuration for simulation execution"""
    simulator: str = "iverilog"        # iverilog, modelsim, verilator
    language_version: str = "2012"     # 1995, 2001, 2005, 2012
    compile_options: List[str] = None
    run_options: List[str] = None
    timeout_seconds: int = 120
    generate_vcd: bool = True
    optimization_level: int = 0        # 0=none, 1=basic, 2=aggressive
    
    def __post_init__(self):
        if self.compile_options is None:
            self.compile_options = []
        if self.run_options is None:
            self.run_options = []


@dataclass
class SimulationResult:
    """Result of simulation execution"""
    success: bool
    return_code: int
    compile_time: float
    run_time: float
    stdout: str
    stderr: str
    vcd_file: Optional[str] = None
    log_file: Optional[str] = None
    coverage_file: Optional[str] = None


class SimulatorInterface(ABC):
    """Abstract interface for EDA simulators"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.logger = logging.getLogger(f"Simulator.{self.__class__.__name__}")
    
    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the simulator is available on the system"""
        pass
    
    @abstractmethod
    def compile(self, source_files: List[str], work_dir: str) -> Tuple[bool, str, str]:
        """Compile SystemVerilog/Verilog source files"""
        pass
    
    @abstractmethod
    def simulate(self, work_dir: str, top_module: str = None) -> Tuple[bool, str, str]:
        """Run simulation"""
        pass
    
    def run_simulation(self, source_files: List[str], work_dir: str = None, 
                      top_module: str = None) -> SimulationResult:
        """Complete simulation flow: compile + simulate"""
        
        start_time = time.time()
        
        # Create working directory if not provided
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="sim_")
            cleanup_work_dir = True
        else:
            cleanup_work_dir = False
            
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Compilation phase
            compile_start = time.time()
            compile_success, compile_stdout, compile_stderr = self.compile(source_files, work_dir)
            compile_time = time.time() - compile_start
            
            if not compile_success:
                return SimulationResult(
                    success=False,
                    return_code=1,
                    compile_time=compile_time,
                    run_time=0.0,
                    stdout=compile_stdout,
                    stderr=compile_stderr
                )
            
            # Simulation phase
            sim_start = time.time()
            sim_success, sim_stdout, sim_stderr = self.simulate(work_dir, top_module)
            sim_time = time.time() - sim_start
            
            # Combine outputs
            combined_stdout = compile_stdout + "\\n" + sim_stdout
            combined_stderr = compile_stderr + "\\n" + sim_stderr
            
            # Look for generated files
            vcd_file = None
            if self.config.generate_vcd:
                vcd_candidates = list(work_path.glob("*.vcd"))
                if vcd_candidates:
                    vcd_file = str(vcd_candidates[0])
            
            log_file = None
            log_candidates = list(work_path.glob("*.log"))
            if log_candidates:
                log_file = str(log_candidates[0])
            
            return SimulationResult(
                success=sim_success,
                return_code=0 if sim_success else 1,
                compile_time=compile_time,
                run_time=sim_time,
                stdout=combined_stdout,
                stderr=combined_stderr,
                vcd_file=vcd_file,
                log_file=log_file
            )
            
        except Exception as e:
            self.logger.error(f"Simulation failed with exception: {e}")
            return SimulationResult(
                success=False,
                return_code=-1,
                compile_time=0.0,
                run_time=0.0,
                stdout="",
                stderr=f"Simulation exception: {str(e)}"
            )
        
        finally:
            if cleanup_work_dir:
                import shutil
                try:
                    shutil.rmtree(work_dir)
                except:
                    pass


class IcarusVerilogSimulator(SimulatorInterface):
    """Icarus Verilog (iverilog) simulator interface"""
    
    def check_availability(self) -> bool:
        """Check if iverilog is available"""
        try:
            result = subprocess.run(['iverilog', '-V'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def compile(self, source_files: List[str], work_dir: str) -> Tuple[bool, str, str]:
        """Compile with iverilog"""
        
        work_path = Path(work_dir)
        output_file = work_path / "simulation_output"
        
        # Build iverilog command
        cmd = ['iverilog']
        
        # Language version
        if self.config.language_version == "2012":
            cmd.extend(['-g2012'])
        elif self.config.language_version == "2005":
            cmd.extend(['-g2005-sv'])
        elif self.config.language_version == "2001":
            cmd.extend(['-g2001'])
        
        # Output file
        cmd.extend(['-o', str(output_file)])
        
        # Additional options
        cmd.extend(self.config.compile_options)
        
        # Source files
        cmd.extend(source_files)
        
        self.logger.info(f"Compiling with command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            success = result.returncode == 0
            
            if success:
                self.logger.info("Compilation successful")
            else:
                self.logger.warning(f"Compilation failed with return code {result.returncode}")
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error("Compilation timeout")
            return False, "", "Compilation timeout expired"
        except Exception as e:
            self.logger.error(f"Compilation error: {e}")
            return False, "", f"Compilation error: {str(e)}"
    
    def simulate(self, work_dir: str, top_module: str = None) -> Tuple[bool, str, str]:
        """Run simulation with vvp"""
        
        work_path = Path(work_dir)
        simulation_binary = work_path / "simulation_output"
        
        if not simulation_binary.exists():
            return False, "", "Simulation binary not found - compilation may have failed"
        
        # Build vvp command
        cmd = ['vvp']
        
        # VCD output
        if self.config.generate_vcd:
            cmd.extend(['-v'])  # Verbose mode for better VCD output
        
        # Additional run options
        cmd.extend(self.config.run_options)
        
        # Simulation binary
        cmd.append(str(simulation_binary))
        
        self.logger.info(f"Running simulation with command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            success = result.returncode == 0
            
            if success:
                self.logger.info("Simulation completed successfully")
            else:
                self.logger.warning(f"Simulation failed with return code {result.returncode}")
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.logger.error("Simulation timeout")
            return False, "", "Simulation timeout expired"
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            return False, "", f"Simulation error: {str(e)}"


class VerilatorSimulator(SimulatorInterface):
    """Verilator simulator interface (high-performance)"""
    
    def check_availability(self) -> bool:
        """Check if Verilator is available"""
        try:
            result = subprocess.run(['verilator', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def compile(self, source_files: List[str], work_dir: str) -> Tuple[bool, str, str]:
        """Compile with Verilator"""
        
        work_path = Path(work_dir)
        
        # Build verilator command
        cmd = ['verilator']
        
        # Language options
        cmd.extend(['--cc', '--exe'])  # Generate C++ executable
        
        if self.config.language_version == "2012":
            cmd.extend(['-sv'])  # SystemVerilog mode
        
        # Performance options
        if self.config.optimization_level >= 1:
            cmd.extend(['-O3'])
        
        # Generate VCD
        if self.config.generate_vcd:
            cmd.extend(['--trace'])
        
        # Additional options
        cmd.extend(self.config.compile_options)
        
        # Source files
        cmd.extend(source_files)
        
        # Create simple C++ main file for testbench
        main_cpp = work_path / "sim_main.cpp"
        with open(main_cpp, 'w') as f:
            f.write('''
#include <iostream>
#include <verilated.h>
#include <verilated_vcd_c.h>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    std::cout << "Verilator simulation completed" << std::endl;
    return 0;
}
''')
        
        cmd.append(str(main_cpp))
        
        self.logger.info(f"Compiling with Verilator: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            success = result.returncode == 0
            
            if success:
                # Build the generated C++ code
                make_cmd = ['make', '-C', 'obj_dir', '-f', 'Vtop.mk', 'Vtop']
                make_result = subprocess.run(
                    make_cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds
                )
                success = make_result.returncode == 0
                
                if success:
                    self.logger.info("Verilator compilation successful")
                else:
                    self.logger.warning("C++ compilation failed")
                    return False, result.stdout + make_result.stdout, result.stderr + make_result.stderr
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Verilator compilation timeout"
        except Exception as e:
            return False, "", f"Verilator compilation error: {str(e)}"
    
    def simulate(self, work_dir: str, top_module: str = None) -> Tuple[bool, str, str]:
        """Run Verilator simulation"""
        
        work_path = Path(work_dir)
        sim_binary = work_path / "obj_dir" / "Vtop"
        
        if not sim_binary.exists():
            return False, "", "Verilator binary not found"
        
        cmd = [str(sim_binary)]
        cmd.extend(self.config.run_options)
        
        self.logger.info(f"Running Verilator simulation: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Verilator simulation timeout"
        except Exception as e:
            return False, "", f"Verilator simulation error: {str(e)}"


class ModelSimSimulator(SimulatorInterface):
    """ModelSim/QuestaSim simulator interface"""
    
    def check_availability(self) -> bool:
        """Check if ModelSim is available"""
        try:
            # Try both vsim and qsim (QuestaSim)
            for sim_cmd in ['vsim', 'qsim']:
                result = subprocess.run([sim_cmd, '-version'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def compile(self, source_files: List[str], work_dir: str) -> Tuple[bool, str, str]:
        """Compile with vlog"""
        
        work_path = Path(work_dir)
        
        # Create work library
        vlib_cmd = ['vlib', 'work']
        vlib_result = subprocess.run(vlib_cmd, cwd=work_dir, capture_output=True, text=True)
        
        if vlib_result.returncode != 0:
            return False, vlib_result.stdout, vlib_result.stderr
        
        # Build vlog command
        cmd = ['vlog']
        
        # Language version
        if self.config.language_version == "2012":
            cmd.extend(['-sv', '-mfcu'])
        elif self.config.language_version == "2005":
            cmd.extend(['-sv'])
        
        # Additional options
        cmd.extend(self.config.compile_options)
        
        # Source files
        cmd.extend(source_files)
        
        self.logger.info(f"Compiling with ModelSim: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            success = result.returncode == 0
            combined_stdout = vlib_result.stdout + "\\n" + result.stdout
            combined_stderr = vlib_result.stderr + "\\n" + result.stderr
            
            return success, combined_stdout, combined_stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "ModelSim compilation timeout"
        except Exception as e:
            return False, "", f"ModelSim compilation error: {str(e)}"
    
    def simulate(self, work_dir: str, top_module: str = None) -> Tuple[bool, str, str]:
        """Run simulation with vsim"""
        
        if not top_module:
            top_module = "work.tb"  # Default testbench module
        
        # Build vsim command
        cmd = ['vsim', '-c']  # Command line mode
        
        # VCD generation
        if self.config.generate_vcd:
            cmd.extend(['-do', 'vcd file sim.vcd; vcd add -r /*; run -all; quit'])
        else:
            cmd.extend(['-do', 'run -all; quit'])
        
        # Additional options
        cmd.extend(self.config.run_options)
        
        # Top module
        cmd.append(top_module)
        
        self.logger.info(f"Running ModelSim simulation: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            # ModelSim may return non-zero even on successful simulation
            success = "# Simulation complete" in result.stdout or result.returncode == 0
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "ModelSim simulation timeout"
        except Exception as e:
            return False, "", f"ModelSim simulation error: {str(e)}"


def get_available_simulators() -> List[str]:
    """Get list of available simulators on the system"""
    available = []
    
    simulators = {
        "iverilog": IcarusVerilogSimulator(),
        "verilator": VerilatorSimulator(),
        "modelsim": ModelSimSimulator()
    }
    
    for name, simulator in simulators.items():
        if simulator.check_availability():
            available.append(name)
    
    return available


def create_simulator(simulator_name: str, config: SimulationConfig = None) -> SimulatorInterface:
    """Factory function to create simulator instances"""
    
    config = config or SimulationConfig()
    
    if simulator_name.lower() == "iverilog":
        return IcarusVerilogSimulator(config)
    elif simulator_name.lower() == "verilator":
        return VerilatorSimulator(config)
    elif simulator_name.lower() in ["modelsim", "questasim"]:
        return ModelSimSimulator(config)
    else:
        raise ValueError(f"Unknown simulator: {simulator_name}")


def auto_select_simulator(config: SimulationConfig = None) -> Optional[SimulatorInterface]:
    """Automatically select the best available simulator"""
    
    available = get_available_simulators()
    
    if not available:
        return None
    
    # Preference order: ModelSim > Verilator > Icarus Verilog
    preference_order = ["modelsim", "verilator", "iverilog"]
    
    for preferred in preference_order:
        if preferred in available:
            return create_simulator(preferred, config)
    
    # Fallback to first available
    return create_simulator(available[0], config)