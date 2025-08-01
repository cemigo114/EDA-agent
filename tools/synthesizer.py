"""
Synthesis Tool Integration

Provides interfaces to various synthesis tools including Yosys (open-source)
and vendor-specific tools for RTL synthesis and optimization.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import logging
import time
import json


@dataclass
class SynthesisConfig:
    """Configuration for synthesis execution"""
    synthesizer: str = "yosys"           # yosys, synplify, vivado, quartus
    target_technology: str = "generic"   # generic, xilinx, altera, asic
    optimization_level: str = "medium"   # low, medium, high, aggressive
    clock_period: float = 10.0          # Target clock period in ns
    area_constraint: Optional[float] = None  # Maximum area constraint
    power_constraint: Optional[float] = None # Maximum power constraint
    synthesis_options: List[str] = None
    timeout_seconds: int = 300
    generate_reports: bool = True
    
    def __post_init__(self):
        if self.synthesis_options is None:
            self.synthesis_options = []


@dataclass
class SynthesisResult:
    """Result of synthesis execution"""
    success: bool
    return_code: int
    synthesis_time: float
    stdout: str
    stderr: str
    netlist_file: Optional[str] = None
    report_file: Optional[str] = None
    timing_report: Optional[str] = None
    area_report: Optional[str] = None
    power_report: Optional[str] = None
    # Synthesis metrics
    gate_count: Optional[int] = None
    flip_flop_count: Optional[int] = None
    lut_count: Optional[int] = None
    max_frequency: Optional[float] = None
    total_power: Optional[float] = None


class SynthesizerInterface(ABC):
    """Abstract interface for synthesis tools"""
    
    def __init__(self, config: SynthesisConfig = None):
        self.config = config or SynthesisConfig()
        self.logger = logging.getLogger(f"Synthesizer.{self.__class__.__name__}")
    
    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the synthesizer is available on the system"""
        pass
    
    @abstractmethod
    def synthesize(self, source_files: List[str], top_module: str, work_dir: str) -> Tuple[bool, str, str]:
        """Synthesize RTL to netlist"""
        pass
    
    @abstractmethod
    def generate_reports(self, work_dir: str) -> Dict[str, str]:
        """Generate synthesis reports (timing, area, power)"""
        pass
    
    def run_synthesis(self, source_files: List[str], top_module: str, 
                     work_dir: str = None) -> SynthesisResult:
        """Complete synthesis flow: synthesize + generate reports"""
        
        start_time = time.time()
        
        # Create working directory if not provided
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="synth_")
            cleanup_work_dir = True
        else:
            cleanup_work_dir = False
            
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Synthesis phase
            synth_success, synth_stdout, synth_stderr = self.synthesize(source_files, top_module, work_dir)
            synth_time = time.time() - start_time
            
            if not synth_success:
                return SynthesisResult(
                    success=False,
                    return_code=1,
                    synthesis_time=synth_time,
                    stdout=synth_stdout,
                    stderr=synth_stderr
                )
            
            # Generate reports if requested
            reports = {}
            if self.config.generate_reports:
                try:
                    reports = self.generate_reports(work_dir)
                except Exception as e:
                    self.logger.warning(f"Failed to generate reports: {e}")
            
            # Extract synthesis metrics
            metrics = self._extract_metrics(synth_stdout, reports)
            
            # Look for generated files
            netlist_file = self._find_netlist_file(work_path)
            
            return SynthesisResult(
                success=synth_success,
                return_code=0,
                synthesis_time=synth_time,
                stdout=synth_stdout,
                stderr=synth_stderr,
                netlist_file=netlist_file,
                report_file=reports.get("main_report"),
                timing_report=reports.get("timing_report"),
                area_report=reports.get("area_report"),
                power_report=reports.get("power_report"),
                gate_count=metrics.get("gate_count"),
                flip_flop_count=metrics.get("flip_flop_count"),
                lut_count=metrics.get("lut_count"),
                max_frequency=metrics.get("max_frequency"),
                total_power=metrics.get("total_power")
            )
            
        except Exception as e:
            self.logger.error(f"Synthesis failed with exception: {e}")
            return SynthesisResult(
                success=False,
                return_code=-1,
                synthesis_time=time.time() - start_time,
                stdout="",
                stderr=f"Synthesis exception: {str(e)}"
            )
        
        finally:
            if cleanup_work_dir:
                import shutil
                try:
                    shutil.rmtree(work_dir)
                except:
                    pass
    
    def _find_netlist_file(self, work_path: Path) -> Optional[str]:
        """Find the generated netlist file"""
        # Common netlist file extensions
        for pattern in ["*.v", "*.vh", "*.sv", "*.net", "*.edif"]:
            candidates = list(work_path.glob(pattern))
            if candidates:
                return str(candidates[0])
        return None
    
    def _extract_metrics(self, stdout: str, reports: Dict[str, str]) -> Dict[str, Any]:
        """Extract synthesis metrics from output and reports"""
        metrics = {}
        
        # Try to extract metrics from stdout
        stdout_lower = stdout.lower()
        
        # Gate count extraction patterns
        import re
        gate_match = re.search(r'(\d+)\s+gates?', stdout_lower)
        if gate_match:
            metrics["gate_count"] = int(gate_match.group(1))
        
        # Flip-flop count
        ff_match = re.search(r'(\d+)\s+flip.?flops?', stdout_lower)
        if ff_match:
            metrics["flip_flop_count"] = int(ff_match.group(1))
        
        # LUT count (for FPGA)
        lut_match = re.search(r'(\d+)\s+luts?', stdout_lower)
        if lut_match:
            metrics["lut_count"] = int(lut_match.group(1))
        
        # Frequency
        freq_match = re.search(r'(\d+\.?\d*)\s*mhz', stdout_lower)
        if freq_match:
            metrics["max_frequency"] = float(freq_match.group(1))
        
        # Extract from reports if available
        if "timing_report" in reports:
            timing_metrics = self._parse_timing_report(reports["timing_report"])
            metrics.update(timing_metrics)
        
        return metrics
    
    def _parse_timing_report(self, report_path: str) -> Dict[str, Any]:
        """Parse timing report for additional metrics"""
        metrics = {}
        try:
            with open(report_path, 'r') as f:
                content = f.read().lower()
                
            # Extract timing metrics
            import re
            slack_match = re.search(r'worst.*slack:\s*([+-]?\d+\.?\d*)', content)
            if slack_match:
                metrics["worst_slack"] = float(slack_match.group(1))
                
        except Exception as e:
            self.logger.warning(f"Failed to parse timing report: {e}")
            
        return metrics


class YosysSynthesizer(SynthesizerInterface):
    """Yosys open-source synthesizer interface"""
    
    def check_availability(self) -> bool:
        """Check if Yosys is available"""
        try:
            result = subprocess.run(['yosys', '-V'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def synthesize(self, source_files: List[str], top_module: str, work_dir: str) -> Tuple[bool, str, str]:
        """Synthesize with Yosys"""
        
        work_path = Path(work_dir)
        script_file = work_path / "synth_script.ys"
        netlist_file = work_path / f"{top_module}_netlist.v"
        
        # Generate Yosys synthesis script
        script_content = self._generate_yosys_script(source_files, top_module, str(netlist_file))
        
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Run Yosys
        cmd = ['yosys', '-s', str(script_file)]
        
        self.logger.info(f"Running Yosys synthesis: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            success = result.returncode == 0 and netlist_file.exists()
            
            if success:
                self.logger.info("Yosys synthesis completed successfully")
            else:
                self.logger.warning(f"Yosys synthesis failed with return code {result.returncode}")
            
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", "Yosys synthesis timeout"
        except Exception as e:
            return False, "", f"Yosys synthesis error: {str(e)}"
    
    def generate_reports(self, work_dir: str) -> Dict[str, str]:
        """Generate Yosys synthesis reports"""
        work_path = Path(work_dir)
        reports = {}
        
        # Main synthesis report
        report_file = work_path / "synthesis_report.txt"
        if report_file.exists():
            reports["main_report"] = str(report_file)
        
        # Yosys generates statistics in the log, so we can create a summary
        stat_file = work_path / "synthesis_stats.json"
        try:
            stats = self._extract_yosys_stats(work_dir)
            with open(stat_file, 'w') as f:
                json.dump(stats, f, indent=2)
            reports["stats_report"] = str(stat_file)
        except Exception as e:
            self.logger.warning(f"Failed to create stats report: {e}")
        
        return reports
    
    def _generate_yosys_script(self, source_files: List[str], top_module: str, netlist_file: str) -> str:
        """Generate Yosys synthesis script"""
        
        script_lines = [
            "# Yosys synthesis script",
            "",
            "# Read source files"
        ]
        
        # Add read commands for source files
        for src_file in source_files:
            if src_file.endswith('.sv') or 'systemverilog' in src_file.lower():
                script_lines.append(f"read_verilog -sv {src_file}")
            else:
                script_lines.append(f"read_verilog {src_file}")
        
        script_lines.extend([
            "",
            f"# Set top module",
            f"hierarchy -top {top_module}",
            "",
            "# Synthesis commands based on target technology"
        ])
        
        # Technology-specific synthesis
        if self.config.target_technology == "xilinx":
            script_lines.extend([
                "synth_xilinx -top {top_module}",
                "# FPGA-specific optimizations",
                "opt -full"
            ])
        elif self.config.target_technology == "altera":
            script_lines.extend([
                "synth_intel -top {top_module}",
                "opt -full"
            ])
        else:
            # Generic synthesis
            script_lines.extend([
                "# Generic synthesis flow",
                "proc; opt; memory; opt",
                "techmap; opt",
                "abc -liberty /dev/null" if self.config.optimization_level == "high" else "abc"
            ])
        
        # Optimization level adjustments
        if self.config.optimization_level == "aggressive":
            script_lines.extend([
                "opt -full",
                "clean -purge"
            ])
        elif self.config.optimization_level in ["medium", "high"]:
            script_lines.append("opt")
        
        script_lines.extend([
            "",
            "# Generate statistics",
            "stat",
            "",
            f"# Write netlist",
            f"write_verilog {netlist_file}",
            "",
            "# Exit",
            "exit"
        ])
        
        return "\n".join(script_lines)
    
    def _extract_yosys_stats(self, work_dir: str) -> Dict[str, Any]:
        """Extract statistics from Yosys output"""
        stats = {}
        
        # Look for Yosys log files
        work_path = Path(work_dir)
        log_files = list(work_path.glob("*.log")) + list(work_path.glob("yosys.out"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Extract statistics from Yosys stat command output
                import re
                
                # Number of wires
                wire_match = re.search(r'Number of wires:\s+(\d+)', content)
                if wire_match:
                    stats["wire_count"] = int(wire_match.group(1))
                
                # Number of cells
                cell_match = re.search(r'Number of cells:\s+(\d+)', content)
                if cell_match:
                    stats["cell_count"] = int(cell_match.group(1))
                
                # Look for specific cell types
                ff_match = re.search(r'\$dff\s+(\d+)', content)
                if ff_match:
                    stats["flip_flop_count"] = int(ff_match.group(1))
                
                break  # Use first log file found
                
            except Exception as e:
                self.logger.warning(f"Failed to parse log file {log_file}: {e}")
        
        return stats


def get_available_synthesizers() -> List[str]:
    """Get list of available synthesizers on the system"""
    available = []
    
    synthesizers = {
        "yosys": YosysSynthesizer()
    }
    
    for name, synthesizer in synthesizers.items():
        if synthesizer.check_availability():
            available.append(name)
    
    return available


def create_synthesizer(synthesizer_name: str, config: SynthesisConfig = None) -> SynthesizerInterface:
    """Factory function to create synthesizer instances"""
    
    config = config or SynthesisConfig()
    
    if synthesizer_name.lower() == "yosys":
        return YosysSynthesizer(config)
    else:
        raise ValueError(f"Unknown synthesizer: {synthesizer_name}")


def auto_select_synthesizer(config: SynthesisConfig = None) -> Optional[SynthesizerInterface]:
    """Automatically select the best available synthesizer"""
    
    available = get_available_synthesizers()
    
    if not available:
        return None
    
    # Preference order: Yosys (only one implemented for now)
    preference_order = ["yosys"]
    
    for preferred in preference_order:
        if preferred in available:
            return create_synthesizer(preferred, config)
    
    # Fallback to first available
    return create_synthesizer(available[0], config)