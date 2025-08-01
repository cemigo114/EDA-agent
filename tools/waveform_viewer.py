"""
Waveform Viewer Integration

Provides interfaces to waveform viewing tools like GTKWave for debugging
and visualization of simulation results.
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
class WaveformConfig:
    """Configuration for waveform viewing"""
    viewer: str = "gtkwave"           # gtkwave, modelsim, vivado
    vcd_file: Optional[str] = None    # Path to VCD file
    save_file: Optional[str] = None   # Save file for viewer configuration
    signal_groups: List[str] = None   # Predefined signal groups
    time_range: Optional[Tuple[float, float]] = None  # Start, end time
    auto_zoom: bool = True            # Auto-zoom to fit signals
    show_hierarchy: bool = True       # Show module hierarchy
    background_mode: bool = False     # Run in background
    
    def __post_init__(self):
        if self.signal_groups is None:
            self.signal_groups = []


@dataclass
class WaveformSession:
    """Waveform viewing session information"""
    session_id: str
    viewer_process: Optional[subprocess.Popen] = None
    vcd_file: Optional[str] = None
    save_file: Optional[str] = None
    pid: Optional[int] = None
    active: bool = False


class WaveformViewer(ABC):
    """Abstract interface for waveform viewers"""
    
    def __init__(self, config: WaveformConfig = None):
        self.config = config or WaveformConfig()
        self.logger = logging.getLogger(f"WaveformViewer.{self.__class__.__name__}")
        self.active_sessions: Dict[str, WaveformSession] = {}
    
    @abstractmethod
    def check_availability(self) -> bool:
        """Check if the waveform viewer is available on the system"""
        pass
    
    @abstractmethod
    def launch_viewer(self, vcd_file: str, save_file: str = None) -> WaveformSession:
        """Launch waveform viewer with VCD file"""
        pass
    
    @abstractmethod
    def generate_save_file(self, vcd_file: str, output_file: str) -> bool:
        """Generate viewer save file with optimal signal grouping"""
        pass
    
    def view_waveform(self, vcd_file: str, session_id: str = None, 
                     save_file: str = None) -> WaveformSession:
        """Open waveform file in viewer"""
        
        if not Path(vcd_file).exists():
            raise FileNotFoundError(f"VCD file not found: {vcd_file}")
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"waveform_{int(time.time())}"
        
        # Generate save file if not provided
        if save_file is None and self.config.signal_groups:
            work_dir = Path(vcd_file).parent
            save_file = str(work_dir / f"{session_id}.gtkw")
            self.generate_save_file(vcd_file, save_file)
        
        self.logger.info(f"Opening waveform viewer for {vcd_file}")
        
        try:
            session = self.launch_viewer(vcd_file, save_file)
            session.session_id = session_id
            self.active_sessions[session_id] = session
            
            self.logger.info(f"Waveform viewer launched successfully (Session: {session_id})")
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to launch waveform viewer: {e}")
            raise
    
    def close_session(self, session_id: str) -> bool:
        """Close waveform viewer session"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        try:
            if session.viewer_process and session.viewer_process.poll() is None:
                session.viewer_process.terminate()
                
                # Wait for graceful termination
                try:
                    session.viewer_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    session.viewer_process.kill()
            
            session.active = False
            del self.active_sessions[session_id]
            
            self.logger.info(f"Closed waveform viewer session: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing session {session_id}: {e}")
            return False
    
    def close_all_sessions(self) -> int:
        """Close all active waveform viewer sessions"""
        closed_count = 0
        
        for session_id in list(self.active_sessions.keys()):
            if self.close_session(session_id):
                closed_count += 1
        
        return closed_count
    
    def list_active_sessions(self) -> List[str]:
        """List all active waveform viewer sessions"""
        return list(self.active_sessions.keys())


class GTKWaveViewer(WaveformViewer):
    """GTKWave waveform viewer interface"""
    
    def check_availability(self) -> bool:
        """Check if GTKWave is available"""
        try:
            result = subprocess.run(['gtkwave', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def launch_viewer(self, vcd_file: str, save_file: str = None) -> WaveformSession:
        """Launch GTKWave with VCD file"""
        
        cmd = ['gtkwave']
        
        # Add VCD file
        cmd.append(vcd_file)
        
        # Add save file if provided
        if save_file and Path(save_file).exists():
            cmd.append(save_file)
        
        # Add GTKWave-specific options
        if self.config.background_mode:
            cmd.append('--no-splash')
        
        self.logger.info(f"Launching GTKWave: {' '.join(cmd)}")
        
        try:
            if self.config.background_mode:
                # Launch in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
            else:
                # Launch in foreground
                process = subprocess.Popen(cmd)
            
            session = WaveformSession(
                session_id="",  # Will be set by caller
                viewer_process=process,
                vcd_file=vcd_file,
                save_file=save_file,
                pid=process.pid,
                active=True
            )
            
            return session
            
        except Exception as e:
            raise RuntimeError(f"Failed to launch GTKWave: {e}")
    
    def generate_save_file(self, vcd_file: str, output_file: str) -> bool:
        """Generate GTKWave save file with optimal signal grouping"""
        
        try:
            # Parse VCD file to extract signal hierarchy
            signals = self._parse_vcd_signals(vcd_file)
            
            # Generate GTKWave save file content
            save_content = self._generate_gtkwave_save_content(signals)
            
            # Write save file
            with open(output_file, 'w') as f:
                f.write(save_content)
            
            self.logger.info(f"Generated GTKWave save file: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate save file: {e}")
            return False
    
    def _parse_vcd_signals(self, vcd_file: str) -> Dict[str, List[str]]:
        """Parse VCD file to extract signal hierarchy"""
        signals = {}
        current_scope = []
        
        try:
            with open(vcd_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    if line.startswith('$scope'):
                        parts = line.split()
                        if len(parts) >= 3:
                            scope_name = parts[2]
                            current_scope.append(scope_name)
                    
                    elif line.startswith('$upscope'):
                        if current_scope:
                            current_scope.pop()
                    
                    elif line.startswith('$var'):
                        parts = line.split()
                        if len(parts) >= 5:
                            var_type = parts[1]
                            var_size = parts[2]
                            var_id = parts[3]
                            var_name = parts[4]
                            
                            # Build hierarchical signal name
                            if current_scope:
                                scope_path = '.'.join(current_scope)
                                full_name = f"{scope_path}.{var_name}"
                            else:
                                full_name = var_name
                            
                            # Group signals by type or module
                            group_key = current_scope[-1] if current_scope else "top"
                            if group_key not in signals:
                                signals[group_key] = []
                            signals[group_key].append(full_name)
                    
                    elif line.startswith('$enddefinitions'):
                        break  # End of variable definitions
            
        except Exception as e:
            self.logger.warning(f"Error parsing VCD file: {e}")
            signals = {"all": []}  # Fallback
        
        return signals
    
    def _generate_gtkwave_save_content(self, signals: Dict[str, List[str]]) -> str:
        """Generate GTKWave save file content"""
        
        save_lines = [
            "[*]",
            "[*] GTKWave Save File - Generated by EDA Agent",
            "[*]",
            "[dumpfile] (null)",
            "[savefile] (null)",
            "[timestart] 0",
            "[size] 1920 1080",
            "[pos] -1 -1",
            "*-26.000000 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1",
            "[treeopen] TOP",
            ""
        ]
        
        # Add signal groups
        for group_name, group_signals in signals.items():
            if group_signals:
                save_lines.append(f"@{len(save_lines) - 10}")  # Group marker
                save_lines.append(f"-{group_name.upper()}")
                
                for signal in group_signals:
                    # Add signal with appropriate formatting
                    save_lines.append(f"{signal}")
                
                save_lines.append("@{len(save_lines) - 10}")  # End group marker
                save_lines.append("")
        
        # Add common GTKWave settings
        save_lines.extend([
            "[pattern_trace] 1",
            "[pattern_trace] 0"
        ])
        
        return "\n".join(save_lines)
    
    def create_fifo_save_file(self, vcd_file: str, output_file: str, 
                             fifo_module_name: str = "async_fifo") -> bool:
        """Create FIFO-specific GTKWave save file with optimized signal grouping"""
        
        try:
            # FIFO-specific signal groups
            fifo_groups = {
                "Clock_and_Reset": [
                    f"{fifo_module_name}.wr_clk",
                    f"{fifo_module_name}.rd_clk", 
                    f"{fifo_module_name}.rst_n"
                ],
                "Write_Interface": [
                    f"{fifo_module_name}.wr_en",
                    f"{fifo_module_name}.wr_data",
                    f"{fifo_module_name}.full",
                    f"{fifo_module_name}.almost_full"
                ],
                "Read_Interface": [
                    f"{fifo_module_name}.rd_en",
                    f"{fifo_module_name}.rd_data",
                    f"{fifo_module_name}.empty",
                    f"{fifo_module_name}.almost_empty"
                ],
                "Internal_Pointers": [
                    f"{fifo_module_name}.wr_ptr",
                    f"{fifo_module_name}.rd_ptr",
                    f"{fifo_module_name}.wr_ptr_gray",
                    f"{fifo_module_name}.rd_ptr_gray",
                    f"{fifo_module_name}.wr_ptr_sync",
                    f"{fifo_module_name}.rd_ptr_sync"
                ],
                "Memory_Array": [
                    f"{fifo_module_name}.mem"
                ]
            }
            
            # Generate save file content
            save_content = self._generate_fifo_gtkwave_content(fifo_groups)
            
            # Write save file
            with open(output_file, 'w') as f:
                f.write(save_content)
            
            self.logger.info(f"Generated FIFO-specific GTKWave save file: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate FIFO save file: {e}")
            return False
    
    def _generate_fifo_gtkwave_content(self, signal_groups: Dict[str, List[str]]) -> str:
        """Generate FIFO-specific GTKWave save file content"""
        
        save_lines = [
            "[*]",
            "[*] GTKWave Save File - FIFO Debug Configuration",
            "[*] Generated by EDA Agent for FIFO Analysis",
            "[*]",
            "[dumpfile] (null)",
            "[savefile] (null)",
            "[timestart] 0",
            "[size] 1920 1080",
            "[pos] -1 -1",
            "*-26.000000 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1",
            "[treeopen] TOP",
            "[treeopen] TOP.async_fifo_tb",
            "[treeopen] TOP.async_fifo_tb.dut",
            ""
        ]
        
        # Add FIFO-specific signal groups with proper formatting
        group_colors = {
            "Clock_and_Reset": "@28",
            "Write_Interface": "@22", 
            "Read_Interface": "@23",
            "Internal_Pointers": "@24",
            "Memory_Array": "@25"
        }
        
        for group_name, signals in signal_groups.items():
            if signals:
                color = group_colors.get(group_name, "@29")
                
                save_lines.append(color)
                save_lines.append(f"-{group_name.replace('_', ' ')}")
                
                for signal in signals:
                    save_lines.append(f"{signal}")
                
                save_lines.append("@29")  # End group
                save_lines.append("")
        
        # Add display settings optimized for FIFO debugging
        save_lines.extend([
            "[pattern_trace] 1",
            "[pattern_trace] 0",
            "[color] 2",
            "[color] 3",
            "[zoom] -1.000000",
            "[posmag] 1"
        ])
        
        return "\n".join(save_lines)


def get_available_viewers() -> List[str]:
    """Get list of available waveform viewers on the system"""
    available = []
    
    viewers = {
        "gtkwave": GTKWaveViewer()
    }
    
    for name, viewer in viewers.items():
        if viewer.check_availability():
            available.append(name)
    
    return available


def create_waveform_viewer(viewer_name: str, config: WaveformConfig = None) -> WaveformViewer:
    """Factory function to create waveform viewer instances"""
    
    config = config or WaveformConfig()
    
    if viewer_name.lower() == "gtkwave":
        return GTKWaveViewer(config)
    else:
        raise ValueError(f"Unknown waveform viewer: {viewer_name}")


def auto_select_viewer(config: WaveformConfig = None) -> Optional[WaveformViewer]:
    """Automatically select the best available waveform viewer"""
    
    available = get_available_viewers()
    
    if not available:
        return None
    
    # Preference order: GTKWave (only one implemented for now)
    preference_order = ["gtkwave"]
    
    for preferred in preference_order:
        if preferred in available:
            return create_waveform_viewer(preferred, config)
    
    # Fallback to first available
    return create_waveform_viewer(available[0], config)