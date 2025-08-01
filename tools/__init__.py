"""
EDA Tool Integration Layer

Provides wrappers for common EDA tools used in the agentic workflow.
"""

from .simulator import (
    SimulatorInterface, IcarusVerilogSimulator, VerilatorSimulator, ModelSimSimulator,
    SimulationConfig, SimulationResult, get_available_simulators, create_simulator, auto_select_simulator
)
from .synthesizer import (
    SynthesizerInterface, YosysSynthesizer, SynthesisConfig, SynthesisResult,
    get_available_synthesizers, create_synthesizer, auto_select_synthesizer
)
from .waveform_viewer import (
    WaveformViewer, GTKWaveViewer, WaveformConfig, WaveformSession,
    get_available_viewers, create_waveform_viewer, auto_select_viewer
)

__all__ = [
    # Simulator classes and functions
    "SimulatorInterface", "IcarusVerilogSimulator", "VerilatorSimulator", "ModelSimSimulator",
    "SimulationConfig", "SimulationResult", "get_available_simulators", "create_simulator", "auto_select_simulator",
    
    # Synthesizer classes and functions
    "SynthesizerInterface", "YosysSynthesizer", "SynthesisConfig", "SynthesisResult",
    "get_available_synthesizers", "create_synthesizer", "auto_select_synthesizer",
    
    # Waveform viewer classes and functions
    "WaveformViewer", "GTKWaveViewer", "WaveformConfig", "WaveformSession",
    "get_available_viewers", "create_waveform_viewer", "auto_select_viewer"
]