# 🤖 Agentic AI-Powered FIFO Design and Verification System

An autonomous EDA (Electronic Design Automation) system that uses multi-agent AI collaboration to design, implement, and verify SystemVerilog FIFO buffers from natural language requirements.

## 🎯 Key Features

### Multi-Agent Architecture
- **🎯 SpecAgent**: Converts natural language to formal RTL specifications
- **⚙️ CodeAgent**: Generates synthesizable SystemVerilog with Gray code pointers
- **🧪 VerifyAgent**: Creates comprehensive testbenches and verification environments
- **🐛 DebugAgent**: Analyzes issues and orchestrates autonomous design iterations

### Autonomous Capabilities
- **Natural Language Understanding**: Human requirements → formal specifications
- **Self-Correction Loops**: Identifies and fixes design issues automatically
- **Multi-Agent Collaboration**: Specialized agents working together seamlessly
- **Tool Integration**: Works with industry-standard EDA tools (iverilog, Yosys, GTKWave)

### FIFO Design Expertise
- **Asynchronous FIFOs** with proper clock domain crossing safety
- **Gray Code Pointers** for metastability prevention
- **Configurable Parameters** (depth, data width, thresholds)
- **Comprehensive Verification** with coverage-driven testing

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ required
pip install -r requirements.txt

# Optional: Install EDA tools for full functionality
# - Icarus Verilog (iverilog)
# - Yosys (synthesis)
# - GTKWave (waveform viewing)
```

### Run the Demo
```bash
# Interactive demonstration
python3 simple_demo.py

# Full system test (requires EDA tools)
python3 main.py --requirement "Design a 32-deep async FIFO with 16-bit data width"

# Tool integration test
python3 test_tool_integration.py
```

## 📁 Project Structure

```
EDA-agent/
├── agents/                 # AI agent implementations
│   ├── spec_agent.py      # Natural language → specification
│   ├── code_agent.py      # SystemVerilog RTL generation
│   ├── verify_agent.py    # Testbench and verification
│   ├── debug_agent.py     # Issue analysis & orchestration
│   └── base_agent.py      # Common agent functionality
├── tools/                 # EDA tool integration layer
│   ├── simulator.py       # Simulator interfaces (iverilog, ModelSim, Verilator)
│   ├── synthesizer.py     # Synthesis tools (Yosys)
│   └── waveform_viewer.py # Waveform viewers (GTKWave)
├── main.py               # Main orchestration system
├── simple_demo.py        # Simplified demonstration
└── test_tool_integration.py # Tool integration tests
```

## 🎮 Usage Examples

### Basic FIFO Design
```python
from main import FIFOAgenticEDA

# Initialize the system
eda_system = FIFOAgenticEDA(output_dir="./my_fifo_design")

# Natural language requirement
requirement = """
Design a parameterizable asynchronous FIFO with configurable depth and data width,
including Gray code pointers for clock domain crossing safety. The FIFO should
support depths of 16, 32, 64, or 128 entries and data widths of 8, 16, or 32 bits.
"""

# Run autonomous design flow
results = eda_system.run_agentic_fifo_design(requirement)

if results['success']:
    print(f"✅ Design completed in {results['total_time']:.2f} seconds")
    print(f"📁 Files generated in: {results['rtl']['export_path']}")
```

### Tool Integration
```python
from tools import auto_select_simulator, SimulationConfig

# Automatically select best available simulator
simulator = auto_select_simulator()
if simulator:
    config = SimulationConfig(generate_vcd=True, timeout_seconds=60)
    result = simulator.run_simulation(['my_fifo.sv'], 'fifo_tb')
```

## 🧪 Example Output

The system autonomously generates:

### Generated SystemVerilog FIFO
```systemverilog
module async_fifo #(
    parameter DEPTH = 16,
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 4
)(
    // Write interface
    input  wire                   wr_clk,
    input  wire                   wr_rst_n,
    input  wire                   wr_en,
    input  wire [DATA_WIDTH-1:0]  wr_data,
    output reg                    full,
    
    // Read interface  
    input  wire                   rd_clk,
    input  wire                   rd_rst_n,
    input  wire                   rd_en,
    output reg  [DATA_WIDTH-1:0]  rd_data,
    output reg                    empty
);
    // Gray code pointer implementation...
```

### Comprehensive Testbench
- Constraint random testing
- Corner case coverage
- Clock domain crossing validation
- Functional coverage points

## 🔧 EDA Tool Support

### Simulators
- **Icarus Verilog** (iverilog) - Open source
- **ModelSim/QuestaSim** - Industry standard
- **Verilator** - High performance

### Synthesizers
- **Yosys** - Open source synthesis
- **Vivado/Quartus** - FPGA synthesis (planned)

### Debug Tools
- **GTKWave** - Waveform viewing with auto-generated save files
- **Custom signal grouping** for FIFO debugging

## 📊 Autonomous Workflow

1. **🎯 Natural Language Parsing**: Human requirement → formal specification
2. **⚙️ RTL Generation**: Specification → synthesizable SystemVerilog
3. **🧪 Verification**: Automated testbench creation and simulation
4. **🐛 Debug & Iterate**: Issue analysis → targeted fixes → re-verification
5. **📄 Documentation**: Comprehensive reports and metrics

## 🎯 Use Cases

- **Rapid Prototyping**: Quickly generate FIFO designs from requirements
- **Design Exploration**: Test different FIFO configurations automatically  
- **Verification Automation**: Generate comprehensive test suites
- **Design Education**: Learn FIFO design patterns and best practices
- **Research Platform**: Experiment with agentic AI in EDA workflows

## 🤝 Contributing

This project demonstrates the potential of agentic AI in hardware design automation. Contributions welcome for:

- Additional RTL design patterns beyond FIFOs
- New EDA tool integrations
- Enhanced verification methodologies
- Performance optimizations

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

## 🔬 Research Context

This system represents cutting-edge research in:
- **Agentic AI Systems** - Autonomous multi-agent collaboration
- **AI for EDA** - Machine learning applied to electronic design
- **Natural Language Hardware Design** - Bridging human intent and RTL
- **Autonomous Verification** - Self-correcting design workflows

---

**🎉 Ready to revolutionize hardware design with autonomous AI agents!**