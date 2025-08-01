"""
SpecAgent: Natural Language to Formal RTL Specification

Parses natural language requirements and generates formal specifications
for FIFO design including interface definitions, constraints, and parameters.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import re
import json

from .base_agent import BaseAgent, AgentConfig, TaskStatus


class FIFOInterface(BaseModel):
    """FIFO Interface specification"""
    # Write interface
    write_clock: str = "wr_clk"
    write_reset: str = "wr_rst_n"
    write_enable: str = "wr_en"
    write_data: str = "wr_data"
    full_flag: str = "full"
    almost_full_flag: str = "almost_full"
    
    # Read interface
    read_clock: str = "rd_clk"
    read_reset: str = "rd_rst_n"
    read_enable: str = "rd_en"
    read_data: str = "rd_data"
    empty_flag: str = "empty"
    almost_empty_flag: str = "almost_empty"
    
    @property
    def input_signals(self) -> List[str]:
        """Get list of input signals"""
        return [
            self.write_clock, self.write_reset, self.write_enable, self.write_data,
            self.read_clock, self.read_reset, self.read_enable
        ]
    
    @property
    def output_signals(self) -> List[str]:
        """Get list of output signals"""
        return [
            self.read_data, self.full_flag, self.almost_full_flag,
            self.empty_flag, self.almost_empty_flag
        ]
    
    @property
    def clock_signals(self) -> List[str]:
        """Get list of clock signals"""
        return [self.write_clock, self.read_clock]


class FIFOParameters(BaseModel):
    """FIFO Parameters specification"""
    depth: int = 16
    data_width: int = 8
    addr_width: Optional[int] = None
    almost_full_threshold: int = 2
    almost_empty_threshold: int = 2
    
    def __post_init__(self):
        if self.addr_width is None:
            # Calculate address width based on depth
            import math
            self.addr_width = max(1, math.ceil(math.log2(self.depth)))


class FIFOConstraints(BaseModel):
    """FIFO Design constraints"""
    async_clocks: bool = True
    gray_code_pointers: bool = True
    reset_type: str = "async_active_low"  # async_active_low, async_active_high, sync
    metastability_stages: int = 2
    enable_assertions: bool = True
    enable_coverage: bool = True


class FIFOSpecification(BaseModel):
    """Complete FIFO formal specification"""
    module_name: str = "async_fifo"
    description: str
    interface: FIFOInterface
    parameters: FIFOParameters
    constraints: FIFOConstraints
    requirements: List[str]
    test_scenarios: List[str]


class SpecAgent(BaseAgent):
    """
    Agent responsible for parsing natural language into formal RTL specifications
    
    Key capabilities:
    - Natural language processing for FIFO requirements
    - Parameter extraction and validation
    - Interface specification generation
    - Constraint formalization
    - Test scenario identification
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.spec_templates = self._load_spec_templates()
        self.current_spec: Optional[FIFOSpecification] = None
    
    def get_capabilities(self) -> List[str]:
        return [
            "parse_natural_language",
            "extract_parameters", 
            "define_interface",
            "specify_constraints",
            "generate_requirements",
            "identify_test_scenarios"
        ]
    
    def _execute_task_impl(self, task_name: str, **kwargs) -> Any:
        """Execute specification-related tasks"""
        if task_name == "parse_natural_language":
            return self._parse_natural_language(kwargs.get("requirement_text", ""))
        elif task_name == "extract_parameters":
            return self._extract_parameters(kwargs.get("text", ""))
        elif task_name == "define_interface":
            return self._define_interface(kwargs.get("parameters"))
        elif task_name == "specify_constraints":
            return self._specify_constraints(kwargs.get("text", ""))
        elif task_name == "generate_requirements":
            return self._generate_requirements()
        elif task_name == "identify_test_scenarios":
            return self._identify_test_scenarios()
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def _load_spec_templates(self) -> Dict[str, str]:
        """Load specification templates"""
        return {
            "fifo_requirements": [
                "Data integrity: Data written to FIFO must be read in FIFO order",
                "No data loss: Data must not be lost when FIFO is not full",
                "No invalid reads: Empty FIFO must not provide valid data",
                "Clock domain safety: Async clocks must not cause metastability",
                "Reset behavior: Reset must initialize FIFO to empty state",
                "Flag accuracy: Full/empty flags must accurately reflect FIFO state"
            ],
            "test_scenarios": [
                "Basic write-then-read sequence",
                "Simultaneous read/write operations",
                "Fill FIFO completely then drain",
                "Random access patterns with different clock ratios",
                "Reset during active operations",
                "Clock domain crossing stress test",
                "Almost-full/almost-empty threshold verification"
            ]
        }
    
    def _parse_natural_language(self, requirement_text: str) -> FIFOSpecification:
        """
        Parse natural language requirement into formal specification
        """
        self.logger.info("Parsing natural language requirement")
        
        # Extract parameters from text
        parameters = self._extract_parameters(requirement_text)
        
        # Define interface based on requirements
        interface = self._define_interface(parameters)
        
        # Specify constraints
        constraints = self._specify_constraints(requirement_text)
        
        # Generate formal requirements
        requirements = self._generate_requirements()
        
        # Identify test scenarios
        test_scenarios = self._identify_test_scenarios()
        
        # Create complete specification
        spec = FIFOSpecification(
            module_name="async_fifo",
            description=self._extract_description(requirement_text),
            interface=interface,
            parameters=parameters,
            constraints=constraints,
            requirements=requirements,
            test_scenarios=test_scenarios
        )
        
        self.current_spec = spec
        self.logger.info(f"Generated specification: {spec.module_name} with depth={spec.parameters.depth}, width={spec.parameters.data_width}")
        
        return spec
    
    def _extract_parameters(self, text: str) -> FIFOParameters:
        """Extract FIFO parameters from text"""
        parameters = FIFOParameters()
        
        # Extract depth
        depth_match = re.search(r'depth\s*(?:of\s*)?(\d+)|(\d+)\s*(?:deep|entries)', text.lower())
        if depth_match:
            parameters.depth = int(depth_match.group(1) or depth_match.group(2))
        
        # Extract data width
        width_match = re.search(r'(\d+)[-\s]*bit|width\s*(?:of\s*)?(\d+)', text.lower())
        if width_match:
            parameters.data_width = int(width_match.group(1) or width_match.group(2))
        
        # Calculate address width
        import math
        parameters.addr_width = max(1, math.ceil(math.log2(parameters.depth)))
        
        # Set thresholds (default to 2 entries from full/empty)
        parameters.almost_full_threshold = min(2, parameters.depth // 4)
        parameters.almost_empty_threshold = min(2, parameters.depth // 4)
        
        self.logger.info(f"Extracted parameters: depth={parameters.depth}, width={parameters.data_width}")
        return parameters
    
    def _define_interface(self, parameters: FIFOParameters) -> FIFOInterface:
        """Define FIFO interface based on parameters"""
        interface = FIFOInterface()
        
        # Interface is standard for async FIFO
        # All signal names already defined in FIFOInterface class
        
        self.logger.info("Defined standard async FIFO interface")
        return interface
    
    def _specify_constraints(self, text: str) -> FIFOConstraints:
        """Extract and specify design constraints"""
        constraints = FIFOConstraints()
        
        # Check for async/sync specification
        if "synchronous" in text.lower() or "same clock" in text.lower():
            constraints.async_clocks = False
        
        # Check for gray code requirement
        if "binary" in text.lower() or "no gray" in text.lower():
            constraints.gray_code_pointers = False
        
        # Check reset type
        if "active high" in text.lower():
            constraints.reset_type = "async_active_high"
        elif "synchronous reset" in text.lower():
            constraints.reset_type = "sync"
        
        # Check metastability stages
        stages_match = re.search(r'(\d+)\s*(?:stage|flip-?flop).*sync', text.lower())
        if stages_match:
            constraints.metastability_stages = int(stages_match.group(1))
        
        self.logger.info(f"Specified constraints: async={constraints.async_clocks}, gray_code={constraints.gray_code_pointers}")
        return constraints
    
    def _extract_description(self, text: str) -> str:
        """Extract or generate description from requirement text"""
        # Clean up the text and use as description
        description = re.sub(r'\\s+', ' ', text.strip())
        if len(description) > 200:
            description = description[:200] + "..."
        
        if not description:
            description = "Parameterizable asynchronous FIFO buffer with Gray code pointers for clock domain crossing safety"
        
        return description
    
    def _generate_requirements(self) -> List[str]:
        """Generate formal requirements list"""
        return self.spec_templates["fifo_requirements"]
    
    def _identify_test_scenarios(self) -> List[str]:
        """Identify test scenarios based on specification"""
        return self.spec_templates["test_scenarios"]
    
    def validate_specification(self, spec: FIFOSpecification) -> Dict[str, Any]:
        """Validate the generated specification"""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check parameter validity
        if spec.parameters.depth < 2:
            validation_results["errors"].append("FIFO depth must be at least 2")
            validation_results["valid"] = False
            
        if spec.parameters.data_width < 1:
            validation_results["errors"].append("Data width must be at least 1")
            validation_results["valid"] = False
        
        # Check if depth is power of 2 for Gray code
        if spec.constraints.gray_code_pointers:
            import math
            if not (spec.parameters.depth & (spec.parameters.depth - 1)) == 0:
                validation_results["warnings"].append("FIFO depth should be power of 2 for optimal Gray code implementation")
        
        # Check threshold values
        if spec.parameters.almost_full_threshold >= spec.parameters.depth:
            validation_results["errors"].append("Almost full threshold must be less than FIFO depth")
            validation_results["valid"] = False
        
        self.logger.info(f"Specification validation: {validation_results}")
        return validation_results
    
    def export_specification(self, output_path: Optional[str] = None) -> str:
        """Export specification to JSON file"""
        if not self.current_spec:
            raise ValueError("No specification available to export")
        
        if output_path is None:
            output_path = str(self.config.output_dir / "fifo_specification.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.current_spec.model_dump(), f, indent=2)
        
        self.logger.info(f"Exported specification to {output_path}")
        return output_path
    
    def generate_interface_header(self) -> str:
        """Generate SystemVerilog interface header"""
        if not self.current_spec:
            raise ValueError("No specification available")
        
        spec = self.current_spec
        
        header = f"""
// {spec.description}
module {spec.module_name} #(
    parameter FIFO_DEPTH = {spec.parameters.depth},
    parameter DATA_WIDTH = {spec.parameters.data_width},
    parameter ADDR_WIDTH = $clog2(FIFO_DEPTH),
    parameter ALMOST_FULL_THRESHOLD = {spec.parameters.almost_full_threshold},
    parameter ALMOST_EMPTY_THRESHOLD = {spec.parameters.almost_empty_threshold}
)(
    // Write interface
    input  wire                 {spec.interface.write_clock},
    input  wire                 {spec.interface.write_reset},
    input  wire                 {spec.interface.write_enable},
    input  wire [DATA_WIDTH-1:0] {spec.interface.write_data},
    output wire                 {spec.interface.full_flag},
    output wire                 {spec.interface.almost_full_flag},
    
    // Read interface  
    input  wire                 {spec.interface.read_clock},
    input  wire                 {spec.interface.read_reset},
    input  wire                 {spec.interface.read_enable},
    output wire [DATA_WIDTH-1:0] {spec.interface.read_data},
    output wire                 {spec.interface.empty_flag},
    output wire                 {spec.interface.almost_empty_flag}
);
"""
        return header.strip()