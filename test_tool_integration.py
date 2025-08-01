#!/usr/bin/env python3
"""
Test Tool Integration Layer

Validates the EDA tool integration interfaces and demonstrates
autonomous tool selection and execution capabilities.
"""

import sys
import logging
import tempfile
from pathlib import Path

# Import tool integration layer
from tools import (
    get_available_simulators, create_simulator, auto_select_simulator,
    get_available_synthesizers, create_synthesizer, auto_select_synthesizer,
    get_available_viewers, create_waveform_viewer, auto_select_viewer,
    SimulationConfig, SynthesisConfig, WaveformConfig
)


def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("ToolIntegrationTest")


def create_sample_verilog():
    """Create a sample SystemVerilog FIFO for testing"""
    
    fifo_code = '''
module simple_fifo #(
    parameter DEPTH = 8,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire wr_en,
    input wire [DATA_WIDTH-1:0] wr_data,
    output reg full,
    input wire rd_en,
    output reg [DATA_WIDTH-1:0] rd_data,
    output reg empty
);

    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH):0] wr_ptr, rd_ptr;
    reg [$clog2(DEPTH):0] count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            count <= 0;
            full <= 0;
            empty <= 1;
        end else begin
            // Write operation
            if (wr_en && !full) begin
                mem[wr_ptr[2:0]] <= wr_data;
                wr_ptr <= wr_ptr + 1;
                count <= count + 1;
            end
            
            // Read operation
            if (rd_en && !empty) begin
                rd_data <= mem[rd_ptr[2:0]];
                rd_ptr <= rd_ptr + 1;
                count <= count - 1;
            end
            
            // Update flags
            full <= (count == DEPTH-1) && wr_en && !rd_en;
            empty <= (count == 1) && rd_en && !wr_en;
        end
    end

    // Generate VCD for waveform viewing
    initial begin
        $dumpfile("simple_fifo.vcd");
        $dumpvars(0, simple_fifo);
    end

endmodule

// Simple testbench
module simple_fifo_tb;
    reg clk, rst_n, wr_en, rd_en;
    reg [7:0] wr_data;
    wire [7:0] rd_data;
    wire full, empty;

    simple_fifo #(.DEPTH(8), .DATA_WIDTH(8)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(wr_en),
        .wr_data(wr_data),
        .full(full),
        .rd_en(rd_en),
        .rd_data(rd_data),
        .empty(empty)
    );

    // Clock generation
    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        wr_en = 0;
        rd_en = 0;
        wr_data = 0;

        // Reset
        #10 rst_n = 1;

        // Write some data
        repeat(5) begin
            @(posedge clk);
            wr_en = 1;
            wr_data = $random;
        end
        wr_en = 0;

        // Read some data
        repeat(3) begin
            @(posedge clk);
            rd_en = 1;
        end
        rd_en = 0;

        // Finish simulation
        #50 $finish;
    end
endmodule
'''
    
    return fifo_code


def test_simulator_integration(logger, work_dir):
    """Test simulator integration"""
    logger.info("=== Testing Simulator Integration ===")
    
    # Check available simulators
    available_sims = get_available_simulators()
    logger.info(f"Available simulators: {available_sims}")
    
    if not available_sims:
        logger.warning("No simulators available - skipping simulation tests")
        return False
    
    # Create sample Verilog file
    verilog_file = work_dir / "simple_fifo.sv"
    with open(verilog_file, 'w') as f:
        f.write(create_sample_verilog())
    
    # Test auto-selection
    simulator = auto_select_simulator()
    if simulator:
        logger.info(f"Auto-selected simulator: {simulator.__class__.__name__}")
        
        # Configure simulation
        config = SimulationConfig(
            generate_vcd=True,
            timeout_seconds=60
        )
        simulator.config = config
        
        # Run simulation
        try:
            result = simulator.run_simulation(
                source_files=[str(verilog_file)],
                work_dir=str(work_dir),
                top_module="simple_fifo_tb"
            )
            
            logger.info(f"Simulation result: {'SUCCESS' if result.success else 'FAILED'}")
            logger.info(f"Compile time: {result.compile_time:.2f}s")
            logger.info(f"Run time: {result.run_time:.2f}s")
            
            if result.vcd_file:
                logger.info(f"VCD file generated: {result.vcd_file}")
                return result.vcd_file
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
    
    return None


def test_synthesizer_integration(logger, work_dir):
    """Test synthesizer integration"""
    logger.info("=== Testing Synthesizer Integration ===")
    
    # Check available synthesizers
    available_synths = get_available_synthesizers()
    logger.info(f"Available synthesizers: {available_synths}")
    
    if not available_synths:
        logger.warning("No synthesizers available - skipping synthesis tests")
        return False
    
    # Create sample Verilog file for synthesis
    verilog_file = work_dir / "simple_fifo_synth.sv"
    with open(verilog_file, 'w') as f:
        # Write only the module, not the testbench
        lines = create_sample_verilog().split('\n')
        synth_lines = []
        in_testbench = False
        
        for line in lines:
            if 'module simple_fifo_tb' in line:
                in_testbench = True
            elif line.startswith('endmodule') and in_testbench:
                in_testbench = False
                continue
            elif not in_testbench and not line.strip().startswith('//'):
                synth_lines.append(line)
        
        f.write('\n'.join(synth_lines))
    
    # Test auto-selection
    synthesizer = auto_select_synthesizer()
    if synthesizer:
        logger.info(f"Auto-selected synthesizer: {synthesizer.__class__.__name__}")
        
        # Configure synthesis
        config = SynthesisConfig(
            target_technology="generic",
            optimization_level="medium",
            timeout_seconds=120
        )
        synthesizer.config = config
        
        # Run synthesis
        try:
            result = synthesizer.run_synthesis(
                source_files=[str(verilog_file)],
                top_module="simple_fifo",
                work_dir=str(work_dir)
            )
            
            logger.info(f"Synthesis result: {'SUCCESS' if result.success else 'FAILED'}")
            logger.info(f"Synthesis time: {result.synthesis_time:.2f}s")
            
            if result.success:
                logger.info(f"Netlist generated: {result.netlist_file}")
                if result.gate_count:
                    logger.info(f"Gate count: {result.gate_count}")
                if result.flip_flop_count:
                    logger.info(f"Flip-flop count: {result.flip_flop_count}")
            
            return result.success
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
    
    return False


def test_waveform_viewer_integration(logger, work_dir, vcd_file):
    """Test waveform viewer integration"""
    logger.info("=== Testing Waveform Viewer Integration ===")
    
    if not vcd_file or not Path(vcd_file).exists():
        logger.warning("No VCD file available - skipping waveform viewer tests")
        return False
    
    # Check available viewers
    available_viewers = get_available_viewers()
    logger.info(f"Available waveform viewers: {available_viewers}")
    
    if not available_viewers:
        logger.warning("No waveform viewers available - skipping viewer tests")
        return False
    
    # Test auto-selection
    viewer = auto_select_viewer()
    if viewer:
        logger.info(f"Auto-selected viewer: {viewer.__class__.__name__}")
        
        # Configure viewer for background mode (non-interactive test)
        config = WaveformConfig(
            background_mode=True,
            auto_zoom=True,
            show_hierarchy=True
        )
        viewer.config = config
        
        # Test save file generation
        try:
            save_file = work_dir / "fifo_debug.gtkw"
            
            # Test FIFO-specific save file generation
            if hasattr(viewer, 'create_fifo_save_file'):
                success = viewer.create_fifo_save_file(
                    vcd_file=vcd_file,
                    output_file=str(save_file),
                    fifo_module_name="simple_fifo"
                )
                
                if success:
                    logger.info(f"FIFO save file generated: {save_file}")
                else:
                    logger.warning("Failed to generate FIFO save file")
            
            # Test opening viewer (in background mode for automated testing)
            session = viewer.view_waveform(
                vcd_file=vcd_file,
                session_id="test_session",
                save_file=str(save_file) if save_file.exists() else None
            )
            
            if session and session.active:
                logger.info(f"Waveform viewer launched successfully (PID: {session.pid})")
                
                # Close the session immediately for automated testing
                import time
                time.sleep(2)  # Give viewer time to start
                viewer.close_session("test_session")
                logger.info("Waveform viewer session closed")
                
                return True
            
        except Exception as e:
            logger.error(f"Waveform viewer test failed: {e}")
    
    return False


def test_tool_availability_reporting(logger):
    """Test comprehensive tool availability reporting"""
    logger.info("=== Tool Availability Report ===")
    
    # Generate comprehensive tool report
    tool_report = {
        "simulators": {
            "available": get_available_simulators(),
            "auto_selected": None
        },
        "synthesizers": {
            "available": get_available_synthesizers(),
            "auto_selected": None
        },
        "waveform_viewers": {
            "available": get_available_viewers(),
            "auto_selected": None
        }
    }
    
    # Test auto-selection for each tool type
    try:
        sim = auto_select_simulator()
        tool_report["simulators"]["auto_selected"] = sim.__class__.__name__ if sim else None
    except:
        pass
    
    try:
        synth = auto_select_synthesizer()
        tool_report["synthesizers"]["auto_selected"] = synth.__class__.__name__ if synth else None
    except:
        pass
    
    try:
        viewer = auto_select_viewer()
        tool_report["waveform_viewers"]["auto_selected"] = viewer.__class__.__name__ if viewer else None
    except:
        pass
    
    # Print detailed report
    for tool_type, info in tool_report.items():
        logger.info(f"{tool_type.upper()}:")
        logger.info(f"  Available: {info['available']}")
        logger.info(f"  Auto-selected: {info['auto_selected']}")
    
    return tool_report


def main():
    """Main test execution"""
    logger = setup_logging()
    
    logger.info("Starting EDA Tool Integration Tests")
    logger.info("=" * 60)
    
    # Create temporary working directory
    with tempfile.TemporaryDirectory(prefix="eda_tool_test_") as temp_dir:
        work_dir = Path(temp_dir)
        logger.info(f"Working directory: {work_dir}")
        
        # Test results tracking
        results = {
            "tool_availability": False,
            "simulator_test": False,
            "synthesizer_test": False,
            "waveform_viewer_test": False
        }
        
        try:
            # Test 1: Tool availability reporting
            tool_report = test_tool_availability_reporting(logger)
            results["tool_availability"] = True
            
            # Test 2: Simulator integration
            vcd_file = test_simulator_integration(logger, work_dir)
            results["simulator_test"] = vcd_file is not None
            
            # Test 3: Synthesizer integration
            results["synthesizer_test"] = test_synthesizer_integration(logger, work_dir)
            
            # Test 4: Waveform viewer integration
            results["waveform_viewer_test"] = test_waveform_viewer_integration(logger, work_dir, vcd_file)
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
        
        # Print final results
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tool integration tests passed!")
            return 0
        else:
            logger.warning(f"⚠️  {total_tests - passed_tests} test(s) failed")
            return 1


if __name__ == "__main__":
    sys.exit(main())