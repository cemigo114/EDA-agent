#!/usr/bin/env python3
"""
Agentic AI-Powered FIFO Design and Verification System
Main orchestration entry point

Demonstrates autonomous EDA workflow with multi-agent collaboration
for FIFO design, implementation, and verification.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

from agents import SpecAgent, CodeAgent, VerifyAgent, DebugAgent
from agents.base_agent import AgentConfig, TaskStatus


class FIFOAgenticEDA:
    """
    Main orchestrator for the Agentic FIFO EDA System
    
    Coordinates the four specialized agents:
    - SpecAgent: Natural language → formal specification
    - CodeAgent: SystemVerilog RTL generation
    - VerifyAgent: Testbench creation and simulation
    - DebugAgent: Issue analysis and iteration orchestration
    """
    
    def __init__(self, output_dir: str = "./reports", max_iterations: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        self.logger.info("Initialized Agentic FIFO EDA System")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all four agents with their configurations"""
        
        agents = {}
        
        # SpecAgent configuration
        spec_config = AgentConfig(
            name="SpecAgent",
            description="Natural language to formal specification parser",
            max_iterations=self.max_iterations,
            output_dir=self.output_dir / "spec"
        )
        agents["spec"] = SpecAgent(spec_config)
        
        # CodeAgent configuration
        code_config = AgentConfig(
            name="CodeAgent", 
            description="SystemVerilog RTL generator and refiner",
            max_iterations=self.max_iterations,
            output_dir=self.output_dir / "code"
        )
        agents["code"] = CodeAgent(code_config)
        
        # VerifyAgent configuration
        verify_config = AgentConfig(
            name="VerifyAgent",
            description="Testbench generator and simulation executor",
            max_iterations=self.max_iterations,
            output_dir=self.output_dir / "verify"
        )
        agents["verify"] = VerifyAgent(verify_config)
        
        # DebugAgent configuration
        debug_config = AgentConfig(
            name="DebugAgent",
            description="Issue analyzer and iteration orchestrator",
            max_iterations=self.max_iterations,
            output_dir=self.output_dir / "debug"
        )
        agents["debug"] = DebugAgent(debug_config)
        
        return agents
    
    def _setup_logging(self) -> logging.Logger:
        """Setup main orchestrator logging"""
        logger = logging.getLogger("AgenticFIFO")
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "main.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def run_agentic_fifo_design(self, requirement: str) -> Dict[str, Any]:
        """
        Execute the complete agentic FIFO design flow
        
        Args:
            requirement: Natural language FIFO requirement
            
        Returns:
            Complete design flow results
        """
        
        self.logger.info("=" * 80)
        self.logger.info("Starting Agentic FIFO Design Flow")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        # Initialize debug session
        session_id = f"fifo_design_{int(time.time())}"
        debug_session = self.agents["debug"].start_debug_session(session_id, self.max_iterations)
        
        try:
            # Phase 1: Specification Generation
            self.logger.info("\\n🎯 Phase 1: Generating Formal Specification")
            spec_result = self.agents["spec"].execute_task("parse_natural_language", requirement_text=requirement)
            
            if spec_result.status != TaskStatus.COMPLETED:
                raise Exception(f"Specification generation failed: {spec_result.error_message}")
            
            specification = spec_result.output
            self.logger.info(f"✅ Generated specification: {specification.module_name} ({specification.parameters.depth}x{specification.parameters.data_width})")
            
            # Phase 2: Initial RTL Generation
            self.logger.info("\\n⚙️  Phase 2: Generating SystemVerilog RTL")
            code_result = self.agents["code"].execute_task("generate_rtl", specification=specification)
            
            if code_result.status != TaskStatus.COMPLETED:
                raise Exception(f"RTL generation failed: {code_result.error_message}")
            
            rtl_generation_result = code_result.output
            self.logger.info(f"✅ Generated RTL with {len(rtl_generation_result.warnings)} warnings, {len(rtl_generation_result.errors)} errors")
            
            # Phase 3: Testbench Generation
            self.logger.info("\\n🧪 Phase 3: Generating Verification Environment") 
            tb_result = self.agents["verify"].execute_task("generate_testbench", specification=specification)
            
            if tb_result.status != TaskStatus.COMPLETED:
                raise Exception(f"Testbench generation failed: {tb_result.error_message}")
            
            testbench = tb_result.output
            self.logger.info("✅ Generated comprehensive testbench")
            
            # Phase 4: Self-Correction Loop
            self.logger.info("\\n🔄 Phase 4: Autonomous Self-Correction Loop")
            
            for iteration in range(1, self.max_iterations + 1):
                self.current_iteration = iteration
                self.logger.info(f"\\n--- Iteration {iteration}/{self.max_iterations} ---")
                
                # Run simulation
                sim_result = self.agents["verify"].execute_task("run_simulation", rtl_files=[])
                simulation_results = sim_result.output if sim_result.status == TaskStatus.COMPLETED else []
                
                # Analyze results and identify issues
                compilation_issues = self.agents["debug"].execute_task("analyze_compilation_errors", code_result=rtl_generation_result)
                simulation_issues = self.agents["debug"].execute_task("analyze_simulation_failures", sim_results=simulation_results)
                
                # Check for coverage gaps
                coverage_result = self.agents["verify"].execute_task("analyze_coverage")
                if coverage_result.status == TaskStatus.COMPLETED:
                    coverage_issues = self.agents["debug"].execute_task("identify_coverage_gaps", coverage_report=coverage_result.output)
                
                # Plan iteration
                iteration_plan_result = self.agents["debug"].execute_task("plan_iteration")
                if iteration_plan_result.status != TaskStatus.COMPLETED:
                    self.logger.error("Failed to plan iteration")
                    break
                
                iteration_plan = iteration_plan_result.output
                
                # Check convergence
                test_results = {
                    "total_tests": len(simulation_results),
                    "passed_tests": len([r for r in simulation_results if r.result.name == "PASS"]),
                }
                
                convergence_result = self.agents["debug"].execute_task("track_convergence", results=test_results)
                convergence_status = convergence_result.output
                
                self.logger.info(f"Success rate: {convergence_status['success_rate']:.1%}")
                self.logger.info(f"Critical issues: {convergence_status['critical_issues_remaining']}")
                
                # Check if converged
                if convergence_status["converged"]:
                    self.logger.info("🎉 Design converged successfully!")
                    break
                
                # Execute refinements
                if len(iteration_plan.actions) > 0:
                    orchestration_result = self.agents["debug"].execute_task("orchestrate_fixes", iteration_plan=iteration_plan)
                    
                    # Apply refinements
                    if any(action["agent"] == "CodeAgent" for action in iteration_plan.actions):
                        issues_for_code = [issue.description for issue in iteration_plan.priority_issues if issue.affected_component == "code"]
                        if issues_for_code:
                            refine_result = self.agents["code"].execute_task("refine_code", issues=issues_for_code)
                            if refine_result.status == TaskStatus.COMPLETED:
                                rtl_generation_result = refine_result.output
                                self.logger.info("✅ Applied code refinements")
                
                else:
                    self.logger.warning("No actions planned for this iteration")
                    break
            
            # Phase 5: Final Results and Documentation
            self.logger.info("\\n📊 Phase 5: Generating Final Results")
            
            # Export final artifacts
            rtl_export_path = self.agents["code"].execute_task("export_code")
            tb_export_path = self.agents["verify"].execute_task("export_testbench")
            spec_export_path = self.agents["spec"].execute_task("export_specification")
            
            # Generate reports
            verification_report = self.agents["verify"].execute_task("create_verification_report")
            debug_report = self.agents["debug"].execute_task("generate_debug_report")
            
            # End debug session
            session_summary = self.agents["debug"].end_debug_session()
            
            total_time = time.time() - start_time
            
            # Compile final results
            results = {
                "success": session_summary["success"],
                "total_time": total_time,
                "iterations": session_summary["total_iterations"],
                "final_success_rate": session_summary["final_success_rate"],
                "specification": {
                    "module_name": specification.module_name,
                    "depth": specification.parameters.depth,
                    "data_width": specification.parameters.data_width,
                    "export_path": spec_export_path.output if spec_export_path.status == TaskStatus.COMPLETED else None
                },
                "rtl": {
                    "syntax_valid": rtl_generation_result.syntax_valid,
                    "warnings": len(rtl_generation_result.warnings),
                    "errors": len(rtl_generation_result.errors),
                    "export_path": rtl_export_path.output if rtl_export_path.status == TaskStatus.COMPLETED else None
                },
                "verification": {
                    "testbench_generated": tb_result.status == TaskStatus.COMPLETED,
                    "simulations_run": len(simulation_results),
                    "export_path": tb_export_path.output if tb_export_path.status == TaskStatus.COMPLETED else None
                },
                "reports": {
                    "verification_report": verification_report.output if verification_report.status == TaskStatus.COMPLETED else None,
                    "debug_report": debug_report.output if debug_report.status == TaskStatus.COMPLETED else None
                },
                "agent_performance": {
                    agent_name: {
                        "tasks_executed": len(agent.execution_history),
                        "success_rate": agent.get_success_rate(),
                        "execution_log": agent.save_execution_log()
                    }
                    for agent_name, agent in self.agents.items()
                }
            }
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 Agentic FIFO Design Flow Completed Successfully!")
            self.logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
            self.logger.info(f"🔄 Iterations: {session_summary['total_iterations']}")
            self.logger.info(f"📈 Success rate: {session_summary['final_success_rate']:.1%}")
            self.logger.info(f"📁 Output directory: {self.output_dir}")
            self.logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Design flow failed: {e}")
            
            # Generate failure report
            failure_results = {
                "success": False,
                "error": str(e),
                "total_time": time.time() - start_time,
                "iterations": self.current_iteration,
                "agent_performance": {
                    agent_name: {
                        "tasks_executed": len(agent.execution_history),
                        "success_rate": agent.get_success_rate()
                    }
                    for agent_name, agent in self.agents.items()
                }
            }
            
            return failure_results
    
    def demonstrate_agentic_capabilities(self):
        """Demonstrate key agentic AI capabilities"""
        
        self.logger.info("\\n🤖 Demonstrating Agentic AI Capabilities:")
        
        # Multi-agent collaboration
        self.logger.info("✅ Multi-Agent Collaboration:")
        for agent_name, agent in self.agents.items():
            capabilities = agent.get_capabilities()
            self.logger.info(f"   • {agent_name}: {', '.join(capabilities[:3])}...")
        
        # Autonomous problem decomposition  
        self.logger.info("✅ Autonomous Problem Decomposition:")
        self.logger.info("   • Natural language → formal specification")
        self.logger.info("   • Specification → synthesizable RTL")
        self.logger.info("   • RTL → comprehensive verification")
        self.logger.info("   • Issues → corrective actions")
        
        # Self-correction loops
        self.logger.info("✅ Self-Correction and Iterative Refinement:")
        self.logger.info(f"   • Up to {self.max_iterations} autonomous iterations")
        self.logger.info("   • Issue analysis and root cause identification")
        self.logger.info("   • Targeted fixes based on failure analysis")
        
        # Decision tracking
        self.logger.info("✅ Decision Tracking and Explainability:")
        self.logger.info("   • Complete execution history logging")
        self.logger.info("   • Issue classification and fix strategies")
        self.logger.info("   • Convergence tracking and recommendations")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Agentic AI-Powered FIFO Design and Verification")
    parser.add_argument("--requirement", "-r", type=str, 
                       default="Design a parameterizable asynchronous FIFO with configurable depth and data width, including Gray code pointers for clock domain crossing safety",
                       help="Natural language FIFO requirement")
    parser.add_argument("--output", "-o", type=str, default="./reports",
                       help="Output directory for generated files")
    parser.add_argument("--max-iterations", "-i", type=int, default=5,
                       help="Maximum number of self-correction iterations")
    parser.add_argument("--demo", action="store_true",
                       help="Run demonstration mode showing agentic capabilities")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize the agentic EDA system
        fifo_eda = FIFOAgenticEDA(
            output_dir=args.output,
            max_iterations=args.max_iterations
        )
        
        if args.demo:
            fifo_eda.demonstrate_agentic_capabilities()
        
        # Run the complete design flow
        results = fifo_eda.run_agentic_fifo_design(args.requirement)
        
        # Print summary
        print("\\n" + "=" * 60)
        print("🎯 AGENTIC FIFO EDA RESULTS")
        print("=" * 60)
        print(f"Success: {'✅ YES' if results['success'] else '❌ NO'}")
        print(f"Total Time: {results['total_time']:.2f} seconds")
        print(f"Iterations: {results['iterations']}")
        if 'final_success_rate' in results:
            print(f"Success Rate: {results['final_success_rate']:.1%}")
        print(f"Output Directory: {args.output}")
        
        if results["success"]:
            print("\\n📁 Generated Artifacts:")
            if results.get("specification", {}).get("export_path"):
                print(f"   • Specification: {results['specification']['export_path']}")
            if results.get("rtl", {}).get("export_path"):
                print(f"   • RTL Code: {results['rtl']['export_path']}")
            if results.get("verification", {}).get("export_path"):
                print(f"   • Testbench: {results['verification']['export_path']}")
        
        print("=" * 60)
        
        # Exit with appropriate code
        sys.exit(0 if results["success"] else 1)
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()