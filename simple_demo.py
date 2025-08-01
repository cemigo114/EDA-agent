#!/usr/bin/env python3
"""
Simplified Agentic AI-Powered EDA System Demo

Demonstrates the key components and capabilities of the system
without requiring full tool installation.
"""

import sys
import logging
from pathlib import Path

# Import individual agents
from agents.spec_agent import SpecAgent
from agents.code_agent import CodeAgent  
from agents.verify_agent import VerifyAgent
from agents.debug_agent import DebugAgent
from agents.base_agent import AgentConfig, TaskStatus

# Import tool integration
from tools import get_available_simulators, get_available_synthesizers, get_available_viewers


def setup_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("SimpleDemoLog")


def print_banner():
    """Print demo banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 AGENTIC AI-POWERED EDA SYSTEM 🤖                       ║
║                                                                              ║
║              Autonomous FIFO Design and Verification Workflow               ║
║                           SIMPLIFIED DEMONSTRATION                           ║
║                                                                              ║
║  🎯 Multi-Agent Collaboration  🔄 Self-Correction Loops                      ║
║  ⚙️  RTL Generation            🧪 Comprehensive Verification                 ║
║  🐛 Autonomous Debugging      📊 Intelligent Reporting                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def demonstrate_agent_capabilities():
    """Demonstrate individual agent capabilities"""
    print("\n🤖 AGENT CAPABILITIES DEMONSTRATION")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize agents
    agents = {}
    
    # SpecAgent
    spec_config = AgentConfig(
        name="SpecAgent",
        description="Natural language to formal specification parser",
        output_dir=output_dir / "spec"
    )
    agents["spec"] = SpecAgent(spec_config)
    
    # CodeAgent
    code_config = AgentConfig(
        name="CodeAgent",
        description="SystemVerilog RTL generator and refiner",
        output_dir=output_dir / "code"
    )
    agents["code"] = CodeAgent(code_config)
    
    # VerifyAgent
    verify_config = AgentConfig(
        name="VerifyAgent",
        description="Testbench generator and simulation executor",
        output_dir=output_dir / "verify"
    )
    agents["verify"] = VerifyAgent(verify_config)
    
    # DebugAgent
    debug_config = AgentConfig(
        name="DebugAgent",
        description="Issue analyzer and iteration orchestrator",
        output_dir=output_dir / "debug"
    )
    agents["debug"] = DebugAgent(debug_config)
    
    # Demonstrate each agent
    for agent_name, agent in agents.items():
        print(f"\n🔧 {agent_name} Capabilities:")
        capabilities = agent.get_capabilities()
        for i, capability in enumerate(capabilities, 1):
            print(f"   {i}. {capability.replace('_', ' ').title()}")
    
    return agents


def demonstrate_spec_agent(spec_agent):
    """Demonstrate SpecAgent functionality"""
    print(f"\n🎯 SPECIFICATION AGENT DEMONSTRATION")
    print("=" * 40)
    
    requirement = """Design a parameterizable asynchronous FIFO with configurable depth and data width, 
    including Gray code pointers for clock domain crossing safety. The FIFO should support 
    depths of 16, 32, 64, or 128 entries and data widths of 8, 16, or 32 bits."""
    
    print(f"Input Requirement:")
    print(f"'{requirement}'")
    print()
    
    try:
        # Parse natural language requirement
        result = spec_agent.execute_task("parse_natural_language", requirement_text=requirement)
        
        if result.status == TaskStatus.COMPLETED:
            spec = result.output
            print("✅ Specification Generated Successfully!")
            print(f"📋 Module Name: {spec.module_name}")
            print(f"📏 Data Width: {spec.parameters.data_width} bits")
            print(f"📦 Depth: {spec.parameters.depth} entries")
            print(f"🔗 Interface Signals: {len(spec.interface.input_signals + spec.interface.output_signals)} total")
            print(f"⚡ Clock Domains: {len(spec.interface.clock_signals)}")
            return spec
            
        else:
            print(f"❌ Specification failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error in specification: {e}")
        return None


def demonstrate_code_agent(code_agent, specification):
    """Demonstrate CodeAgent functionality"""
    print(f"\n⚙️  CODE GENERATION AGENT DEMONSTRATION")
    print("=" * 40)
    
    if not specification:
        print("⚠️  Skipping code generation - no specification available")
        return None
    
    try:
        # Generate RTL code
        result = code_agent.execute_task("generate_rtl", specification=specification)
        
        if result.status == TaskStatus.COMPLETED:
            code_result = result.output
            print("✅ RTL Generation Completed!")
            print(f"📄 Generated {len(code_result.generated_files)} files")
            print(f"✅ Syntax Valid: {code_result.syntax_valid}")
            print(f"⚠️  Warnings: {len(code_result.warnings)}")
            print(f"❌ Errors: {len(code_result.errors)}")
            
            # Show sample code snippet
            if code_result.generated_files:
                main_file = code_result.generated_files[0]
                print(f"\n📝 Sample from {main_file.filename}:")
                lines = main_file.content.split('\n')[:10]  # First 10 lines
                for i, line in enumerate(lines, 1):
                    print(f"   {i:2d}: {line}")
                if len(lines) < len(main_file.content.split('\n')):
                    print(f"   ... ({len(main_file.content.split('\n')) - len(lines)} more lines)")
            
            return code_result
            
        else:
            print(f"❌ Code generation failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error in code generation: {e}")
        return None


def demonstrate_verify_agent(verify_agent, specification):
    """Demonstrate VerifyAgent functionality"""
    print(f"\n🧪 VERIFICATION AGENT DEMONSTRATION")
    print("=" * 40)
    
    if not specification:
        print("⚠️  Skipping verification - no specification available")
        return None
    
    try:
        # Generate testbench
        result = verify_agent.execute_task("generate_testbench", specification=specification)
        
        if result.status == TaskStatus.COMPLETED:
            testbench = result.output
            print("✅ Testbench Generation Completed!")
            print(f"📋 Test Name: {testbench.name}")
            print(f"🎯 Test Scenarios: {len(testbench.test_scenarios)}")
            print(f"📊 Coverage Points: {len(testbench.coverage_points)}")
            
            # Show test scenarios
            print(f"\n🧪 Test Scenarios:")
            for i, scenario in enumerate(testbench.test_scenarios[:3], 1):  # Show first 3
                print(f"   {i}. {scenario.name}: {scenario.description}")
            if len(testbench.test_scenarios) > 3:
                print(f"   ... and {len(testbench.test_scenarios) - 3} more scenarios")
            
            return testbench
            
        else:
            print(f"❌ Testbench generation failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"❌ Error in verification: {e}")
        return None


def demonstrate_debug_agent(debug_agent):
    """Demonstrate DebugAgent functionality"""
    print(f"\n🐛 DEBUG AGENT DEMONSTRATION")
    print("=" * 40)
    
    try:
        # Start a debug session
        session = debug_agent.start_debug_session("demo_session", max_iterations=3)
        print(f"✅ Debug Session Started: {session.session_id}")
        
        # Simulate some issues for demonstration
        from agents.debug_agent import Issue, IssueType, IssueSeverity
        
        demo_issues = [
            Issue(
                issue_type=IssueType.SYNTAX_ERROR,
                severity=IssueSeverity.CRITICAL,
                description="Undefined identifier 'gray_code_ptr' in line 45",
                location="line 45",
                suggested_fix="Define gray_code_ptr variable or correct the reference",
                affected_component="code"
            ),
            Issue(
                issue_type=IssueType.LOGIC_ERROR,
                severity=IssueSeverity.HIGH,
                description="Empty flag incorrectly asserted during read operation",
                location="test_basic_read",
                suggested_fix="Review empty flag generation logic",
                affected_component="code"
            )
        ]
        
        # Add issues to session
        session.issues.extend(demo_issues)
        
        # Plan iteration
        result = debug_agent.execute_task("plan_iteration")
        
        if result.status == TaskStatus.COMPLETED:
            iteration_plan = result.output
            print(f"✅ Iteration Plan Generated!")
            print(f"🔄 Iteration: {iteration_plan.iteration_number}")
            print(f"🎯 Priority Issues: {len(iteration_plan.priority_issues)}")
            print(f"⚡ Planned Actions: {len(iteration_plan.actions)}")
            print(f"📈 Expected Success Rate: {iteration_plan.estimated_success_rate:.1%}")
            
            # Show planned actions
            print(f"\n📋 Planned Actions:")
            for i, action in enumerate(iteration_plan.actions, 1):
                print(f"   {i}. {action['agent']}: {action['task']}")
                print(f"      Expected: {action['expected_outcome']}")
        
        # End debug session
        summary = debug_agent.end_debug_session()
        print(f"\n📊 Debug Session Summary:")
        print(f"   Success: {'✅ YES' if summary['success'] else '❌ NO'}")
        print(f"   Total Issues: {summary['total_issues']}")
        print(f"   Iterations: {summary['total_iterations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in debugging: {e}")
        return False


def demonstrate_tool_integration():
    """Demonstrate tool integration capabilities"""
    print(f"\n🔧 TOOL INTEGRATION DEMONSTRATION")
    print("=" * 40)
    
    # Check available tools
    simulators = get_available_simulators()
    synthesizers = get_available_synthesizers()  
    viewers = get_available_viewers()
    
    print(f"📡 Available Simulators: {simulators if simulators else 'None (framework ready)'}")
    print(f"⚙️  Available Synthesizers: {synthesizers if synthesizers else 'None (framework ready)'}")
    print(f"📊 Available Waveform Viewers: {viewers if viewers else 'None (framework ready)'}")
    
    # Show tool integration status
    total_tools = len(simulators) + len(synthesizers) + len(viewers)
    
    if total_tools == 0:
        print(f"\n💡 TOOL INTEGRATION STATUS:")
        print(f"   🏗️  Framework: ✅ Ready")
        print(f"   🔧 EDA Tools: ⚠️  None detected")
        print(f"   📝 Recommendation: Install iverilog, yosys, gtkwave for full functionality")
        print(f"   🎯 Current Mode: Specification and design generation only")
    else:
        print(f"\n✅ TOOL INTEGRATION: {total_tools} tools ready for autonomous workflow!")
    
    return total_tools > 0


def main():
    """Main demo execution"""
    logger = setup_logging()
    
    print_banner()
    
    print("🚀 Starting Simplified Agentic EDA Demonstration...")
    print("   This demo showcases the autonomous AI agents without requiring EDA tools.")
    print()
    
    try:
        # Demonstrate agent capabilities
        agents = demonstrate_agent_capabilities()
        
        # Run agent demonstrations
        print(f"\n📋 RUNNING AGENT DEMONSTRATIONS")
        print("=" * 40)
        
        # SpecAgent demo
        specification = demonstrate_spec_agent(agents["spec"])
        
        # CodeAgent demo  
        code_result = demonstrate_code_agent(agents["code"], specification)
        
        # VerifyAgent demo
        testbench = demonstrate_verify_agent(agents["verify"], specification)
        
        # DebugAgent demo
        debug_success = demonstrate_debug_agent(agents["debug"])
        
        # Tool integration demo
        tools_available = demonstrate_tool_integration()
        
        # Final summary
        print(f"\n{'='*70}")
        print("🎉 SIMPLIFIED DEMONSTRATION COMPLETED!")
        print(f"{'='*70}")
        
        successes = sum([
            specification is not None,
            code_result is not None,
            testbench is not None,
            debug_success,
        ])
        
        print(f"✅ Agent Demonstrations: {successes}/4 successful")
        print(f"🔧 Tool Integration: {'Ready' if tools_available else 'Framework Ready'}")
        print(f"📁 Output Directory: ./demo_output")
        
        print(f"\n🤖 KEY AGENTIC AI CAPABILITIES DEMONSTRATED:")
        print(f"   • Natural language understanding and formal specification generation")
        print(f"   • Autonomous SystemVerilog RTL code generation")
        print(f"   • Comprehensive verification testbench creation")
        print(f"   • Intelligent issue analysis and iterative debugging")
        print(f"   • Multi-agent collaboration and orchestration")
        print(f"   • Tool integration framework (simulators, synthesizers, viewers)")
        
        if successes == 4:
            print(f"\n🎉 All agent capabilities demonstrated successfully!")
            return 0
        else:
            print(f"\n⚠️  {4-successes} agent demonstration(s) had issues")
            return 1
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Demo interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())