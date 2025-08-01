#!/usr/bin/env python3
"""
Agentic AI-Powered FIFO Design and Verification System
End-to-End Demonstration

This script demonstrates the complete autonomous EDA workflow
with multi-agent collaboration for FIFO design and verification.
"""

import sys
import time
import logging
from pathlib import Path

# Import the main orchestration system
from main import FIFOAgenticEDA


def setup_demo_logging():
    """Setup logging for the demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('demo.log')
        ]
    )
    return logging.getLogger("AgenticEDADemo")


def print_banner():
    """Print demo banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🤖 AGENTIC AI-POWERED EDA SYSTEM 🤖                       ║
║                                                                              ║
║              Autonomous FIFO Design and Verification Workflow               ║
║                                                                              ║
║  🎯 Multi-Agent Collaboration  🔄 Self-Correction Loops                      ║
║  ⚙️  RTL Generation            🧪 Comprehensive Verification                 ║
║  🐛 Autonomous Debugging      📊 Intelligent Reporting                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def demonstrate_requirements():
    """Demonstrate different FIFO requirements"""
    requirements = [
        {
            "name": "Basic Asynchronous FIFO",
            "description": "Design a parameterizable asynchronous FIFO with configurable depth and data width, including Gray code pointers for clock domain crossing safety",
            "complexity": "Medium"
        },
        {
            "name": "High-Performance FIFO",
            "description": "Create a high-performance asynchronous FIFO with almost-full and almost-empty flags, supporting burst transfers and optimized for minimal latency",
            "complexity": "High"
        },
        {
            "name": "Safety-Critical FIFO",
            "description": "Design a fault-tolerant asynchronous FIFO with error detection, correction capabilities, and comprehensive self-checking mechanisms",
            "complexity": "Very High"
        },
        {
            "name": "Simple Synchronous FIFO", 
            "description": "Implement a basic synchronous FIFO with standard read/write interfaces and depth monitoring",
            "complexity": "Low"
        }
    ]
    
    print("\n📋 AVAILABLE FIFO DESIGN REQUIREMENTS:")
    print("=" * 60)
    
    for i, req in enumerate(requirements, 1):
        print(f"{i}. {req['name']} (Complexity: {req['complexity']})")
        print(f"   {req['description']}")
        print()
    
    return requirements


def demonstrate_agentic_capabilities(fifo_eda):
    """Demonstrate key agentic AI capabilities"""
    print("\n🤖 AGENTIC AI CAPABILITIES DEMONSTRATION")
    print("=" * 50)
    
    fifo_eda.demonstrate_agentic_capabilities()
    
    print("\n🔧 TOOL INTEGRATION STATUS:")
    print("=" * 30)
    
    # Import tool checking functions
    from tools import get_available_simulators, get_available_synthesizers, get_available_viewers
    
    simulators = get_available_simulators()
    synthesizers = get_available_synthesizers()
    viewers = get_available_viewers()
    
    print(f"📡 Simulators:     {simulators if simulators else 'None detected (framework ready)'}")
    print(f"⚙️  Synthesizers:   {synthesizers if synthesizers else 'None detected (framework ready)'}")
    print(f"📊 Waveform Viewers: {viewers if viewers else 'None detected (framework ready)'}")
    
    if not (simulators or synthesizers or viewers):
        print("\n💡 NOTE: No EDA tools detected, but the framework is ready!")
        print("   Install tools like: iverilog, yosys, gtkwave for full functionality")


def run_interactive_demo():
    """Run interactive demo allowing user to select requirements"""
    logger = setup_demo_logging()
    
    print_banner()
    
    # Initialize the agentic EDA system
    print("🚀 Initializing Agentic EDA System...")
    fifo_eda = FIFOAgenticEDA(
        output_dir="./demo_reports",
        max_iterations=3  # Reduced for demo
    )
    
    # Demonstrate capabilities
    demonstrate_agentic_capabilities(fifo_eda)
    
    # Show available requirements
    requirements = demonstrate_requirements()
    
    # Interactive selection
    print("🎯 SELECT A FIFO DESIGN REQUIREMENT:")
    print("Enter the number (1-4) or 'q' to quit:")
    
    while True:
        try:
            choice = input("\n>>> ").strip().lower()
            
            if choice == 'q':
                print("👋 Goodbye!")
                return 0
            
            req_index = int(choice) - 1
            if 0 <= req_index < len(requirements):
                selected_req = requirements[req_index]
                break
            else:
                print("❌ Invalid selection. Please enter 1-4 or 'q'.")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number 1-4 or 'q'.")
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user.")
            return 130
    
    # Run the selected design flow
    print(f"\n🎯 RUNNING: {selected_req['name']}")
    print("=" * 60)
    print(f"Requirement: {selected_req['description']}")
    print(f"Complexity: {selected_req['complexity']}")
    print()
    
    print("🚀 Starting Autonomous Design Flow...")
    print("   This may take a few moments...")
    print()
    
    start_time = time.time()
    
    try:
        # Execute the agentic design flow
        results = fifo_eda.run_agentic_fifo_design(selected_req['description'])
        
        total_time = time.time() - start_time
        
        # Display results
        print("\n" + "=" * 70)
        print("🎉 AUTONOMOUS DESIGN FLOW COMPLETED!")
        print("=" * 70)
        
        print(f"✅ Success: {'YES' if results['success'] else 'NO'}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🔄 Iterations: {results.get('iterations', 0)}")
        
        if 'final_success_rate' in results:
            print(f"📈 Success Rate: {results['final_success_rate']:.1%}")
        
        print(f"📁 Output Directory: ./demo_reports")
        
        # Show agent performance
        if 'agent_performance' in results:
            print("\n🤖 AGENT PERFORMANCE SUMMARY:")
            for agent_name, perf in results['agent_performance'].items():
                print(f"   {agent_name}: {perf['tasks_executed']} tasks, {perf['success_rate']:.1%} success rate")
        
        # Show generated artifacts
        if results.get('success'):
            print("\n📄 GENERATED ARTIFACTS:")
            if results.get('specification', {}).get('export_path'):
                print(f"   📋 Specification: {results['specification']['export_path']}")
            if results.get('rtl', {}).get('export_path'):
                print(f"   ⚙️  RTL Code: {results['rtl']['export_path']}")
            if results.get('verification', {}).get('export_path'):
                print(f"   🧪 Testbench: {results['verification']['export_path']}")
            
            # Show RTL quality metrics
            rtl_info = results.get('rtl', {})
            if rtl_info:
                print(f"\n📊 RTL QUALITY METRICS:")
                print(f"   Syntax Valid: {'✅ YES' if rtl_info.get('syntax_valid') else '❌ NO'}")
                print(f"   Warnings: {rtl_info.get('warnings', 0)}")
                print(f"   Errors: {rtl_info.get('errors', 0)}")
            
            # Show verification metrics
            verify_info = results.get('verification', {})
            if verify_info:
                print(f"\n🧪 VERIFICATION METRICS:")
                print(f"   Testbench Generated: {'✅ YES' if verify_info.get('testbench_generated') else '❌ NO'}")
                print(f"   Simulations Run: {verify_info.get('simulations_run', 0)}")
        
        print("\n" + "=" * 70)
        
        # Ask if user wants to try another requirement
        print("\n🔄 Would you like to try another FIFO design? (y/n):")
        if input(">>> ").strip().lower().startswith('y'):
            return run_interactive_demo()
        else:
            print("🎉 Thank you for trying the Agentic EDA System!")
            return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Design flow interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed with error: {e}")
        return 1


def run_batch_demo():
    """Run batch demo with all requirements"""
    logger = setup_demo_logging()
    
    print_banner()
    
    print("🚀 RUNNING BATCH DEMONSTRATION")
    print("Testing all FIFO requirements automatically...")
    print()
    
    # Initialize the agentic EDA system
    fifo_eda = FIFOAgenticEDA(
        output_dir="./batch_demo_reports",
        max_iterations=2  # Reduced for batch demo
    )
    
    # Get all requirements
    requirements = demonstrate_requirements()
    
    results_summary = []
    
    for i, req in enumerate(requirements, 1):
        print(f"\n{'='*60}")
        print(f"🎯 TEST {i}/{len(requirements)}: {req['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            results = fifo_eda.run_agentic_fifo_design(req['description'])
            test_time = time.time() - start_time
            
            results_summary.append({
                'name': req['name'],
                'success': results['success'],
                'time': test_time,
                'iterations': results.get('iterations', 0),
                'complexity': req['complexity']
            })
            
            status = "✅ SUCCESS" if results['success'] else "❌ FAILED"
            print(f"Result: {status} in {test_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")
            results_summary.append({
                'name': req['name'],
                'success': False,
                'time': time.time() - start_time,
                'iterations': 0,
                'complexity': req['complexity'],
                'error': str(e)
            })
    
    # Print final batch summary
    print(f"\n{'='*70}")
    print("🎉 BATCH DEMONSTRATION COMPLETED")
    print(f"{'='*70}")
    
    successful_tests = sum(1 for r in results_summary if r['success'])
    total_tests = len(results_summary)
    
    print(f"\n📊 OVERALL RESULTS: {successful_tests}/{total_tests} tests passed")
    print(f"⏱️  Total Time: {sum(r['time'] for r in results_summary):.2f} seconds")
    print()
    
    for result in results_summary:
        status = "✅" if result['success'] else "❌"
        error_info = f" ({result.get('error', '')[:50]}...)" if 'error' in result else ""
        print(f"{status} {result['name']} - {result['time']:.1f}s - {result['complexity']}{error_info}")
    
    return 0 if successful_tests == total_tests else 1


def main():
    """Main demo entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Agentic AI-Powered EDA System Demo")
    parser.add_argument("--batch", action="store_true", help="Run batch demo with all requirements")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        if args.batch:
            return run_batch_demo()
        else:
            return run_interactive_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
        return 130
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())