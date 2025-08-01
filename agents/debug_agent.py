"""
DebugAgent: Issue Analysis and Orchestration

Analyzes failures from code generation and verification, identifies root causes,
and orchestrates design iteration loops for autonomous self-correction.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from pathlib import Path

from .base_agent import BaseAgent, AgentConfig
from .spec_agent import FIFOSpecification
from .code_agent import CodeGenerationResult
from .verify_agent import SimulationResult, TestResult, CoverageReport


class IssueType(Enum):
    SYNTAX_ERROR = "syntax_error"
    LOGIC_ERROR = "logic_error"
    TIMING_ERROR = "timing_error"
    FUNCTIONAL_ERROR = "functional_error"
    COVERAGE_GAP = "coverage_gap"
    SPECIFICATION_ERROR = "specification_error"
    TESTBENCH_ERROR = "testbench_error"


class IssueSeverity(Enum):
    CRITICAL = "critical"     # Blocks compilation/simulation
    HIGH = "high"            # Causes functional failures
    MEDIUM = "medium"        # Causes warnings or coverage gaps
    LOW = "low"             # Minor improvements needed


@dataclass
class Issue:
    """Identified issue in the design or verification"""
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    location: Optional[str] = None
    suggested_fix: Optional[str] = None
    affected_component: Optional[str] = None  # spec, code, testbench
    iteration_detected: int = 0


@dataclass
class DebugSession:
    """Debug session tracking multiple issues and iterations"""
    session_id: str
    issues: List[Issue]
    iterations: int
    max_iterations: int = 5
    convergence_threshold: float = 0.95  # Success rate threshold
    current_success_rate: float = 0.0


@dataclass
class IterationPlan:
    """Plan for next iteration to fix identified issues"""
    iteration_number: int
    priority_issues: List[Issue]
    actions: List[Dict[str, Any]]  # Actions for each agent
    expected_improvements: str
    estimated_success_rate: float


class DebugAgent(BaseAgent):
    """
    Agent responsible for debugging and orchestrating design iterations
    
    Key capabilities:
    - Analyze compilation and simulation failures
    - Identify root causes of issues
    - Plan corrective actions for other agents
    - Orchestrate multi-iteration design loops
    - Track convergence and success metrics
    - Generate debug reports and recommendations
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.current_session: Optional[DebugSession] = None
        self.issue_patterns = self._load_issue_patterns()
        self.fix_strategies = self._load_fix_strategies()
        self.iteration_history: List[Dict[str, Any]] = []
    
    def get_capabilities(self) -> List[str]:
        return [
            "analyze_compilation_errors",
            "analyze_simulation_failures", 
            "identify_coverage_gaps",
            "diagnose_timing_issues",
            "plan_iteration",
            "orchestrate_fixes",
            "track_convergence",
            "generate_debug_report"
        ]
    
    def _execute_task_impl(self, task_name: str, **kwargs) -> Any:
        """Execute debug and orchestration tasks"""
        if task_name == "analyze_compilation_errors":
            return self._analyze_compilation_errors(kwargs.get("code_result"))
        elif task_name == "analyze_simulation_failures":
            return self._analyze_simulation_failures(kwargs.get("sim_results", []))
        elif task_name == "identify_coverage_gaps":
            return self._identify_coverage_gaps(kwargs.get("coverage_report"))
        elif task_name == "diagnose_timing_issues":
            return self._diagnose_timing_issues(kwargs.get("timing_data", {}))
        elif task_name == "plan_iteration":
            return self._plan_iteration()
        elif task_name == "orchestrate_fixes":
            return self._orchestrate_fixes(kwargs.get("iteration_plan"))
        elif task_name == "track_convergence":
            return self._track_convergence(kwargs.get("results"))
        elif task_name == "generate_debug_report":
            return self._generate_debug_report()
        else:
            raise ValueError(f"Unknown task: {task_name}")
    
    def _load_issue_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying common issues"""
        return {
            "syntax_errors": [
                r"syntax error.*near.*",
                r"unexpected.*token",
                r"missing.*semicolon",
                r"unmatched.*parentheses",
                r"undefined.*identifier"
            ],
            "logic_errors": [
                r"data mismatch",
                r"flag.*incorrect",
                r"overflow.*detected",
                r"underflow.*detected",
                r"gray code.*violation"
            ],
            "timing_errors": [
                r"setup.*violation",
                r"hold.*violation",  
                r"metastability",
                r"clock.*domain.*crossing"
            ],
            "functional_errors": [
                r"assertion.*failed",
                r"test.*failed",
                r"incorrect.*behavior",
                r"protocol.*violation"
            ]
        }
    
    def _load_fix_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load fix strategies for different issue types"""
        return {
            "gray_code_violation": {
                "component": "code",
                "action": "fix_gray_code_generation",
                "description": "Fix Gray code pointer calculation",
                "priority": "high"
            },
            "metastability": {
                "component": "code", 
                "action": "add_synchronizer_stages",
                "description": "Add proper synchronizer flip-flops",
                "priority": "critical"
            },
            "flag_generation": {
                "component": "code",
                "action": "fix_flag_logic",
                "description": "Correct full/empty flag generation",
                "priority": "high"
            },
            "data_mismatch": {
                "component": "testbench",
                "action": "fix_reference_model", 
                "description": "Update reference model logic",
                "priority": "medium"
            },
            "coverage_gap": {
                "component": "testbench",
                "action": "add_test_scenarios",
                "description": "Add missing test coverage",
                "priority": "medium"
            },
            "syntax_error": {
                "component": "code",
                "action": "fix_syntax",
                "description": "Correct SystemVerilog syntax",
                "priority": "critical"
            }
        }
    
    def start_debug_session(self, session_id: str, max_iterations: int = 5) -> DebugSession:
        """Start a new debug session"""
        self.current_session = DebugSession(
            session_id=session_id,
            issues=[],
            iterations=0,
            max_iterations=max_iterations
        )
        
        self.logger.info(f"Started debug session: {session_id}")
        return self.current_session
    
    def _analyze_compilation_errors(self, code_result: CodeGenerationResult) -> List[Issue]:
        """Analyze compilation errors and warnings"""
        self.logger.info("Analyzing compilation errors")
        
        if not code_result:
            return []
        
        issues = []
        
        # Analyze errors
        for error in code_result.errors:
            issue_type = self._classify_error(error)
            severity = IssueSeverity.CRITICAL if issue_type == IssueType.SYNTAX_ERROR else IssueSeverity.HIGH
            
            issue = Issue(
                issue_type=issue_type,
                severity=severity,
                description=error,
                location=self._extract_location(error),
                suggested_fix=self._suggest_fix(error),
                affected_component="code",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        # Analyze warnings
        for warning in code_result.warnings:
            issue = Issue(
                issue_type=IssueType.SYNTAX_ERROR,
                severity=IssueSeverity.MEDIUM,
                description=warning,
                location=self._extract_location(warning),
                suggested_fix=self._suggest_fix(warning),
                affected_component="code",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        if self.current_session:
            self.current_session.issues.extend(issues)
        
        self.logger.info(f"Found {len(issues)} compilation issues")
        return issues
    
    def _analyze_simulation_failures(self, sim_results: List[SimulationResult]) -> List[Issue]:
        """Analyze simulation failures and extract issues"""
        self.logger.info(f"Analyzing {len(sim_results)} simulation results")
        
        issues = []
        
        for result in sim_results:
            if result.result in [TestResult.FAIL, TestResult.ERROR]:
                # Analyze errors
                for error in result.errors:
                    issue_type = self._classify_simulation_error(error)
                    severity = self._determine_severity(error, issue_type)
                    
                    issue = Issue(
                        issue_type=issue_type,
                        severity=severity,
                        description=f"Test {result.test_name}: {error}",
                        location=result.test_name,
                        suggested_fix=self._suggest_simulation_fix(error),
                        affected_component=self._identify_affected_component(error),
                        iteration_detected=self.current_session.iterations if self.current_session else 0
                    )
                    issues.append(issue)
            
            # Analyze warnings even for passing tests
            for warning in result.warnings:
                issue = Issue(
                    issue_type=IssueType.FUNCTIONAL_ERROR,
                    severity=IssueSeverity.LOW,
                    description=f"Test {result.test_name}: {warning}",
                    location=result.test_name,
                    suggested_fix=self._suggest_simulation_fix(warning),
                    affected_component="testbench",
                    iteration_detected=self.current_session.iterations if self.current_session else 0
                )
                issues.append(issue)
        
        if self.current_session:
            self.current_session.issues.extend(issues)
        
        self.logger.info(f"Found {len(issues)} simulation issues")
        return issues
    
    def _identify_coverage_gaps(self, coverage_report: CoverageReport) -> List[Issue]:
        """Identify coverage gaps and missing test scenarios"""
        self.logger.info("Identifying coverage gaps")
        
        if not coverage_report:
            return []
        
        issues = []
        
        # Check functional coverage
        if coverage_report.functional_coverage < 95.0:
            issue = Issue(
                issue_type=IssueType.COVERAGE_GAP,
                severity=IssueSeverity.MEDIUM,
                description=f"Functional coverage {coverage_report.functional_coverage:.1f}% below target (95%)",
                suggested_fix="Add more comprehensive test scenarios",
                affected_component="testbench",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        # Check code coverage
        if coverage_report.code_coverage < 90.0:
            issue = Issue(
                issue_type=IssueType.COVERAGE_GAP,
                severity=IssueSeverity.MEDIUM,
                description=f"Code coverage {coverage_report.code_coverage:.1f}% below target (90%)",
                suggested_fix="Exercise all code paths with additional tests",
                affected_component="testbench",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        # Check toggle coverage
        if coverage_report.toggle_coverage < 85.0:
            issue = Issue(
                issue_type=IssueType.COVERAGE_GAP,
                severity=IssueSeverity.LOW,
                description=f"Toggle coverage {coverage_report.toggle_coverage:.1f}% below target (85%)",
                suggested_fix="Add tests to toggle all signal states",
                affected_component="testbench",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        # Process specific coverage gaps
        for gap in coverage_report.coverage_gaps:
            issue = Issue(
                issue_type=IssueType.COVERAGE_GAP,
                severity=IssueSeverity.MEDIUM,
                description=gap,
                suggested_fix="Add targeted test for this coverage gap",
                affected_component="testbench",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        if self.current_session:
            self.current_session.issues.extend(issues)
        
        self.logger.info(f"Identified {len(issues)} coverage gaps")
        return issues
    
    def _diagnose_timing_issues(self, timing_data: Dict[str, Any]) -> List[Issue]:
        """Diagnose timing-related issues"""
        self.logger.info("Diagnosing timing issues")
        
        issues = []
        
        # Check for metastability issues
        if timing_data.get("metastability_warnings", 0) > 0:
            issue = Issue(
                issue_type=IssueType.TIMING_ERROR,
                severity=IssueSeverity.HIGH,
                description="Potential metastability in clock domain crossing",
                suggested_fix="Add proper synchronizer flip-flop chains",
                affected_component="code",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        # Check for setup/hold violations
        if timing_data.get("setup_violations", 0) > 0:
            issue = Issue(
                issue_type=IssueType.TIMING_ERROR,
                severity=IssueSeverity.HIGH,
                description="Setup time violations detected",
                suggested_fix="Review critical timing paths and add pipeline stages",
                affected_component="code",
                iteration_detected=self.current_session.iterations if self.current_session else 0
            )
            issues.append(issue)
        
        if self.current_session:
            self.current_session.issues.extend(issues)
        
        return issues
    
    def _classify_error(self, error_msg: str) -> IssueType:
        """Classify error message into issue type"""
        error_lower = error_msg.lower()
        
        # Check syntax patterns
        for pattern in self.issue_patterns["syntax_errors"]:
            if re.search(pattern, error_lower):
                return IssueType.SYNTAX_ERROR
        
        # Check logic patterns
        for pattern in self.issue_patterns["logic_errors"]:
            if re.search(pattern, error_lower):
                return IssueType.LOGIC_ERROR
        
        # Check timing patterns
        for pattern in self.issue_patterns["timing_errors"]:
            if re.search(pattern, error_lower):
                return IssueType.TIMING_ERROR
        
        # Default to syntax error
        return IssueType.SYNTAX_ERROR
    
    def _classify_simulation_error(self, error_msg: str) -> IssueType:
        """Classify simulation error message"""
        error_lower = error_msg.lower()
        
        if any(pattern in error_lower for pattern in ["data mismatch", "incorrect", "violation"]):
            return IssueType.FUNCTIONAL_ERROR
        elif any(pattern in error_lower for pattern in ["overflow", "underflow", "full", "empty"]):
            return IssueType.LOGIC_ERROR
        elif any(pattern in error_lower for pattern in ["gray code", "pointer"]):
            return IssueType.LOGIC_ERROR
        else:
            return IssueType.FUNCTIONAL_ERROR
    
    def _determine_severity(self, error_msg: str, issue_type: IssueType) -> IssueSeverity:
        """Determine severity of an issue"""
        error_lower = error_msg.lower()
        
        if "critical" in error_lower or "fatal" in error_lower:
            return IssueSeverity.CRITICAL
        elif issue_type == IssueType.SYNTAX_ERROR:
            return IssueSeverity.CRITICAL
        elif "error" in error_lower:
            return IssueSeverity.HIGH
        elif "warning" in error_lower:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _extract_location(self, message: str) -> Optional[str]:
        """Extract location information from error message"""
        # Look for line numbers
        line_match = re.search(r'line\s+(\d+)', message)
        if line_match:
            return f"line {line_match.group(1)}"
        
        # Look for module/task names
        module_match = re.search(r'in\s+(\w+)', message)
        if module_match:
            return module_match.group(1)
        
        return None
    
    def _suggest_fix(self, error_msg: str) -> Optional[str]:
        """Suggest fix based on error message"""
        error_lower = error_msg.lower()
        
        if "semicolon" in error_lower:
            return "Add missing semicolon"
        elif "parentheses" in error_lower:
            return "Check parentheses matching"
        elif "undefined" in error_lower:
            return "Define missing identifier"
        elif "gray code" in error_lower:
            return "Fix Gray code pointer calculation"
        elif "metastability" in error_lower:
            return "Add proper clock domain crossing synchronizers"
        else:
            return "Review and correct the identified issue"
    
    def _suggest_simulation_fix(self, error_msg: str) -> Optional[str]:
        """Suggest fix for simulation errors"""
        error_lower = error_msg.lower()
        
        if "data mismatch" in error_lower:
            return "Check FIFO data path and reference model"
        elif "flag mismatch" in error_lower:
            return "Review full/empty flag generation logic"
        elif "overflow" in error_lower:
            return "Fix write enable logic when FIFO is full"
        elif "underflow" in error_lower:
            return "Fix read enable logic when FIFO is empty"
        else:
            return "Debug the failing test scenario"
    
    def _identify_affected_component(self, error_msg: str) -> str:
        """Identify which component is affected by the error"""
        error_lower = error_msg.lower()
        
        if any(keyword in error_lower for keyword in ["testbench", "reference", "expected"]):
            return "testbench"
        elif any(keyword in error_lower for keyword in ["gray code", "pointer", "flag", "memory"]):
            return "code"
        elif any(keyword in error_lower for keyword in ["specification", "parameter"]):
            return "spec"
        else:
            return "code"  # Default assumption
    
    def _plan_iteration(self) -> IterationPlan:
        """Plan the next iteration to fix identified issues"""
        if not self.current_session:
            raise ValueError("No active debug session")
        
        self.logger.info("Planning next iteration")
        
        # Filter and prioritize issues
        critical_issues = [i for i in self.current_session.issues if i.severity == IssueSeverity.CRITICAL]
        high_issues = [i for i in self.current_session.issues if i.severity == IssueSeverity.HIGH]
        medium_issues = [i for i in self.current_session.issues if i.severity == IssueSeverity.MEDIUM]
        
        # Take top priority issues for this iteration
        priority_issues = critical_issues + high_issues[:3] + medium_issues[:2]
        
        # Plan actions for each affected component
        actions = []
        
        # Group issues by component
        code_issues = [i for i in priority_issues if i.affected_component == "code"]
        testbench_issues = [i for i in priority_issues if i.affected_component == "testbench"]
        spec_issues = [i for i in priority_issues if i.affected_component == "spec"]
        
        if code_issues:
            actions.append({
                "agent": "CodeAgent",
                "task": "refine_code",
                "issues": [i.description for i in code_issues],
                "expected_outcome": "Fix compilation and logic errors"
            })
        
        if testbench_issues:
            actions.append({
                "agent": "VerifyAgent", 
                "task": "create_test_scenarios",
                "issues": [i.description for i in testbench_issues],
                "expected_outcome": "Improve test coverage and fix verification issues"
            })
        
        if spec_issues:
            actions.append({
                "agent": "SpecAgent",
                "task": "validate_specification", 
                "issues": [i.description for i in spec_issues],
                "expected_outcome": "Correct specification parameters"
            })
        
        # Estimate success rate improvement
        current_critical = len(critical_issues)
        expected_critical_fixed = min(len(code_issues), current_critical)
        estimated_success_rate = min(0.95, self.current_session.current_success_rate + 
                                   (expected_critical_fixed * 0.2))
        
        iteration_plan = IterationPlan(
            iteration_number=self.current_session.iterations + 1,
            priority_issues=priority_issues,
            actions=actions,
            expected_improvements=f"Fix {len(priority_issues)} priority issues",
            estimated_success_rate=estimated_success_rate
        )
        
        self.logger.info(f"Planned iteration {iteration_plan.iteration_number} with {len(actions)} actions")
        
        return iteration_plan
    
    def _orchestrate_fixes(self, iteration_plan: IterationPlan) -> Dict[str, Any]:
        """Orchestrate fixes across multiple agents"""
        self.logger.info(f"Orchestrating fixes for iteration {iteration_plan.iteration_number}")
        
        if not self.current_session:
            raise ValueError("No active debug session")
        
        orchestration_result = {
            "iteration": iteration_plan.iteration_number,
            "actions_planned": len(iteration_plan.actions),
            "actions_executed": 0,
            "success": True,
            "results": {}
        }
        
        # Execute planned actions
        for action in iteration_plan.actions:
            try:
                self.logger.info(f"Executing action for {action['agent']}: {action['task']}")
                # In a real implementation, this would call the appropriate agent
                # For now, we simulate the action execution
                orchestration_result["results"][action["agent"]] = {
                    "task": action["task"],
                    "status": "executed",
                    "issues_addressed": len(action.get("issues", []))
                }
                orchestration_result["actions_executed"] += 1
                
            except Exception as e:
                self.logger.error(f"Failed to execute action for {action['agent']}: {e}")
                orchestration_result["success"] = False
                orchestration_result["results"][action["agent"]] = {
                    "task": action["task"],
                    "status": "failed",
                    "error": str(e)
                }
        
        # Update session
        self.current_session.iterations += 1
        
        # Record iteration history
        self.iteration_history.append({
            "iteration": iteration_plan.iteration_number,
            "issues_addressed": len(iteration_plan.priority_issues),
            "actions": iteration_plan.actions,
            "orchestration_result": orchestration_result
        })
        
        self.logger.info(f"Orchestration completed: {orchestration_result['actions_executed']}/{orchestration_result['actions_planned']} actions executed")
        
        return orchestration_result
    
    def _track_convergence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Track convergence towards successful design"""
        if not self.current_session:
            raise ValueError("No active debug session")
        
        self.logger.info("Tracking convergence")
        
        # Calculate success metrics
        total_tests = results.get("total_tests", 0)
        passed_tests = results.get("passed_tests", 0)
        success_rate = passed_tests / max(total_tests, 1)
        
        # Update session
        self.current_session.current_success_rate = success_rate
        
        # Check convergence
        converged = (success_rate >= self.current_session.convergence_threshold and
                    len([i for i in self.current_session.issues if i.severity == IssueSeverity.CRITICAL]) == 0)
        
        max_iterations_reached = self.current_session.iterations >= self.current_session.max_iterations
        
        convergence_status = {
            "converged": converged,
            "success_rate": success_rate,
            "target_success_rate": self.current_session.convergence_threshold,
            "iterations": self.current_session.iterations,
            "max_iterations": self.current_session.max_iterations,
            "max_iterations_reached": max_iterations_reached,
            "critical_issues_remaining": len([i for i in self.current_session.issues if i.severity == IssueSeverity.CRITICAL]),
            "recommendation": self._get_convergence_recommendation(converged, max_iterations_reached, success_rate)
        }
        
        self.logger.info(f"Convergence status: {success_rate:.1%} success rate, {convergence_status['critical_issues_remaining']} critical issues")
        
        return convergence_status
    
    def _get_convergence_recommendation(self, converged: bool, max_iterations: bool, success_rate: float) -> str:
        """Get recommendation based on convergence status"""
        if converged:
            return "Design has converged successfully - ready for final verification"
        elif max_iterations:
            return "Maximum iterations reached - consider manual review or specification changes"
        elif success_rate < 0.5:
            return "Low success rate - consider fundamental design changes"
        else:
            return "Continue iterations - showing progress towards convergence"
    
    def _generate_debug_report(self) -> Dict[str, Any]:
        """Generate comprehensive debug report"""
        self.logger.info("Generating debug report")
        
        if not self.current_session:
            raise ValueError("No active debug session")
        
        # Categorize issues by type and severity
        issues_by_type = {}
        issues_by_severity = {}
        
        for issue in self.current_session.issues:
            # By type
            issue_type = issue.issue_type.value
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
            
            # By severity
            severity = issue.severity.value
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Generate summary statistics
        report = {
            "session_summary": {
                "session_id": self.current_session.session_id,
                "total_iterations": self.current_session.iterations,
                "max_iterations": self.current_session.max_iterations,
                "current_success_rate": self.current_session.current_success_rate,
                "convergence_threshold": self.current_session.convergence_threshold,
                "total_issues": len(self.current_session.issues)
            },
            "issue_analysis": {
                "by_type": {t: len(issues) for t, issues in issues_by_type.items()},
                "by_severity": {s: len(issues) for s, issues in issues_by_severity.items()},
                "critical_issues": [
                    {
                        "type": i.issue_type.value,
                        "description": i.description,
                        "location": i.location,
                        "suggested_fix": i.suggested_fix,
                        "iteration_detected": i.iteration_detected
                    }
                    for i in self.current_session.issues if i.severity == IssueSeverity.CRITICAL
                ]
            },
            "iteration_history": self.iteration_history,
            "recommendations": self._generate_final_recommendations(),
            "lessons_learned": self._extract_lessons_learned()
        }
        
        # Export report
        report_file = self.config.output_dir / "debug_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Debug report saved to {report_file}")
        
        return report
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations based on debug session"""
        recommendations = []
        
        if not self.current_session:
            return recommendations
        
        critical_issues = [i for i in self.current_session.issues if i.severity == IssueSeverity.CRITICAL]
        
        if len(critical_issues) > 0:
            recommendations.append(f"Address {len(critical_issues)} critical issues before proceeding")
        
        if self.current_session.current_success_rate < 0.9:
            recommendations.append("Improve test pass rate through additional debugging")
        
        if self.current_session.iterations >= self.current_session.max_iterations:
            recommendations.append("Consider increasing iteration limit or manual intervention")
        
        # Component-specific recommendations
        code_issues = [i for i in self.current_session.issues if i.affected_component == "code"]
        if len(code_issues) > 5:
            recommendations.append("Consider code refactoring to reduce complexity")
        
        testbench_issues = [i for i in self.current_session.issues if i.affected_component == "testbench"]
        if len(testbench_issues) > 3:
            recommendations.append("Enhance testbench with more comprehensive scenarios")
        
        return recommendations
    
    def _extract_lessons_learned(self) -> List[str]:
        """Extract lessons learned from the debug session"""
        lessons = []
        
        if not self.current_session:
            return lessons
        
        # Analyze iteration patterns
        if len(self.iteration_history) > 1:
            if all(iter_data["issues_addressed"] < 2 for iter_data in self.iteration_history):
                lessons.append("Small incremental fixes were more effective than large changes")
            
            syntax_iterations = sum(1 for iter_data in self.iteration_history 
                                  if any("syntax" in str(action).lower() for action in iter_data.get("actions", [])))
            if syntax_iterations > 1:
                lessons.append("Multiple syntax issues suggest need for better code templates")
        
        # Analyze issue types
        issue_types = [i.issue_type for i in self.current_session.issues]
        if issue_types.count(IssueType.LOGIC_ERROR) > issue_types.count(IssueType.SYNTAX_ERROR):
            lessons.append("Logic errors were more prevalent than syntax errors")
        
        return lessons
    
    def end_debug_session(self) -> Dict[str, Any]:
        """End the current debug session and return final summary"""
        if not self.current_session:
            raise ValueError("No active debug session")
        
        final_summary = {
            "session_id": self.current_session.session_id,
            "success": self.current_session.current_success_rate >= self.current_session.convergence_threshold,
            "final_success_rate": self.current_session.current_success_rate,
            "total_iterations": self.current_session.iterations,
            "total_issues": len(self.current_session.issues),
            "remaining_critical_issues": len([i for i in self.current_session.issues if i.severity == IssueSeverity.CRITICAL])
        }
        
        self.logger.info(f"Debug session ended: {final_summary}")
        
        # Reset for next session
        self.current_session = None
        
        return final_summary