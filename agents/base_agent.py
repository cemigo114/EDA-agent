"""
Base Agent class for the Agentic FIFO EDA System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time
import json
from pathlib import Path


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskResult:
    """Result of an agent task execution"""
    agent_name: str
    task_name: str
    status: TaskStatus
    output: Any
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentConfig:
    """Configuration for agent behavior"""
    name: str
    description: str
    max_iterations: int = 5
    timeout_seconds: int = 300
    log_level: str = "INFO"
    output_dir: Path = Path("./reports")
    enable_self_correction: bool = True


class BaseAgent(ABC):
    """
    Abstract base class for all EDA agents
    
    Provides common functionality:
    - Logging and error handling
    - Task execution tracking
    - Configuration management
    - Inter-agent communication
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.execution_history: List[TaskResult] = []
        self.current_iteration = 0
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.config.name}: {self.config.description}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"Agent.{self.config.name}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = self.config.output_dir / f"{self.config.name.lower()}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(getattr(logging, self.config.log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def execute_task(self, task_name: str, **kwargs) -> TaskResult:
        """
        Execute a task with error handling and logging
        """
        start_time = time.time()
        self.logger.info(f"Starting task: {task_name}")
        
        try:
            # Execute the actual task
            result = self._execute_task_impl(task_name, **kwargs)
            
            execution_time = time.time() - start_time
            task_result = TaskResult(
                agent_name=self.config.name,
                task_name=task_name,
                status=TaskStatus.COMPLETED,
                output=result,
                execution_time=execution_time
            )
            
            self.logger.info(f"Task {task_name} completed in {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task {task_name} failed: {str(e)}"
            self.logger.error(error_msg)
            
            task_result = TaskResult(
                agent_name=self.config.name,
                task_name=task_name,
                status=TaskStatus.FAILED,
                output=None,
                error_message=error_msg,
                execution_time=execution_time
            )
        
        # Store execution history
        self.execution_history.append(task_result)
        return task_result
    
    @abstractmethod
    def _execute_task_impl(self, task_name: str, **kwargs) -> Any:
        """
        Implement the actual task execution logic
        Must be overridden by concrete agent classes
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of tasks this agent can perform"""
        return []
    
    def reset_state(self):
        """Reset agent state for new design iteration"""
        self.execution_history.clear()
        self.current_iteration = 0
        self.logger.info(f"Reset {self.config.name} state")
    
    def save_execution_log(self) -> Path:
        """Save execution history to JSON file"""
        log_file = self.config.output_dir / f"{self.config.name.lower()}_execution.json"
        
        # Convert TaskResults to serializable format
        history_data = []
        for result in self.execution_history:
            data = {
                "agent_name": result.agent_name,
                "task_name": result.task_name,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "error_message": result.error_message,
                "metadata": result.metadata,
                "output_summary": str(result.output)[:500] if result.output else None
            }
            history_data.append(data)
        
        with open(log_file, 'w') as f:
            json.dump({
                "agent": self.config.name,
                "total_tasks": len(self.execution_history),
                "successful_tasks": len([r for r in self.execution_history if r.status == TaskStatus.COMPLETED]),
                "failed_tasks": len([r for r in self.execution_history if r.status == TaskStatus.FAILED]),
                "execution_history": history_data
            }, f, indent=2)
        
        self.logger.info(f"Saved execution log to {log_file}")
        return log_file
    
    def get_last_result(self) -> Optional[TaskResult]:
        """Get the result of the last executed task"""
        return self.execution_history[-1] if self.execution_history else None
    
    def get_success_rate(self) -> float:
        """Calculate task success rate"""
        if not self.execution_history:
            return 0.0
        
        successful = len([r for r in self.execution_history if r.status == TaskStatus.COMPLETED])
        return successful / len(self.execution_history)
    
    def __str__(self) -> str:
        return f"{self.config.name}({len(self.execution_history)} tasks, {self.get_success_rate():.1%} success)"