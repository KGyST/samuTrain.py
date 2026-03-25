# What is this file for: Thread-based interruption management for samuTrain training
# When it is created: 2026-03-25

import threading
import signal
import sys
import logging
from typing import Optional, Callable
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


class ShutdownState(Enum):
  """Enum for shutdown states"""
  RUNNING = "running"
  SHUTDOWN_REQUESTED = "shutdown_requested"
  SHUTTING_DOWN = "shutting_down"
  TERMINATED = "terminated"


class TrainingInterruptionManager:
  """Manages training interruption and graceful shutdown"""
  
  def __init__(self):
    self.shutdown_event = threading.Event()
    self.training_active = threading.Event()
    self.current_training_thread: Optional[threading.Thread] = None
    self.shutdown_state = ShutdownState.RUNNING
    self.shutdown_callbacks: list[Callable] = []
    self._lock = threading.Lock()
    
    # Setup signal handlers
    self._setup_signal_handlers()
  
  def _setup_signal_handlers(self):
    """Setup signal handlers for graceful shutdown"""
    try:
      # Unix-like systems
      signal.signal(signal.SIGINT, self._signal_handler)
      signal.signal(signal.SIGTERM, self._signal_handler)
    except (OSError, ValueError) as e:
      logger.warning(f"Could not setup signal handlers: {e}")
  
  def _signal_handler(self, signum, frame=None):
    """Handle signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    self.request_shutdown()
  
  def request_shutdown(self, reason: str = "User requested"):
    """Request graceful shutdown"""
    with self._lock:
      if self.shutdown_state == ShutdownState.RUNNING:
        self.shutdown_state = ShutdownState.SHUTDOWN_REQUESTED
        self.shutdown_event.set()
        logger.info(f"Shutdown requested: {reason}")
        
        # Call shutdown callbacks
        for callback in self.shutdown_callbacks:
          try:
            callback(reason)
          except Exception as e:
            logger.error(f"Error in shutdown callback: {e}")
        
        # Mark as shutting down
        self.shutdown_state = ShutdownState.SHUTTING_DOWN
  
  def is_shutdown_requested(self) -> bool:
    """Check if shutdown was requested"""
    return self.shutdown_event.is_set()
  
  def is_training_active(self) -> bool:
    """Check if training is currently active"""
    return self.training_active.is_set()
  
  def start_training(self, training_thread: threading.Thread):
    """Mark training as started and set thread reference"""
    with self._lock:
      if self.shutdown_state != ShutdownState.RUNNING:
        logger.warning("Cannot start training - shutdown in progress")
        return False
      
      self.current_training_thread = training_thread
      self.training_active.set()
      logger.info("Training started")
      return True
  
  def stop_training(self):
    """Mark training as stopped"""
    with self._lock:
      self.training_active.clear()
      self.current_training_thread = None
      
      if self.shutdown_state == ShutdownState.SHUTTING_DOWN:
        self.shutdown_state = ShutdownState.TERMINATED
        logger.info("Training stopped and shutdown completed")
      else:
        logger.info("Training stopped normally")
  
  def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
    """Wait for shutdown to complete"""
    return self.shutdown_event.wait(timeout)
  
  def add_shutdown_callback(self, callback: Callable[[str], None]):
    """Add callback to be called on shutdown"""
    with self._lock:
      self.shutdown_callbacks.append(callback)
  
  def force_terminate_training(self, timeout: float = 5.0) -> bool:
    """Force terminate training thread (last resort)"""
    if not self.current_training_thread or not self.current_training_thread.is_alive():
      return True
    
    logger.warning("Force terminating training thread...")
    
    # Wait for thread to finish naturally
    self.current_training_thread.join(timeout)
    
    if self.current_training_thread.is_alive():
      logger.error("Training thread did not terminate gracefully")
      # Note: Python doesn't support force killing threads safely
      # This is a limitation we have to live with
      return False
    
    return True
  
  def get_shutdown_state(self) -> ShutdownState:
    """Get current shutdown state"""
    return self.shutdown_state
  
  def reset(self):
    """Reset the interruption manager (for testing)"""
    with self._lock:
      self.shutdown_event.clear()
      self.training_active.clear()
      self.current_training_thread = None
      self.shutdown_state = ShutdownState.RUNNING
      self.shutdown_callbacks.clear()


# Global instance for application-wide use
_global_interruption_manager: Optional[TrainingInterruptionManager] = None


def get_interruption_manager() -> TrainingInterruptionManager:
  """Get the global interruption manager instance"""
  global _global_interruption_manager
  if _global_interruption_manager is None:
    _global_interruption_manager = TrainingInterruptionManager()
  return _global_interruption_manager


def reset_interruption_manager():
  """Reset the global interruption manager (for testing)"""
  global _global_interruption_manager
  if _global_interruption_manager:
    _global_interruption_manager.reset()
  _global_interruption_manager = None


# Context manager for training operations
class TrainingContext:
  """Context manager for training operations with interruption support"""
  
  def __init__(self, interruption_manager: Optional[TrainingInterruptionManager] = None):
    self.manager = interruption_manager or get_interruption_manager()
    self.training_thread = threading.current_thread()
  
  def __enter__(self):
    if not self.manager.start_training(self.training_thread):
      raise RuntimeError("Cannot start training - shutdown in progress")
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    self.manager.stop_training()
    
    # If we're exiting due to KeyboardInterrupt, request shutdown
    if exc_type is KeyboardInterrupt:
      self.manager.request_shutdown("Keyboard interrupt during training")
    
    return False  # Don't suppress exceptions
