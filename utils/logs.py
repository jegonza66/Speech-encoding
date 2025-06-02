from datetime import datetime
from pathlib import Path
import logging
import sys

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset color
        'BOLD': '\033[1m',        # Bold
        'DIM': '\033[2m'          # Dim
    }
    
    def format(self, record):
        # Get the original message
        original_format = super().format(record)
        
        # Add color based on log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        bold = self.COLORS['BOLD']
        
        # Format: [TIMESTAMP] [LEVEL] [FILE:LINE] MESSAGE
        colored_format = f"{color}{bold}[{record.levelname}]{reset} {color}{original_format}{reset}"
        
        return colored_format


def setup_logger(
    name: str = "speech_encoding",
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    console_output: bool = True
) -> logging.Logger:
    """
    Setup a logger with colored console output and optional file logging
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to save logs to file
        log_dir: Directory to save log files
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_file = Path(log_dir) / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get an existing logger or create a new one
    
    Args:
        name: Logger name (if None, uses the calling module name)
    
    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module name
        frame = sys._getframe(1)
        name = Path(frame.f_globals.get('__file__', 'unknown')).stem
    
    # Check if logger already exists
    if name in logging.Logger.manager.loggerDict:
        return logging.getLogger(name)
    
    # Create new logger with default settings
    return setup_logger(name)


def set_global_log_level(level: str):
    """
    Set the log level for all existing loggers
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper())
    
    # Set for all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)
        
        # Update handlers
        for handler in logger.handlers:
            handler.setLevel(numeric_level)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time
    
    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            pass
    """
    def wrapper(*args, **kwargs):
        logger = get_logger()
        func_name = func.__name__
        
        # Log function entry
        logger.debug(f"🔄 Calling {func_name} with args={args}, kwargs={kwargs}")
        
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.debug(f"✅ {func_name} completed in {duration:.3f}s")
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"❌ {func_name} failed after {duration:.3f}s: {str(e)}")
            raise
    
    return wrapper


def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        logger = get_logger()
        logger.debug(f"💾 Memory usage: {memory_mb:.1f} MB")
        
    except ImportError:
        pass  # psutil not available


def log_progress(current: int, total: int, message: str = "Progress"):
    """
    Log progress with percentage
    
    Args:
        current: Current iteration
        total: Total iterations
        message: Progress message
    """
    logger = get_logger()
    percentage = (current / total) * 100
    
    # Use different log levels based on progress
    if percentage < 25:
        logger.info(f"📊 {message}: {current}/{total} ({percentage:.1f}%)")
    elif percentage < 75:
        logger.info(f"📈 {message}: {current}/{total} ({percentage:.1f}%)")
    else:
        logger.info(f"🎯 {message}: {current}/{total} ({percentage:.1f}%)")


# # Example usage and testing
# if __name__ == "__main__":
#     # Setup logger with different levels
#     logger = setup_logger("test_logger", level="DEBUG")
    
#     # Test different log levels
#     logger.debug("🔍 This is a debug message")
#     logger.info("ℹ️ This is an info message")
#     logger.warning("⚠️ This is a warning message")
#     logger.error("❌ This is an error message")
#     logger.critical("🚨 This is a critical message")
    
#     # Test progress logging
#     for i in range(0, 101, 25):
#         log_progress(i, 100, "Training")
    
#     # Test function decorator
#     @log_function_call
#     def example_function(x, y=10):
#         import time
#         time.sleep(0.1)  # Simulate work
#         return x + y
    
#     result = example_function(5, y=15)
#     logger.info(f"Result: {result}")
    
#     # Test memory logging
#     log_memory_usage()
# else:
#     # Make config logger
#     import config

#     # Global logger configuration
#     LOG_LEVEL = getattr(config, 'LOG_LEVEL', 'INFO')  # Default to INFO if not in config
#     LOG_TO_FILE = getattr(config, 'LOG_TO_FILE', True)
#     LOG_DIR = getattr(config, 'LOG_DIR', 'save/detailed_logs')

#     # Setup main logger
#     main_logger = setup_logger(
#         name="speech_encoding",
#         level=LOG_LEVEL,
#         log_to_file=LOG_TO_FILE,
#         log_dir=LOG_DIR
#     )

#     # Export for easy import
#     logger = main_logger
