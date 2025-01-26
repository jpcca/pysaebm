import logging 

def setup_logging(log_file: str):
    """
    Set up logging to a file and console.
    Ensures logs are flushed immediately after each message.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove only the previous handlers while keeping the reference
    while logger.handlers:
        handler = logger.handlers[0]
        handler.close()  # Close the handler to ensure file is properly closed
        logger.removeHandler(handler)
    
    # Create a file handler
    file_handler = logging.FileHandler(log_file, mode='w')  # Use 'w' mode to start fresh
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and set it for both handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)