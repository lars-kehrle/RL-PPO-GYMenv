import logging
logging.basicConfig(
    level=logging.INFO,                      # Minimum level to capture
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",  # Log format
    handlers=[
        logging.StreamHandler()              # Output to console
    ]
)
