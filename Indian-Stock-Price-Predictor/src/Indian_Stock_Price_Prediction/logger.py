import logging
import os
from datetime import datetime

# Create the log filename
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create the logs directory path (without the filename)
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Create the full file path
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

print(f"Log file created at: {LOG_FILE_PATH}")
