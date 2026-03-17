from src.Indian_Stock_Price_Prediction.logger import logging
from src.Indian_Stock_Price_Prediction.exception import CustomException

import sys
import os


if __name__ == "__main__":

    try:
        logging.info("Logger and Exception working normally.")


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys) from e


import yfinance as yf
print(yf.__version__)
  