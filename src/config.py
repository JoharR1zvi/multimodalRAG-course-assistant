import os
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
