import sys
import os

# Replace 'yourusername' with your PythonAnywhere username
path = '/home/Aayushkhadka/Payverify_Nepal'
if path not in sys.path:
    sys.path.append(path)

os.chdir(path)

from app import app as application