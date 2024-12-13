import subprocess
import os
import time
from datetime import timedelta

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
RESET = '\033[0m'

# subprocess.run(["python", "primalgo.py"])
out_folder = os.getcwd() + "/GISFiles/output"
# # Create the Output Directory if It Doesn't Exist
# os.makedirs((out_folder), exist_ok=True)

runs = 10

total_time=0

for x in range(runs):
    os.makedirs((out_folder), exist_ok=True)
    dir = os.listdir(out_folder)
    if len(dir) > 0:
        print('Please clear output directories')
        exit()
    start = time.time()
    subprocess.run(["python", "corridor.py"])
    subprocess.run(["python", "genetic-selection.py"])
    subprocess.run(["python", "corridor2.py"])
    subprocess.run(["python", "genetic-selection-line2.py"])
    subprocess.run(["python", "corridor3.py"])
    subprocess.run(["python", "genetic-selection-line3.py"])
    subprocess.run(["python", "fitness-graph.py"])
    subprocess.run(["python", "linealgo.py"])
    subprocess.run(["python", "make-corridor-plot.py"])
    subprocess.run(["python", "make-lines-plot.py"])
    end = time.time()
    elapsed_time = int(end-start)
    estimated_time = elapsed_time*(runs-1-x)
    total_time+=elapsed_time
    print(f"{YELLOW}Elapsed Time: {str(timedelta(seconds=total_time))}")
    print(f"{YELLOW}Estimated remaining Time: {str(timedelta(seconds=estimated_time))}{RESET}")
    os.rename(out_folder, out_folder+str(x+1))