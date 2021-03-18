from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pdb
import inspect
import faulthandler
faulthandler.enable()

print(sys.executable)

inspect.getmembers(plt)
for i in tqdm(range(500)):
    i = 0
