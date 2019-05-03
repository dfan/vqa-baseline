import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str)
parser.add_argument('--output', '-o', type=str)
args = parser.parse_args()
args = vars(args)

input_dir = args['input']
output_dir = args['output']

# Plot loss from saved cluster output
f = open(input_dir, 'r')
losses = [float(num) for num in re.findall(r'Batch Loss: (.*)', f.read())]
iterations = [1 + 50*i for i in range(len(losses))]

# Plot loss and accuracy curves
plt.plot(iterations, losses, label='Cross-Entropy Loss')
plt.yticks(np.arange(0, max(losses), 0.5))
plt.legend()
plt.xlabel("Iterations", fontweight='bold')
plt.ylabel("Training Batch Loss", fontweight='bold')

if not os.path.exists('figures'):
  os.mkdir('figures')
plt.savefig('figures/{}'.format(output_dir), dpi=500)
plt.close()
