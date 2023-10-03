import subprocess
import re

def run_training_and_capture_output(num_head, num_layer, max_iters=500):
    command = ["python3", "train.py", "config/train_shakespeare_char.py", f"--n_head={num_head}", f"--n_layer={num_layer}"]

    result = subprocess.run(command, text=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout = result.stdout  # This captures the standard output without the warnings

    if result.returncode != 0:
        print("Error encountered:")
        print(result.stderr)
        return None

    pattern = r"step (\d+): train loss (\d+\.\d+), val loss (\d+\.\d+)"
    matches = re.findall(pattern, stdout)

    data = {
        "num_head": num_head,
        "num_layer": num_layer,
        "losses": matches
    }

    return data

# Range of n_head and n_layer values you want to test
n_heads = [6, 8, 12, 16 ]
n_layers = [6, 8, 10, 12]

results = []

for n_head in n_heads:
    for n_layer in n_layers:
        result = run_training_and_capture_output(n_head, n_layer)
        print(result)
        results.append(result)

print(results)
