# written by copilot, 2026 05 05

def load_floats(filename):
    values = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                values.append(float(line))
    return values


def mean_absolute_error(a, b):
    n = min(len(a), len(b))
    if n == 0:
        raise ValueError("No comparable data found")

    mae = sum(abs(a[i] - b[i]) for i in range(n)) / n
    return mae, n


if __name__ == "__main__":
    result_file = "result.txt"
    ref_file = "simulation/questa/result_sim1.txt"
    #ref_file = "resultpy.txt"

    result_values = load_floats(result_file)
    ref_values = load_floats(ref_file)

    mae = mean_absolute_error(result_values, ref_values)

    print(f"MAE = {mae}")