import numpy as np
from matplotlib import pyplot as plt
import timeit
import argparse
import os

LOG_MEAN = 0
LOG_STD = 1


def create_and_plot_data(mu, sigma, num_samples, outdir):
    global LOG_MEAN
    global LOG_STD

    data = np.random.normal(mu, sigma, num_samples)
    LOG_MEAN = round(data.mean(), 3)
    LOG_STD = round(data.std(), 3)

    plot_path = os.path.join(outdir, f'distribution_{num_samples}.png')
    # Plotting
    plt.figure(figsize=(9, 6))
    plt.grid()
    plt.title(f'Samples from $\mathcal{{N}}({mu}, {sigma})$ distribution')
    plt.xlabel('Sample Value')
    plt.ylabel('Count')
    plt.hist(data, bins=20 * int(np.log10(num_samples)),
             density=False, alpha=1, color='skyblue')
    plt.savefig(fname=plot_path)
    plt.cla()
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample and Plot normal distibution')
    parser.add_argument("--mean", type=float, default=0,
                        help='Mean (mu) of the distribution')
    parser.add_argument("--std", type=float, default=1,
                        help='Standard Deviation (sigma) of the distribution')
    parser.add_argument("--min-power", type=int, default=1,
                        help='Minimum exponent (start from 10^min-power)')
    parser.add_argument("--max-power", type=int, default=8,
                        help='Maximum exponent (end at 10^max-power)')
    parser.add_argument("--out", type=str, default='test',
                        help='Output directory for plots')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out, exist_ok=True)
    logs_path = os.path.join(args.out, 'logs.txt')

    with open("log.txt", 'w') as f:
        for i in range(args.min_power, args.max_power + 1):
            num_samples = 10 ** i
            time = timeit.timeit(
                lambda: create_and_plot_data(
                    args.mean, args.std, num_samples, args.out),
                number=1)
            msg = f"[{str(round(time, 2)).rjust(5)}s] Samples={str(num_samples).ljust(12)} (mu={str(LOG_MEAN).ljust(len(str(args.mean)) + 2)}, sigma={str(LOG_STD).ljust(len(str(args.std)) + 2)})\n"
            print(msg, end='')
            f.write(msg)
