# Distribution Sampling and Plotting

> “Plot the distribution by randomly sampling 10, 100, 1,000, 10,000 (and up to 100,000,000) from a normal distribution.”

It allows sampling from a normal (Gaussian) distribution with configurable parameters, plotting histograms of the samples, and recording statistics and timing.

## Task Overview

- **Randomly sample** from a normal distribution with user-specified mean and standard deviation.
- **Plot histograms** for increasing sample sizes: $10^1, 10^2, \ldots, 10^n$.
- **Save plots** to a directory.

## How to Run

### Prerequisites

- Python 3.x
- Required libraries: `numpy`, `matplotlib`

Install dependencies if needed:
```bash
pip install numpy matplotlib
```

### Running the Script

```bash
python sample.py [opts]
```

#### Options

| Option        | Description                                       | Default |
| ------------- | ------------------------------------------------- | ------- |
| `--mean`      | Mean ($\mu$) of the normal distribution           | 0       |
| `--std`       | Standard deviation ($\sigma$) of the distribution | 1       |
| `--min-power` | Smallest power of 10 for sample count             | 1       |
| `--max-power` | Largest power of 10 for sample count              | 8       |
| `--out`       | Output directory for plots and logs               | test    |

#### Examples

Sample from $\mathcal{N}(0, 1)$, plot for 10, 100, $\ldots$, 100,000,000 samples, save in the default `test/` folder:
```bash
python sample.py
```

For a custom distribution and output directory:
```bash
python sample.py --mean 2       \
                 --std 3        \
                 --min-power 2  \
                 --max-power 6  \
                 --out results
```

## Output

- **Plots:** In output directory, one for each sample size.
- **Log File:** `logs.txt` for your records, and also prints a line per run as:
  ```
  [1.07s] Samples=1000        (mu=0.014 , sigma=1.023 )
  ```

## Notes

- I've used the binning formula as `bins=20*floor(log10(num_samples))`, i.e., number of bins/buckets based on the number of samples but it could be changed as desired.
- The log file records only the empirical sample mean and std for each run, not the theoretical input as you know it already.

# Acknowledgements

I've created this README with help of [perplexity.ai](https://www.perplexity.ai)