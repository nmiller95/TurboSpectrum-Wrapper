# A friendly script to generate your random input parameters file
# NJM - 29.04.2025

import random


class SpectrumGridGenerator:
    def __init__(self, limits: dict):
        """
        Generate random grid of stellar parameters compatible with Turbospectrum. Uses Magg+22 solar abundances.

        Args:
            limits(dict): Dictionary of tuples (low, high) for each parameter. Must contain (Teff, logg, Vturb, Z),
                            may also contain a number of abundances in [X/Fe]
        """
        self.limits = limits
        self.existing_params = {}

        # Limits taken from scripts in TASS
        self.similarity_tol = {
            "Teff": 5,
            "logg": 0.05,
            "Vturb": 0.01,
            "Z": 0.001,
            "abundances": 0.001
        }

        # Recommended solar abundances taken from Magg+22
        self.solar_abund = {
            'C': 8.56,
            'N': 7.98,
            'O': 8.77,
            'F': 4.40,
            'Ne': 8.15,
            'Na': 6.29,
            'Mg': 7.55,
            'Al': 6.43,
            'Si': 7.59,
            'P': 5.41,
            'S': 7.16,
            'K': 5.14,
            'Ca': 6.37,
            'Sc': 3.07,
            'Ti': 4.94,
            'V': 3.95,
            'Cr': 5.74,
            'Mn': 5.52,
            'Fe': 7.50,
            'Co': 4.95,
            'Ni': 6.24,
        }

    def _is_duplicate(self, candidate: dict):
        """
        Check if the candidate spectrum is identical to an existing spectrum within tolerance limits

        Args:
            candidate (dict): Dictionary containing parameters of candidate spectrum
        """
        n = len(next(iter(self.existing_params.values()), []))
        if n == 0:
            return False
        varying_params = {k for k, (lo, hi) in self.limits.items() if lo != hi}

        for i in range(n):
            match = True
            for param in varying_params:
                val1 = candidate.get(param)
                val2 = self.existing_params.get(param, [None]*n)[i]

                if val1 is None or val2 is None:
                    match = False
                    break

                tol = self.similarity_tol.get(param, self.similarity_tol.get("abundances", 1e-6))
                if abs(val1 - val2) > tol:
                    match = False
                    break

            if match:
                return True
        return False

    def _append_candidate(self, candidate: dict):
        """
        Append candidate set of spectral parameters to the existing set.

        Args:
            candidate (dict): Dictionary containing parameters of candidate spectrum
        """
        for key, value in candidate.items():
            if key not in self.existing_params:
                self.existing_params[key] = []
            self.existing_params[key].append(value)

    def _convert_abundances(self, candidate: dict):
        """
        Convert [X/Fe] to log ε(X) using [Fe/H] and solar ε(X)

        Args:
            candidate (dict): Dictionary containing parameters of candidate spectrum
        """
        feh = candidate["FeH"]
        log_eps = {}
        for elem, xfe in candidate.items():
            if elem in self.solar_abund and elem != "Fe":
                solar_val = self.solar_abund[elem]
                log_eps[elem] = solar_val + feh + xfe
        log_eps["Fe"] = self.solar_abund["Fe"] + feh
        return log_eps

    def sample_one(self, max_attempts=1000):
        """
        Generate one random set of spectral parameters

        Args:
            max_attempts (int): Maximum number of attempts to generate
        """
        for _ in range(max_attempts):
            candidate = {}
            for param, (lo, hi) in self.limits.items():
                if hi == lo:
                    candidate[param] = lo
                else:
                    candidate[param] = random.uniform(lo, hi)
            if not self._is_duplicate(candidate):
                self._append_candidate(candidate)
                return candidate
        return None

    def generate_samples(self, n: int):
        """
        Generate n random sets of spectral parameters, checking for and removing duplicates.

        Args:
            n (int): Number of spectra to generate
        """
        samples = []
        for _ in range(n):
            raw = self.sample_one()
            if raw:
                logeps = self._convert_abundances(raw)
                merged = {**raw, **logeps}
                samples.append(merged)
        return samples

    def export_txt(self, samples: list, filename: str):
        """
        Write space-separated .txt file with headers and custom formatting. Last column is fixed to H = 12.0.

        Args:
            samples (list): List containing parameters of the grid of spectral parameters
            filename (str): Filename of output file
        """
        order = ("Teff", "logg", "Vturb", "FeH")  # Mandatory parameters, will always be given
        excluded = set(order)
        abundances_order = [key for key in self.limits if key not in excluded]

        with open(filename, "w") as f:
            # Write header
            header = list(order) + abundances_order + ["H"]
            f.write(" ".join(header) + "\n")

            for sample in samples:
                row = []

                # Stellar parameters - rounding where appropriate
                for param in order:
                    val = sample.get(param, 0)
                    if param == "Teff":
                        row.append(f"{val:.0f}")
                    elif param == "logg" or param == "Vturb":
                        row.append(f"{val:.2f}")
                    elif param == "FeH":
                        row.append(f"{val:.3f}")
                    else:
                        row.append(f"{val:.6f}")

                # Abundances - rounding where appropriate
                for elem in abundances_order:
                    row.append(f"{sample.get(elem, 0):.4f}")

                # Add hydrogen last
                row.append("12.0")

                f.write(" ".join(row) + "\n")


if __name__ == "__main__":

    # Number of spectra to generate
    n_spectra = 50

    # Lower, upper limits to place on each parameter.

    # Teff, logg, vturb and [Fe/H] MUST be specified. The other parameters may be added/removed.
    # To fix at a single value, let lower=upper.
    # Give [X/Fe] relative abundances - they will be converted to absolute values using Magg+22 solar abundances
    grid_limits = {
        'Teff': (2800, 4500), # Split into two chunks: 2800-3900K for logg 4.5-5.5, and 3901-4500K for logg 4.5-5.0
        'logg': (4.5, 5.5),
        'Vturb': (0.01, 2.0),
        'FeH': (-2.5, 0.6),  # Lower, so as to account for low-metallicity edge effects (see FGK pipeline)
        # 14 elements as per Souto+22
        # Inspected Terese's published spectra with the current line mask:
        'C': (-0.5, 0.5),  # A handful (5-10) of molecular lines, strongest ones being CN
        'O': (-0.5, 0.5),  # Lots of molecules, especially OH
        # 'Na': (-0.5, 0.5),  # None
        'Mg': (-0.5, 0.5),  # Around 4 lines but all are very weak
        'Al': (-0.5, 0.5),  # 2 strong lines
        'Si': (-0.5, 0.5),  # A handful of Si and SiH lines, most quite weak but 2 relatively strong
        'K': (-0.5, 0.5),  # 2 strong lines
        'Ca': (-0.5, 0.5),  # Around 6 lines, 2-3 are strong
        'Ti': (-0.5, 0.5),  # Around 4 lines, 2-3 relatively strong
        # 'V': (-0.5, 0.5),  # Only 1 line, not super strong
        # 'Cr': (-0.5, 0.5),  # Only 1 line, quite weak
        # 'Mn': (-0.5, 0.5),  # Around 8 lines, all but one are very weak
        'Fe': (-0.5, 0.5),  # Quite a few instances, but usually very weak lines. 2-3 Fe lines and 2-3 FeH are good
        # 'Ni': (-0.5, 0.5)  # 3 lines, only one is relatively strong
        # My recommendation: C, O, Mg, Al, Si, K, Ca, Ti, Fe
    }

    gen = SpectrumGridGenerator(grid_limits)
    grid = gen.generate_samples(n_spectra)
    gen.export_txt(grid, "../input/random_grid.txt")
