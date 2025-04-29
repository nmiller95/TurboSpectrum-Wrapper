# TurboSpectrum-Wrapper for M-dwarf SAPP

[Original version](https://github.com/EkaterinaSe/TurboSpectrum-Wrapper) by E. Magg, M. Bergemann, B. Plez, J. Gerber.
Before use, please refer to the 
[original documentation](https://ekaterinase.github.io/TurboSpectrum-Wrapper/build/html/index.html).

This modified version is specifically tailored for the purpose for generating a grid of synthetic spectra for training a 
neural network for the SAPP pipeline (see 
[Kovalev+19](https://ui.adsabs.harvard.edu/abs/2019A%26A...628A..54K/abstract), 
[Gent+22](https://ui.adsabs.harvard.edu/abs/2022A%26A...658A.147G/abstract), 
[Olander+25](https://ui.adsabs.harvard.edu/abs/2025A%26A...696A..62O/abstract)).

## Installation 

### This wrapper
Clone this repository:
```bash
git clone https://github.com/nmiller95/TurboSpectrum-Wrapper.git 
```
Make sure the Python dependencies are installed. This can be done manually or install with:
```bash
pip install -r requirements.txt
```

#### Configuration
You can customise several aspects of the wrapper in `/input/config.txt`, e.g. number of CPUs to use,
the wavelength range over which to generate your spectra, and file paths.

### TurboSpectrum
Make sure you have [TurboSpectrum v.20](https://github.com/bertrandplez/Turbospectrum2020)
installed. If you installing it specifically for this project, you may find it convenient to place the files in the 
`/turbospectrum` directory in this project.

Update the variable `ts_root` in `/input/config.txt` to match the *absolute* path of your TurboSpectrum installation.

#### Known issues
Since this repo is a quick modification of the `TurboSpectrum-Wrapper` code, I didn't fix all of the problems. 
If the code crashes when you run it for the first time, check that the compiler on lines 51 and 135 of `run_ts.py` 
match your fortran compiler. For my Mac with gfortran, I changed 'exec' to 'exec-gf'. 
Available compiler options are: 'exec', 'exec-gf', 'exec-ifx', 'exec-intel'.

### MARCS model atmospheres
Download MARCS models over the parameter range you are interested in and place them in `/input/marcs_models`.
These can be downloaded from the [main site](https://marcs.oreme.org/) or [Uppsala mirror](https://marcs.astro.uu.se/).
You will only need the `.mod` files.

If you prefer to place these elsewhere, you can update the variable `atmos_path` in `/input/config.txt`

### Line lists
Line lists suitable for generating M-dwarf spectra should be added to the `/input/linelists` directory. 
Place or replace line lists here and update the variable `linelist` in `/input/config.txt`.
You can download the line lists used in this version of the wrapper 
from [APOGEE](https://data.sdss.org/sas/dr17/apogee/spectro/speclib/linelists/turbospec/).

## Running the wrapper

### Random grid parameter generation

To generate a file containing a set of random parameters, navigate to the script `/source/make_random_grid_input.py`.
You'll need to go to the bottom of the script to tailor the parameters you want to use and their ranges.
Here, you will find some guidance. Once happy with the setup, navigate to the `/source` directory and run: 
```bash
python make_random_grid_input.py
```

Check that the correct file name/path is in `/input/config.txt` before proceeding to the next step.

### Running TurboSpectrum

Now you can run the whole thing. Make sure you're in `/source` directory. Basic usage:
```bash
python generate_random_grid.py
```
This assumes you're using `/input/config.txt` as the configuration file.
You can instead customise the configuration file path and job names with:
```bash
python generate_random_grid.py configFile.txt jobName
```
