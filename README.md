Scripts for sampling GEOS model run collections at locations specified by CSU-type scene files.

## Installation directions

The following steps require a working internet connection. First, install UV via
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Optional - set the cache directory to make sure `uv` downloads packages to a place where you have enough space. Otherwise it will likely go into your home directory in `~/.cache`:
```
# Change this!
export UV_CACHE_DIR=/discover/nobackup/psomkuti/.cache
```
Clone the simulator code

```
git clone https://github.com/PeterSomkuti/CSU-GEOS-Sampler.git
```

Inside the cloned directory, instantiate the environment

```
uv sync
```

From here onwards, no internet connection is needed.

## Usage

To run the sampler, at least 3 arguments to the program are needed:

* `--scene`, the CSU scene file to be re-written
* `--output`, the path to the new output file

Either of the following two must be supplied, but not both and not neither

* `--DYAMONDroot`, the top-level path to the DYAMOND model run
* `--GEOSroot`, the top-level path to a "standard" GEOS run, e.g. IT

And these are the optional flags, which have default values set:

* `--NN`, number of nearest neighbors used for spatial interpolation of scene locations (default: `1`)
* `--overwrite`, whether to overwrite existing output file (default: `False`)
* `--use_cf`, whether to use cloud fraction data to produce weighted cloud wather paths for use in 2-column RT simulations

Run the program via `uv`:

```
uv run Sample_GEOS.py \
    --scene ${CSU_SCENE_FILE} \
    --output ${NEW_SCENE_FILE} \
    --DYAMONDroot /css/g5nr/DYAMONDv2/03KM/DYAMONDv2_c2880_L181 \
    --NN 4 \
    --overwrite \
    --use_cf
```
