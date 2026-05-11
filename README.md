Scripts for sampling GEOS model run collections at locations specified by CSU-type scene files.

This program takes an existing CSU scene file, extracts scene locations and observation geometries, and then produces a new scene file with a new atmospheric state derived from either a GEOS DYAMOND run or some other more generic GEOS run. The new scene will have the native GEOS model run vertical resolution, and **all existing atmospheric states of the original scene are removed and replaced by those of the source GEOS run**. Water and ice clouds are either used as-is, or modified via the model cloud fractions with a two-column scheme from 10.1175/2009JAMC2170.1.

Aerosol distributions and profiles from GEOS are ingested into the new scene file as well, however it is required to use the ECMWF CAMS aerosol reanalysis to distribute the aerosol masses per species into different size bins. The final scene file will thus have the atmospheric state (pressure levels, surface pressure, gas volume mixing ratios, aerosol profiles and aerosol spatial distributions) from GEOS, but the aerosols optical properties and the splitting into various size bins is derived from CAMS.

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

## GEOS Sampler Usage

To run the GEOS sampler, at least 3 arguments to the program are needed:

* `--scene`, the CSU scene file to be re-written
* `--output`, the path to the new output file

Either of the following two must be supplied, but not both and not neither

* `--DYAMONDroot`, the top-level path to the DYAMOND model run
* `--GEOSroot`, the top-level path to a "standard" GEOS run, e.g. IT

And these are the optional flags, which have default values set:

* `--NN`, number of nearest neighbors used for spatial interpolation of scene locations (default: `1`)
* `--overwrite`, whether to overwrite existing output file (default: `False`)
* `--use_cf`, whether to use cloud fraction data to produce weighted cloud wather paths for use in 2-column RT simulations (default: `False`)

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

## CAMS Sampler Usage

In order to distribute the GEOS aerosol masses (per mixture) into size bins with different optical properties, run the CAMS sampler:

```
uv run Sample_CAMS_aerosols.py \
    --scene ${NEW_SCENE_FILE} \
    --aerdb aerdb_CAMS_oco2.h5
    --modelpath /data10/CAMS-AEROSOL
```

Where `--aerdb` must point to the correct aerosol property database file, and `--modelpath` should point to the top-level path containing the CAMS aerosol files.
