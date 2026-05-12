#!/usr/bin/env python3
import argparse
import logging


from dask.distributed import Client
import h5py
import glob
import numpy as np
import os
import pandas as pd
import pickle

# this is a slimmer and faster library, however cannot be pickled! - shame!
# from pykdtree.kdtree import KDTree
from scipy.spatial import cKDTree
import sys
from tqdm import tqdm
import xarray as xr
import warnings


# Useful constants:
Mwv = 0.0180153  # [kg/mol]
Mdry = 0.0289644  # [kg/mol]
eps = Mwv / Mdry


def setup_argparse():
    """
    Sets up the arguments for the program

    Returns:
        args (argparse.Namespace): parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Sample GEOS for CSU scene files", epilog="Author: Peter Somkuti"
    )

    # Required
    parser.add_argument("--scene", type=str, required=True, help="Scene file location")
    parser.add_argument(
        "--output", type=str, required=True, help="Output file location."
    )

    # Optional
    parser.add_argument(
        "--DYAMONDroot",
        type=str,
        required=False,
        help="Path to DYAMOND run root directory.",
    )

    parser.add_argument(
        "--NN",
        type=int,
        required=False,
        default=1,
        help="Number of nearest neighbors for cube-sphere sampling.",
    )

    # Boolean flags
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output file.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use distributed computing?",
    )
    parser.add_argument(
        "--use_cf",
        action="store_true",
        default=False,
        help="Use cloud fraction to compute LWP/IWP?",
    )

    return parser.parse_args()


def setup_logging():
    """
    Sets up logging module, setting log format and level
    """
    log_level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] (%(funcName)s:%(lineno)d) %(message)s",
        level=log_level,
    )


def round_to_multiple(x, multiple):
    return np.round(x / multiple) * multiple


def find_indices(input_list, search_string):
    return [xc for xc, x in enumerate(input_list) if search_string in str(x)]


def find_first_nonempty(row):

    if type(row[0]) is np.bytes_:
        for icol, col in enumerate(row):
            if b"none" in col:
                return icol

    elif type(row[0]) is str:
        for icol, col in enumerate(row):
            if "none" in col:
                return icol
    else:
        logging.warning("This function was designed only for bytes or str types!")
        return -1

    return -1


def hdf5_to_dict(hdf5_object):
    """
    Recursively reads an HDF5 file or group into a nested Python dictionary.

    Parameters
    ----------
    hdf5_object : h5py.File | h5py.Group | str
        An open h5py File/Group object, or a file path string.

    Returns
    -------
    dict
        A nested dictionary mirroring the HDF5 group/dataset structure.
    """
    # If a file path string is passed, open the file and recurse
    if isinstance(hdf5_object, str):
        with h5py.File(hdf5_object, "r") as f:
            return hdf5_to_dict(f)

    result = {}

    for key, item in hdf5_object.items():
        if isinstance(item, h5py.Group):
            # Recursively build a sub-dictionary for this group
            result[key] = hdf5_to_dict(item)

        elif isinstance(item, h5py.Dataset):
            # Read the dataset contents into the dictionary
            data = item[()]

            # Decode byte strings to regular Python strings if needed
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            elif isinstance(data, np.ndarray) and data.dtype.kind in ("S", "O"):
                data = data.astype(str)

            result[key] = data

    return result


def dict_to_hdf5(data_dict, hdf5_object, compression="gzip", compression_opts=4):
    """
    Recursively writes a nested Python dictionary into an HDF5 file or group,
    with optional compression on all datasets.

    Parameters
    ----------
    data_dict : dict
        A nested dictionary to write. Sub-dictionaries become HDF5 Groups,
        all other values become HDF5 Datasets.
    hdf5_object : h5py.File | h5py.Group | str
        An open h5py File/Group object to write into, or a file path string.
        If a file path string is given, a new file is created (or overwritten).
    compression : str or None
        Compression filter to use. Common options:
            "gzip"  — best compatibility, moderate speed (default)
            "lzf"   — faster, slightly less compression, h5py built-in
            "szip"  — fast, requires HDF5 szip library
            None    — no compression
    compression_opts : int or None
        Compression level. For gzip: 0 (fastest) to 9 (smallest).
        Defaults to 4 (balanced). Ignored if compression is None or "lzf".

    Returns
    -------
    None
    """
    # If a file path string is passed, open/create the file and recurse
    if isinstance(hdf5_object, str):
        with h5py.File(hdf5_object, "w") as f:
            dict_to_hdf5(
                data_dict, f, compression=compression, compression_opts=compression_opts
            )
        return

    for key, value in data_dict.items():
        if isinstance(value, dict):
            # Create a new group and recurse into it
            group = hdf5_object.require_group(key)
            dict_to_hdf5(
                value, group, compression=compression, compression_opts=compression_opts
            )

        else:
            # --- Sanitize the value before writing ---

            # Convert plain Python lists to NumPy arrays
            if isinstance(value, list):
                value = np.array(value)

            # Encode regular Python strings to bytes for HDF5 compatibility
            elif isinstance(value, str):
                value = np.bytes_(value)

            # Encode NumPy string arrays to bytes
            elif isinstance(value, np.ndarray) and value.dtype.kind == "U":
                value = value.astype("S")

            # Write the dataset, overwriting if it already exists
            if key in hdf5_object:
                del hdf5_object[key]

            # Scalar values cannot be compressed — skip compression for those
            is_compressible = isinstance(value, np.ndarray) and value.shape != ()

            hdf5_object.create_dataset(
                key,
                data=value,
                compression=compression if is_compressible else None,
                compression_opts=compression_opts if is_compressible else None,
            )


def sample_with_cubesphere(dataset, locations_df, varlist_3d, varlist_2d, NN=4):

    # Create the per-face KDTrees for coordinate lookup
    # (this takes a few seconds sadly..)
    if not os.path.exists("face_trees.pickle"):
        face_trees = {}

        for face in range(6):
            logging.info(f"Building tree for face {face}")
            tree_coords = np.column_stack([
                dataset.lons[face, :, :].values.flatten(),
                dataset.lats[face, :, :].values.flatten(),
            ])

            face_trees[face] = cKDTree(tree_coords)
            # face_trees[face] = BallTree(data, metric='haversine') # this is too slow!!

        logging.info("Saving face trees ..")
        with open("face_trees.pickle", "wb") as file:
            pickle.dump(face_trees, file)
        logging.info("Done creating face trees.")

    else:
        # Load pickled tree
        logging.info("Loading face trees ..")
        with open("face_trees.pickle", "rb") as file:
            face_trees = pickle.load(file)
        logging.info("Done loading face trees.")

    # Now we find the right face and indices where we can look up
    # the various GEOS variables

    _coords = np.column_stack([locations_df.lon.values, locations_df.lat.values])

    dist_all = np.zeros((6, len(locations_df), NN), dtype=float)
    idx_all = np.zeros((6, len(locations_df), NN), dtype=int)

    for face in range(6):
        _res = face_trees[face].query(_coords, k=NN)

        if NN == 1:
            # .query doesn't return a 2-d array if k = 1
            dist_all[face, :, 0], idx_all[face, :, 0] = _res
        else:
            dist_all[face, :, :], idx_all[face, :, :] = _res

    # Allocate and set to default values
    locations_df.loc[:, "face_idx"] = -1
    for i in range(NN):
        locations_df.loc[:, f"x_idx_{i}"] = -1
        locations_df.loc[:, f"y_idx_{i}"] = -1
        locations_df.loc[:, f"weight_{i}"] = 0.0

    # For every scene, we pick the face that shows the smallest
    # distance to a model grid point.
    face_idx_all = np.argmin(dist_all[:, :, 0], axis=0)
    locations_df.loc[:, "face_idx"] = face_idx_all

    # Calculate x/y indices from flat indices, taken from the
    # face we believe is the one to use.
    xy_idx_sel = np.unravel_index(
        idx_all[face_idx_all, np.arange(len(locations_df))], dataset.lons.shape[1:]
    )

    for i in range(NN):
        # Pay attention to the indices: 0 -> y, 1 -> x
        locations_df.loc[:, f"x_idx_{i}"] = xy_idx_sel[1][:, i]
        locations_df.loc[:, f"y_idx_{i}"] = xy_idx_sel[0][:, i]

    # Calculate interpolation weights take from the face be believe is the one to use.
    # (we use 1 / distance^2 weighting)
    weight_sel = 1 / (dist_all[face_idx_all, np.arange(len(locations_df))]) ** 2
    # Normalize
    # weight_sel = (weight_sel.T / weight_sel.sum(axis=1)).T
    weight_sel /= weight_sel.sum(axis=1)[:, np.newaxis]

    for i in range(NN):
        locations_df.loc[:, f"weight_{i}"] = weight_sel[:, i]

    # Now we can sample each scene at each of the NN locations, but we must do it for each
    # level separately. These arrays have dimensions (scene, level, neighbor)
    sample_dict = {}
    for var3d in varlist_3d:
        sample_dict[var3d] = np.zeros((len(locations_df), len(dataset.lev), NN))

    for var2d in varlist_2d:
        sample_dict[var2d] = np.zeros((len(locations_df), NN))

    print("Performing sampling..")
    for var3d in varlist_3d:
        print(f"Sampling 3D var: {var3d}")

        for i in tqdm(range(NN)):
            _grab = dataset[var3d].isel(
                time=0,
                nf=xr.DataArray(locations_df.face_idx, dims="points"),
                Ydim=xr.DataArray(locations_df[f"y_idx_{i}"], dims="points"),
                Xdim=xr.DataArray(locations_df[f"x_idx_{i}"], dims="points"),
                lev=range(len(dataset.lev)),
            )

            _grab.load()
            sample_dict[var3d][:, :, i] = _grab.values.T

    for var2d in varlist_2d:
        print(f"Sampling 2D var: {var2d}")

        for i in tqdm(range(NN)):
            _grab = dataset[var2d].isel(
                time=0,
                nf=xr.DataArray(locations_df.face_idx, dims="points"),
                Ydim=xr.DataArray(locations_df[f"y_idx_{i}"], dims="points"),
                Xdim=xr.DataArray(locations_df[f"x_idx_{i}"], dims="points"),
            )

            _grab.load()
            sample_dict[var2d][:, i] = _grab.values.T

    # Check if there are any zero-only arrays, i.e.
    # the sampling didn't yield anything useful.
    for var in sample_dict.keys():
        if np.all(sample_dict[var] == 0):
            logging.warning(
                f"Variable {var} sampled to all zeroes! (maybe something went wrong)"
            )

    # Apply the distance-based weighting for all variables
    sample_weighted = {}

    for var3d in varlist_3d:
        print(f"Calculating weighted {var3d}")
        tmp = np.zeros((len(locations_df), len(dataset.lev)))

        for i in range(NN):
            tmp[:, :] += (
                sample_dict[var3d][:, :, i]
                * locations_df[f"weight_{i}"].values[:, np.newaxis]
            )

        sample_weighted[var3d] = tmp

    for var2d in varlist_2d:
        print(f"Calculating weighted {var2d}")
        tmp = np.zeros(len(locations_df))

        for i in range(NN):
            tmp[:] += sample_dict[var2d][:, i] * locations_df[f"weight_{i}"].values[:]

        sample_weighted[var2d] = tmp

    return sample_weighted


def find_closest_indices(X, Y):

    # For each element in Y, find the index in X with minimum absolute difference
    srt = np.array([np.argmin(np.abs(X - y)) for y in Y])
    return srt


def g_from_latitude(lat):

    latrad = np.deg2rad(lat)
    return 9.780327 * (
        1 + 0.0053024 * np.sin(latrad) ** 2 - 0.0000058 * np.sin(2 * latrad) ** 2
    )


def g_from_alt_and_lat(alt, lat):

    # Alt must be in m for this..
    Re = 6_371_000.0
    g0 = g_from_latitude(lat)
    return g0 * (Re / (Re + alt)) ** 2


def main():

    # Get rid of this xarray warning for duplicate dimensions. We cannot
    # really fix this due to the way how the cube-sphere files are structured..
    warnings.filterwarnings(
        "ignore", message=".*We do not yet support duplicate dimension names.*"
    )

    args = setup_argparse()
    setup_logging()

    logging.info(args.overwrite)
    # Check here already if the file exists
    if os.path.exists(args.output) and (not args.overwrite):
        logging.warning(
            f"Output file location: {args.output} exists! Use `--overwrite "
            "if you want to overwrite existing files!"
        )
        return

    if os.path.exists(args.output) and args.overwrite:
        logging.warning(
            f"Output file location: {args.output} exists! You chose to overwrite!"
        )

    if args.parallel:
        logging.info("Using parallel computing:")
        client = Client()
        logging.info(client)
    else:
        client = None
        logging.info("Only serial computations..")

    # Flags that tell our code whether we are working with
    #   a) DYAMOND model run files
    #   b) GEOS CARB ana files

    mode_DYAMOND = False
    mode_GEOSCARB = False

    if args.DYAMONDroot is not None:
        mode_DYAMOND = True
        root = args.DYAMONDroot
        # Should be something like "/css/g5nr/DYAMONDv2/03KM/DYAMONDv2_c2880_L181"
        logging.info(f"Root directory for DYAMOND supplied: {root}")

    if (not mode_DYAMOND) and (not mode_GEOSCARB):
        logging.error("Need to set at least `DYAMONDroot` or `GEOSCARBroot`!")
        sys.exit(1)

    h5_scene = h5py.File(args.scene, "r")

    scene_lons = h5_scene["Simulation/Geometry/longitude"][:]
    scene_lats = h5_scene["Simulation/Geometry/latitude"][:]
    scene_epoch = h5_scene["Simulation/Time/epoch"][:]

    Nframe, Nband, Nfp = scene_lons.shape
    N_scene, N_scene_levels = h5_scene["Simulation/Thermodynamic/pressure_level"].shape

    # Fudge scene epoch so we can use the DYAMOND runs
    scene_epoch[:, :, :, 0] = 2020
    scene_epoch[:, :, :, 1] = 2
    scene_epoch[:, :, :, 2] = 1

    # Create datetime objects
    scene_times = np.zeros((Nframe, Nband, Nfp), dtype="datetime64[ms]")
    for fr in range(Nframe):
        for band in range(Nband):
            for fp in range(Nfp):
                scene_times[fr, band, fp] = (
                    pd
                    .Timestamp(*(scene_epoch[fr, band, fp]))
                    .to_numpy()
                    .astype("datetime64[ms]")
                )

    # Flatten things out into a data array to be sampled
    # (note that here we will ignore the band dimension, assume that all bands
    #  will sample the same point on Earth at the same time)

    locations_df = pd.DataFrame(index=range(Nframe))

    locations_df.loc[:, "lon"] = scene_lons[:, 0, :].flatten()
    locations_df.loc[:, "lat"] = scene_lats[:, 0, :].flatten()
    locations_df.loc[:, "time"] = scene_times[:, 0, :].flatten()

    locations_df.loc[:, "year"] = scene_epoch[:, 0, :, 0].flatten()
    locations_df.loc[:, "month"] = scene_epoch[:, 0, :, 1].flatten()
    locations_df.loc[:, "day"] = scene_epoch[:, 0, :, 2].flatten()
    locations_df.loc[:, "hour"] = scene_epoch[:, 0, :, 3].flatten()

    locations_df.loc[:, ["idx_lon", "idx_lat"]] = -1

    def return_ymd(row):
        return f"{row.year:04d}{row.month:02d}{row.day:02d}"

    def return_ymdh(row):
        return f"{row.year:04d}{row.month:02d}{row.day:02d}_{100 * row.hour:04d}"

    locations_df["ymd"] = locations_df.apply(return_ymd, axis=1)
    locations_df["ymdh"] = locations_df.apply(return_ymdh, axis=1)

    # Figure out which YMDH we have and thus need to consider in the file list
    un_ymdh = locations_df.ymdh.unique()

    # Make a list of files that we need to *consider* reading
    flist = []

    varlist_3d = ["AIRDENS", "QL", "RL", "QI", "RI", "QV", "DELP", "P", "CO2", "H", "T"]
    if args.use_cf:
        varlist_3d.append("FCLD")
    # add aerosols
    # (note that SU aerosols are named `SO4` in the DYAMOND collection)
    aerlist = ["BC", "DU", "OC", "SS", "SO4"]
    varlist_3d += aerlist

    varlist_2d = ["PS", "PHIS"]

    for ymdh in un_ymdh:
        ym = ymdh[:6]
        ymd = ymdh[:8]

        for var3d in varlist_3d:
            if mode_DYAMOND:
                fname = glob.glob(
                    f"{root}/inst_01hr_3d_{var3d}_Mv/{ym}/DYAMONDv2_c2880_L181.inst_01hr_3d_{var3d}_Mv.{ymdh}z.nc4"
                )[0]

            flist.append(fname)

        # Add the 2D 15min data (but we can sample at the hour)
        if mode_DYAMOND:
            flist.append(
                f"{root}/inst_15mn_2d_asm_Mx/{ym}/DYAMONDv2_c2880_L181.inst_15mn_2d_asm_Mx.{ymdh}z.nc4"
            )

    dataset = xr.open_mfdataset(
        flist,
        drop_variables=[
            "anchor",
            "contacts",
            "ncontact",
            "corner_lons",
            "corner_lats",
            "orientation",
        ],
        compat="override",
        coords="all",
        chunks={"Nf": "auto"},
        parallel=(client is not None),
    )

    # We need to load the const data seperately because they have different time stamps
    # Add the 2D const data (only 0000z)

    if mode_DYAMOND:
        dataset_const = xr.open_dataset(
            f"{root}/const_2d_asm_Mx/{ym}/DYAMONDv2_c2880_L181.const_2d_asm_Mx.{ymd}_0000z.nc4",
            drop_variables=[
                "anchor",
                "contacts",
                "ncontact",
                "corner_lons",
                "corner_lats",
                "orientation",
            ],
            chunks={"Nf": "auto"},
        )

    dataset_const["time"] = dataset["time"]
    # push into main dataset ..
    dataset = xr.merge([dataset, dataset_const], compat="override")

    logging.info("Loading lon/lat into memory..")
    # Load into memory the lon/lat grid
    dataset.lons.load()
    dataset.lats.load()

    # Switch here whether we do cubesphere or regular lon/lat grid
    # TODO!
    NN = max(1, args.NN)
    logging.info(f"Using NN={NN} nearest neighbours for sampling.")
    logging.info(f"(user-supplied value: {args.NN})")

    sampled_data = sample_with_cubesphere(
        dataset, locations_df, varlist_3d, varlist_2d, NN=NN
    )

    # Calculate surface-altitude from PHIS
    sampled_data["Z0"] = sampled_data["PHIS"] / 9.80665

    N_GEOS_lay = sampled_data["P"].shape[1]
    # Produce pressure edges for sampled data
    sampled_data["Plevs"] = np.zeros((N_scene, N_GEOS_lay + 1))

    # Insert surface pressure
    sampled_data["Plevs"][:, N_GEOS_lay] = sampled_data["PS"]

    for lev in range(N_GEOS_lay - 1, -1, -1):  # count from surface to top
        sampled_data["Plevs"][:, lev] = (
            sampled_data["Plevs"][:, lev + 1] - sampled_data["DELP"][:, lev]
        )

    # Occasionally, the top pressure level will be < 0 (for some reason), so we replace those
    # by some predetermined nudge value and make sure it's not larger than the level below.
    neg_P = np.where(sampled_data["Plevs"][:, 0] < 0)[0]
    sampled_data["Plevs"][neg_P, 0] = np.maximum(
        0.1, sampled_data["Plevs"][neg_P, 1] - 0.1
    )

    if np.any(sampled_data["Plevs"][neg_P, 0] > sampled_data["Plevs"][neg_P, 1]):
        logging.error("Problem with fixing top-level pressure!")
        logging.error(
            np.where(sampled_data["Plevs"][neg_P, 0] > sampled_data["Plevs"][neg_P, 1])
        )
        sys.exit(1)

    # This is obviously an approximation. We do not have quick access to
    # the layer thickness in [m], so instead of calculating QI * \Delta z,
    # we use QI * \Delta p / g. Since we also do not have access to per-layer g,
    # we choose some value and live with the inaccuracies..
    LWP = sampled_data["QL"] * sampled_data["DELP"] / 9.80665
    IWP = sampled_data["QI"] * sampled_data["DELP"] / 9.80665

    WP = IWP + LWP

    # If the user wants to use cloud fractions, we will produce weighted water paths:
    if args.use_cf:
        logging.info("Using cloud-fractions!")

        # Average cloud fraction for the total column, weighted by the total cloud water content
        # See 10.1175/2009JAMC2170.1, scheme 2O
        Cav = np.zeros(len(locations_df))
        idx_good = WP.sum(axis=1) > 0
        Cav[idx_good] = ((WP) * sampled_data["FCLD"])[idx_good, :].sum(axis=1) / (
            WP[idx_good, :]
        ).sum(axis=1)

        # LWP_w = LWP / Cav[:, np.newaxis]
        # LWP_w[np.isnan(LWP_w)] = 0
        # IWP_w = IWP / Cav[:, np.newaxis]
        # IWP_w[np.isnan(IWP_w)] = 0

        LWP_w = np.where(Cav[:, np.newaxis] != 0, LWP / Cav[:, np.newaxis], 0.0)
        IWP_w = np.where(Cav[:, np.newaxis] != 0, IWP / Cav[:, np.newaxis], 0.0)

    else:
        # If user does NOT want cloud fractions, we just use the
        # non-weighted LWP/IWPs
        LWP_w = LWP
        IWP_w = IWP

    # Optical depth approximation:
    # ----------------------------
    # 3 / 2 * LWP / (RL * \rho_water) = 1.5 / \rho_water * LWP / RL
    # = 0.0015 * LWP / RL (since \rho_water is ~1000 kg / m+3)
    TAU_CL = (0.0015 * LWP / sampled_data["RL"]).sum(axis=1)
    TAU_CL_w = (0.0015 * LWP_w / sampled_data["RL"]).sum(axis=1)
    TAU_CL_w[np.isnan(TAU_CL_w)] = 0

    # 3 / 2 * IWP / (RL * \rho_ice) = 1.5 / \rho_ice * IWP / RI
    # = ~0.00164 * IWP / RL (since \rho_ice is ~917 kg / m+3)
    TAU_CI = (0.00164 * IWP / sampled_data["RI"]).sum(axis=1)
    TAU_CI_w = (0.00164 * IWP_w / sampled_data["RI"]).sum(axis=1)
    TAU_CI_w[np.isnan(TAU_CI_w)] = 0

    # Prepare data to be written out:
    out = dict()

    # Following contents of the original scene file can be copied 1:1,
    # without the need for modification:
    for d in [
        "Footprint",
        "Geometry",
        "Metadata",
        "Orbit",
        "Sounding",
        "Surface",
        "Time",
    ]:
        out[d] = hdf5_to_dict(h5_scene["Simulation"][d])

    out["Thermodynamic"] = dict()
    out["Gas"] = dict()
    out["Aerosol"] = dict()

    # Set up
    out["Thermodynamic"]["altitude_level"] = np.zeros((N_scene, N_GEOS_lay + 1))
    out["Thermodynamic"]["gravity_layer"] = np.zeros((N_scene, N_GEOS_lay))
    out["Thermodynamic"]["num_layers"] = np.repeat(N_GEOS_lay, N_scene)
    out["Thermodynamic"]["pressure_level"] = sampled_data["Plevs"]
    out["Thermodynamic"]["temperature_level"] = np.zeros((N_scene, N_GEOS_lay + 1))

    out["Gas"]["num_species"] = np.zeros(N_scene, dtype="int")
    out["Gas"]["species_id"] = np.zeros((N_scene, 10), dtype="|S16")

    # Note that these seem to be hard-coded in the CSU simulator
    out["Gas"]["species_id"][:, 0] = "AIR_moist".ljust(16, " ")
    out["Gas"]["species_id"][:, 1] = "AIR_dry".ljust(16, " ")
    out["Gas"]["species_id"][:, 2] = "H2O".ljust(16, " ")
    out["Gas"]["species_id"][:, 3] = "CO2".ljust(16, " ")
    out["Gas"]["species_id"][:, 4] = "O2".ljust(16, " ")
    # out["Gas"]["species_id"][:,5] = "O3".ljust(16, " ")
    # out["Gas"]["species_id"][:,6] = "CH4".ljust(16, " ")
    # out["Gas"]["species_id"][:,7] = "CO".ljust(16, " ")
    # out["Gas"]["species_id"][:,8] = "HDO".ljust(16, " ")

    out["Gas"]["species_density"] = np.zeros((N_scene, 10, N_GEOS_lay))

    out["Aerosol"]["num_species"] = np.zeros(N_scene, dtype="int")
    out["Aerosol"]["species_id"] = np.zeros((N_scene, 100), dtype="|S16")
    out["Aerosol"]["species_id"][:, :] = "none".ljust(16, " ")
    out["Aerosol"]["species_density"] = np.zeros(
        (N_scene, 100, N_GEOS_lay), dtype=np.float32
    )

    logging.info("Calculating meteorology ...")

    # Copy all surface elevations, these are the lowest level of the new altitude
    # level grid.
    out["Thermodynamic"]["altitude_level"][:, -1] = sampled_data["Z0"]
    # Altitude at level `lay`, say it's roughly between the mid-layer for layer `lay`
    # and the one below. The lowest level is always the surface, and the topmost level
    # is just approximated (per-scene, see below..)
    out["Thermodynamic"]["altitude_level"][:, 1:-1] = np.stack([
        sampled_data["H"][:, :-1],
        sampled_data["H"][:, 1:],
    ]).mean(0)

    # =======================================
    # Temperature GEOS layers -> scene levels
    # This is a lazy conversion that does not obey the laws of thermodynamics,
    # but may be decent enough if enough layers/levels are given such that
    # the converted profile will be similarly "smooth".

    # Set the top T level to be the top model layer
    out["Thermodynamic"]["temperature_level"][:, 0] = sampled_data["T"][:, 0]
    # Set the bottom T level to be the bottom model layer
    out["Thermodynamic"]["temperature_level"][:, -1] = sampled_data["T"][:, -1]
    # Set the other levels to be mid-point of the adjacent layers
    out["Thermodynamic"]["temperature_level"][:, 1:-1] = 0.5 * (
        sampled_data["T"][:, 0:-1] + sampled_data["T"][:, 1:]
    )

    # =======================================

    for i_scene in tqdm(range(N_scene)):
        # Mid-layer gravity (approx.)
        out["Thermodynamic"]["gravity_layer"][i_scene, :] = g_from_alt_and_lat(
            sampled_data["H"][i_scene, :], locations_df.lat[i_scene]
        )

        # Quick hack: fit a line through log10(p) vs. altitude for the profile, and
        # then evaluate for the missing point at the top of the atmosphere..
        _p = np.poly1d(
            np.polyfit(
                np.log10(out["Thermodynamic"]["pressure_level"][i_scene, 1:]),
                out["Thermodynamic"]["altitude_level"][i_scene, 1:],
                1,
            )
        )

        out["Thermodynamic"]["altitude_level"][i_scene, 0] = _p(
            np.log10(out["Thermodynamic"]["pressure_level"][i_scene, 0])
        )

    logging.info("Calculating gases ...")
    for i_scene in tqdm(range(N_scene)):
        # =========
        # Set gases
        # =========
        q = sampled_data["QV"][i_scene]  # shortcut

        M_moist = Mdry * (1 - q) + Mwv * q

        # Moles of moist air per layer
        out["Gas"]["species_density"][i_scene, 0, :] = sampled_data["DELP"][i_scene] / (
            out["Thermodynamic"]["gravity_layer"][i_scene] * M_moist
        )

        # Moles of dry air per layer
        out["Gas"]["species_density"][i_scene, 1, :] = sampled_data["DELP"][i_scene] / (
            out["Thermodynamic"]["gravity_layer"][i_scene] * Mdry
        )
        out["Gas"]["species_density"][i_scene, 1, :] *= 1 - sampled_data["QV"][i_scene]

        # Produce H2O VMR from humidity
        # (h2o  = q / ((1 - q) * MM_H2O_TO_AIR + q))

        out["Gas"]["species_density"][i_scene, 2, :] = q / ((1 - q) * eps + q)

        # Set O2 always as 0.20945 parts of dry air
        out["Gas"]["species_density"][i_scene, 4, :] = (
            0.20945 * out["Gas"]["species_density"][i_scene, 1, :]
        )

        # Set CO2 to be whatever the GEOS output tells us
        out["Gas"]["species_density"][i_scene, 3, :] = (
            sampled_data["CO2"][i_scene, :]
            * out["Gas"]["species_density"][i_scene, 1, :]
        )

        out["Gas"]["num_species"][i_scene] = 5  # (moist air, dry air, H2O, CO2, O2)

    logging.info("Inserting water clouds ...")
    LWP_thres = 1e-20
    for i_scene in tqdm(range(N_scene)):
        # -----------
        # WATER CLOUD
        # -----------

        if LWP_w[i_scene].sum() == 0:
            continue

        this_RL_profile = round_to_multiple(sampled_data["RL"][i_scene] * 1e6, 1)
        # Cap water clouds between 001 and 060
        this_RL_profile = np.maximum(np.minimum(this_RL_profile, 60), 1)
        # We only look for Reff values which have corresponding LWP > some value
        # (GEOS produces Reff even if there is no cloud)
        unique_RL = np.unique(this_RL_profile[LWP_w[i_scene] > LWP_thres])
        RL_labels = [
            "water_cloud_{:03d}".format(int(x)).ljust(15, " ") for x in unique_RL
        ]

        start_insert = find_first_nonempty(out["Aerosol"]["species_id"][i_scene])

        for k, RL in enumerate(unique_RL):
            _idx = np.where((this_RL_profile == RL) & (LWP_w[i_scene] > LWP_thres))[0]

            if len(_idx) == 0:
                continue

            if start_insert + k < out["Aerosol"]["species_id"].shape[1]:
                out["Aerosol"]["species_id"][i_scene, start_insert + k] = RL_labels[k]
                out["Aerosol"]["species_density"][i_scene, start_insert + k, _idx] = (
                    LWP_w[i_scene, _idx]
                )
            else:
                logging.info(f"Can't insert water cloud profile: {i_scene}")

    logging.info("Inserting ice clouds ...")
    IWP_thres = 1e-20
    for i_scene in tqdm(range(N_scene)):
        if IWP_w[i_scene].sum() == 0:
            continue

        # -----------
        # ICE CLOUD
        # -----------

        this_RI_profile = round_to_multiple(sampled_data["RI"][i_scene] * 1e6, 5)
        # Cap ice clouds between 010 and 090
        this_RI_profile = np.maximum(np.minimum(this_RI_profile, 90), 10)
        # We only look for Reff values which have corresponding LWP > some value
        # (GEOS produces Reff even if there is no cloud)
        unique_RI = np.unique(this_RI_profile[IWP_w[i_scene] > IWP_thres])
        RI_labels = [
            "ice_cloud_{:03d}".format(int(x)).ljust(16, " ") for x in unique_RI
        ]

        start_insert = find_first_nonempty(out["Aerosol"]["species_id"][i_scene])

        for k, RI in enumerate(unique_RI):
            _idx = np.where((this_RI_profile == RI) & (IWP_w[i_scene] > IWP_thres))[0]

            if len(_idx) == 0:
                continue

            if start_insert + k < out["Aerosol"]["species_id"].shape[1]:
                out["Aerosol"]["species_id"][i_scene, start_insert + k] = RI_labels[k]
                out["Aerosol"]["species_density"][i_scene, start_insert + k, _idx] = (
                    IWP_w[i_scene, _idx]
                )
            else:
                logging.info(f"Can't insert ice cloud profile: {i_scene}")

    logging.info("Calculate aerosol total masses")
    # Aerosol mass in kg

    for aer in aerlist:
        logging.info(f"{aer}...")
        out["Aerosol"][f"{aer}_total_mass"] = np.zeros(N_scene)

        for i_scene in tqdm(range(N_scene)):
            out["Aerosol"][f"{aer}_total_mass"][i_scene] = (
                sampled_data[aer][i_scene, :]
                * sampled_data["DELP"][i_scene, :]
                / 9.80665
            ).sum()

    ## Computations are done!
    ## .. put together output dictionary!

    ## Add a cloud grop for cloud diagnostics

    out["Cloud"] = dict()

    # Only store Cav if user does cloud fractions
    if args.use_cf:
        out["Cloud"]["Cav"] = Cav

    out["Cloud"]["cloud_liquid_water_optical_depth"] = TAU_CL
    out["Cloud"]["cloud_ice_water_optical_depth"] = TAU_CI

    ## AFTER both clouds and aerosols are added,
    ## we count up the number of slots used. Do this
    ## the slow way since comparing against b"none    " etc.
    ## in a vectorized fashion may be prone to errors.
    logging.info("Counting up the number of aerosol species per scene ...")
    for i_scene in tqdm(range(N_scene)):
        this_count = 0
        for x in out["Aerosol"]["species_id"][i_scene]:
            if b"none" not in x:
                this_count += 1

        out["Aerosol"]["num_species"][i_scene] = this_count

    # Now write out the contents of `out` into a new file
    h5_out = h5py.File(args.output, "w")
    h5_out.create_group("Simulation")
    dict_to_hdf5(out, h5_out["Simulation"])

    # Finalize
    h5_scene.close()
    dataset.close()

    logging.info("Finish.")


if __name__ == "__main__":
    main()
