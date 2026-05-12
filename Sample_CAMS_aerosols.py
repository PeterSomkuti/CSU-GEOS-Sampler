#!/usr/bin/env python3

# Note: this code was partially refactored via ChatGSFC

import argparse
import logging
import os
import sys
from datetime import timedelta

import h5py
import numpy as np
import pandas as pd
import tables as tb
from netCDF4 import Dataset, num2date, date2num
from scipy import interpolate as spi
from tqdm import tqdm

# Set up standardized logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sat_vp(t, t_in_k=True):
    """Calculates the saturation water vapor pressure in Pa. t in C or K."""
    t = np.asarray(t) - 273.15 if t_in_k else np.asarray(t)

    ps = np.where(
        t > 0.0,
        np.exp(34.494 - (4924.99 / (t + 237.1))) / np.power(t + 105, 1.57),
        np.exp(43.494 - (6545.80 / (t + 278.0))) / np.power(t + 858, 2.00),
    )
    return ps.item() if ps.ndim == 0 else ps


def rh2sh(rh, p, t, t_in_k=True):
    """Converts relative humidity to specific humidity (p in Pa)."""
    a = 0.621865
    ps = sat_vp(t, t_in_k=t_in_k)
    e = ps * rh / 100.0
    return a * e / (p - (1.0 - a) * e)


def sh2rh(sh, p, t, t_in_k=True):
    """Converts specific humidity to relative humidity (p in Pa)."""
    a = 0.621865
    ps = sat_vp(t, t_in_k=t_in_k)
    return (p / (ps * (1.0 - a) + a * ps / sh)) * 100.0


def round_to_closest(arr, interval):
    """In-place rounding of array elements to the closest value in interval."""
    interval = np.asarray(interval)
    idx = np.abs(arr[..., np.newaxis] - interval).argmin(axis=-1)
    arr[...] = interval[idx]


def main(scene_h5, aerdb, modelpath):

    aernames = {
        "aermr01": "/SS1",
        "aermr02": "/SS2",
        "aermr03": "/SS3",
        "aermr04": "/DD1",
        "aermr05": "/DD2",
        "aermr06": "/DD3",
        "aermr07": "/OM_phil",
        "aermr08": "/OM_phob",
        "aermr09": "/BC_phil",
        "aermr10": "/BC_phob",
        "aermr11": "/SU",
    }

    # These map GEOS names -> CAMS variables and group them
    aergroups = {
        "SS": ["aermr01", "aermr02", "aermr03"],
        "DU": ["aermr04", "aermr05", "aermr06"],
        "OC": ["aermr07", "aermr08"],
        "BC": ["aermr09", "aermr10"],
        "SO4": ["aermr11"],
    }

    # Reverse lookup dict
    aerlookup = {aer: grp for grp, aers in aergroups.items() for aer in aers}

    logger.info(f"Processing scene file: {scene_h5.filename}")

    # Obtain scene time stamps
    scene_tai = scene_h5["Simulation/Time/tai"][:, 0, 0]
    n_scene = len(scene_tai)

    scene_times = num2date(
        scene_tai,
        units="Seconds since 1993-01-01",
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    # Determine required dates for CAMS model profiles
    ymd_min = scene_times.min()
    ymd_max = scene_times.max()
    if ymd_max.hour >= 21:
        ymd_max += timedelta(days=1)

    date_list = pd.date_range(
        ymd_min.strftime("%Y-%m-%d"), ymd_max.strftime("%Y-%m-%d"), freq="D"
    )

    filenames = [
        f"{modelpath}/{d.year}/{d.month:02d}/CAMS-aerosol-profiles-{d.strftime('%Y-%m-%d')}.nc"
        for d in date_list
    ]

    logger.info("Sampling CAMS files:")
    for fname in filenames:
        logger.info(f"  -> {fname}")
        if not os.path.exists(fname):
            logger.critical(f"File {fname} does not exist! Aborting.")
            sys.exit(1)

    # Dictionary to hold lists of arrays for deferred concatenation
    cams_lists = {k: [] for k in list(aernames.keys()) + ["time", "q"]}

    for i, fname in enumerate(filenames):
        with Dataset(fname, "r") as nc:
            if i == 0:
                level_var = "level" if "level" in nc.variables else "pressure_level"
                cams_lists["level"] = nc.variables[level_var][:] * 100.0
                cams_lists["longitude"] = nc.variables["longitude"][:]
                cams_lists["latitude"] = nc.variables["latitude"][:][::-1]

            time_var = "time" if "time" in nc.variables else "valid_time"
            _tmp_time = num2date(
                nc.variables[time_var][:],
                units=nc.variables[time_var].units,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )

            cams_lists["time"].append(_tmp_time)
            cams_lists["q"].append(nc.variables["q"][:][:, :, ::-1, :])

            for aer in aernames.keys():
                cams_lists[aer].append(nc.variables[aer][:][:, :, ::-1, :])

    logger.info("Concatenating CAMS data...")
    cams_dict = {
        "level": cams_lists["level"],
        "longitude": cams_lists["longitude"],
        "latitude": cams_lists["latitude"],
        "time": np.concatenate(cams_lists["time"], axis=0),
        "q": np.concatenate(cams_lists["q"], axis=0),
    }

    for aer in aernames.keys():
        cams_dict[aer] = np.concatenate(cams_lists[aer], axis=0)

    cams_dict["tai"] = date2num(cams_dict["time"], units="Seconds since 1993-01-01")
    cams_dict["logp"] = np.log(cams_dict["level"])

    scene_nlayers = scene_h5["Simulation/Thermodynamic/num_layers"][:]
    if len(np.unique(scene_nlayers)) != 1:
        logger.critical("Sorry - scenes have varying layer numbers.")
        sys.exit(1)
    scene_nlayers = scene_nlayers.min()

    scene_plevels = scene_h5["Simulation/Thermodynamic/pressure_level"][
        :, : scene_nlayers + 1
    ]
    scene_tlevels = scene_h5["Simulation/Thermodynamic/temperature_level"][
        :, : scene_nlayers + 1
    ]
    scene_glayers = scene_h5["Simulation/Thermodynamic/gravity_layer"][
        :, :scene_nlayers
    ]

    scene_plevels[scene_plevels == 0.0] = 1.0
    scene_logp = np.log(scene_plevels)

    scene_lat = scene_h5["Simulation/Geometry/latitude"][:, 0, 0]
    scene_lon360 = scene_h5["Simulation/Geometry/longitude"][:, 0, 0]
    scene_lon360 = np.where(scene_lon360 < 0, scene_lon360 + 360.0, scene_lon360)

    sample_coords = np.array(
        [
            (scene_tai[i], scene_logp[i, j], scene_lat[i], scene_lon360[i])
            for i in range(n_scene)
            for j in range(scene_nlayers + 1)
        ]
    )

    logger.info("Sampling specific humidity and aerosols...")
    grid_axes = (
        cams_dict["tai"],
        cams_dict["logp"],
        cams_dict["latitude"],
        cams_dict["longitude"],
    )

    # Process Specific Humidity
    rgi_q = spi.RegularGridInterpolator(
        grid_axes, cams_dict["q"], bounds_error=False, fill_value=None
    )
    scene_sampled_q = rgi_q(sample_coords).reshape(n_scene, scene_nlayers + 1)

    scene_sampled_mmr = {}
    for aer in tqdm(aernames.keys(), desc="Sampling Aerosols"):
        rgi = spi.RegularGridInterpolator(
            grid_axes, cams_dict[aer], bounds_error=False, fill_value=None
        )
        sampled = rgi(sample_coords).reshape(n_scene, scene_nlayers + 1)
        sampled = np.maximum(sampled, 0.0)  # Compact zero-clamping

        # Adjust sea salt MMRs to dry air
        if "SS" in aernames[aer]:
            sampled /= 4.3
        scene_sampled_mmr[aer] = sampled

    # Free up memory
    del cams_dict, cams_lists

    # Prepare scene data
    species_id = scene_h5["Simulation/Aerosol/species_id"][:, :]
    species_density = scene_h5["Simulation/Aerosol/species_density"][:, :, :]
    species_num = scene_h5["Simulation/Aerosol/num_species"][:]

    new_density = np.zeros_like(species_density)
    new_id = np.full_like(species_id, b"none")

    # Remove non-water/ice aerosols
    for i in range(n_scene):
        idx_noaer = [
            j
            for j in range(species_density.shape[1])
            if species_id[i, j].startswith((b"water", b"ice"))
        ]
        new_density[i, : len(idx_noaer), :] = species_density[i, idx_noaer, :]
        new_id[i, : len(idx_noaer)] = species_id[i, idx_noaer]

    species_id, species_density = new_id, new_density

    q = 0.5 * (scene_sampled_q[:, 1:] + scene_sampled_q[:, :-1])
    scene_layerp = 0.5 * (scene_plevels[:, 1:] + scene_plevels[:, :-1])
    scene_layert = 0.5 * (scene_tlevels[:, 1:] + scene_tlevels[:, :-1])
    scene_layerrh = np.clip(sh2rh(q, scene_layerp, scene_layert), 0, 95)

    dP = scene_plevels[:, 1:] - scene_plevels[:, :-1]

    logger.info("Calculating total aerosol mass for groups and species...")
    scene_sampled_grp_mass = {grp: np.zeros(n_scene) for grp in aergroups.keys()}
    scene_sampled_aer_mass = {}

    for grp, aers in aergroups.items():
        for aer in aers:
            mmr_layer = 0.5 * (
                scene_sampled_mmr[aer][:, :-1] + scene_sampled_mmr[aer][:, 1:]
            )
            this_mass = (mmr_layer * dP / scene_glayers).sum(axis=1)
            scene_sampled_grp_mass[grp] += this_mass
            scene_sampled_aer_mass[aer] = this_mass

    # Scale the CAMS aerosols so the total mass matches the GEOS aerosols,
    # but the partitioning is taken from CAMS.

    # (first read-in of GEOS total aerosol mass)
    GEOS_total_mass = {}
    for grp, ears in aergroups.items():
        if f"{grp}_total_mass" in scene_h5["Simulation/Aerosol"].keys():
            GEOS_total_mass[grp] = scene_h5[f"Simulation/Aerosol/{grp}_total_mass"][:]

    if args.scale:
        for aer in scene_sampled_mmr.keys():
            grp = aerlookup[aer]

            # Skip this aerosol if we do not have a GEOS total mass
            if grp not in GEOS_total_mass.keys():
                continue

            logger.info(f"Scaling up CAMS aerosol {aer} from GEOS group {grp}")
            # Relative weight in CAMS per species
            relative_group_weight = (
                scene_sampled_aer_mass[aer] / scene_sampled_grp_mass[grp]
            )
            # Scale factor to move up species to GEOS
            scale_factor = GEOS_total_mass[grp] / scene_sampled_grp_mass[grp]
            # Scale the CAMS MMRs according to total mass scale factor..
            scene_sampled_mmr[aer] *= (
                scale_factor[:, np.newaxis] * relative_group_weight[:, np.newaxis]
            )
    else:
        logging.info("Not scaling aerosol masses!")

    rhs = aerdb.root.rh.read()
    if np.count_nonzero(rhs == 0.0) != 1:
        logger.critical("Error - exactly one RH needs to be 0!")
        sys.exit(1)

    round_to_closest(scene_layerrh, rhs)
    idx_rh = np.searchsorted(rhs, scene_layerrh)
    idx_rh[idx_rh > len(rhs) - 1] = 0  # Fix bounds

    particle_counts = {}
    tau_total = np.zeros(n_scene)
    tau_group = {grp: 0 for grp in aergroups.keys()}

    for aer, aer_mmr in scene_sampled_mmr.items():
        aer_node = aerdb.get_node(aernames[aer])
        rh_dep = aer_node.rh_dep.read()
        mmr_layer = 0.5 * (aer_mmr[:, :-1] + aer_mmr[:, 1:])

        if rh_dep:
            sigma_ext = aer_node.mie.read()[idx_rh, 0, 0] * 1e-12
            mec = aer_node.MEC.read()[idx_rh, 0]
        else:
            sigma_ext = aer_node.mie.read()[0, 0] * 1e-12
            mec = aer_node.MEC.read()[0]

        particle_counts[aer] = mec / sigma_ext * mmr_layer / scene_glayers * dP
        this_aod = mec * mmr_layer / scene_glayers * dP

        tau_total += this_aod.sum(axis=1)
        tau_group[aerlookup[aer]] += this_aod.sum(axis=1)

    for aer in scene_sampled_mmr.keys():
        logger.info(f"Processing species {aer} {aernames[aer]}")
        rh_dep = aerdb.get_node(aernames[aer]).rh_dep.read()

        for i in tqdm(range(n_scene), desc=f"Mapping {aer} to slots", leave=False):
            if not np.any(particle_counts[aer][i] > 0):
                continue

            un_rhs = (
                np.unique(scene_layerrh[i][particle_counts[aer][i] > 0.0])
                if rh_dep
                else [0]
            )
            none_idx = np.where((species_id[i] == b"none") | (species_id[i] == b""))[0]

            if len(none_idx) < len(un_rhs):
                logger.critical(
                    f"ERROR! No more space for scene #{i}. Extend slots or use fewer species."
                )
                sys.exit(1)

            start_idx = none_idx[0]
            skips = 0

            for j, this_rh in enumerate(un_rhs):
                mask = particle_counts[aer][i] > 0.0
                if rh_dep:
                    mask &= scene_layerrh[i] == this_rh

                rhs_where = np.where(mask)[0]

                if not len(rhs_where):
                    skips += 1
                    continue

                slot = start_idx + j - skips
                species_density[i, slot, rhs_where] = particle_counts[aer][i][rhs_where]

                name_suffix = f"_{int(this_rh):03d}" if rh_dep else ""
                species_id[i, slot] = f"{aernames[aer][1:]}{name_suffix}".encode()

    species_num[:] = ((species_id != b"none") & (species_id != b"")).sum(axis=1)

    logger.warning("OVERWRITING SCENE FILE CONTENTS WITH NEW AEROSOL DATA")

    scene_h5["Simulation/Aerosol/species_id"][:, :] = species_id
    scene_h5["Simulation/Aerosol/species_density"][:, :, :] = species_density
    scene_h5["Simulation/Aerosol/num_species"][:] = species_num

    aod_grp = scene_h5.require_group("/Simulation/Aerosol/AOD_predicted")
    for grp_name, data in tau_group.items():
        if grp_name in aod_grp:
            aod_grp[grp_name][:] = data
        else:
            aod_grp.create_dataset(grp_name, data=data)

    if "total" in aod_grp:
        aod_grp["total"][:] = tau_total
    else:
        aod_grp.create_dataset("total", data=tau_total)

    logger.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add CAMS aerosols to scene file.")
    parser.add_argument(
        "--scene", help="Path to single scene file.", type=str, required=True
    )
    parser.add_argument(
        "--aerdb", help="Path to CAMS aerosol database.", type=str, required=True
    )
    parser.add_argument(
        "--modelpath",
        help="Path to CAMS model profiles (top-level)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        default=False,
        help="Scale up CAMS aerosol mass to match /Simulation/Aerosol/??_total_mass values?",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.modelpath):
        logger.critical(f"Model path: {args.modelpath} is not a valid directory!")
        sys.exit(1)

    try:
        scene_h5 = h5py.File(args.scene, "r+")
    except Exception as e:
        logger.critical(f"Error opening scene file: {e}")
        sys.exit(1)

    try:
        aerdb = tb.open_file(args.aerdb, "r")
    except Exception as e:
        logger.critical(f"Error opening aerosol database file: {e}")
        sys.exit(1)

    logger.info("Initial paths and files verified. Moving on...")

    try:
        main(scene_h5, aerdb, args.modelpath)
    finally:
        scene_h5.close()
        aerdb.close()
