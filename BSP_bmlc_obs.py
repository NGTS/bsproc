import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def extract_title(obsfilename):
    # Read the title in comments and column_names 
    header_lines = []
    with open(obsfilename) as f:
        for _ in range(6):
            line = f.readline()
            if not line:
                break
            header_lines.append(line.strip())
    keep_idx = [0, 2, 3]
    title_lines = [
        header_lines[i].lstrip("#").strip()
        for i in keep_idx
        if i < len(header_lines)
    ]
        
    title = "\n".join(title_lines)
    columns_line = header_lines[-1].lstrip("#").strip()
    return title, columns_line


def merge_duplicates(time, flux, flux_err):
    """
    Merge duplicated time points by averaging fluxes and flux errors.
    """
    time = np.array(time)
    flux = np.array(flux)
    flux_err = np.array(flux_err)

    uniq_time, inv_idx, counts = np.unique(time, return_inverse=True, return_counts=True)

    # flux mean
    flux_sum = np.bincount(inv_idx, weights=flux)
    flux_mean = flux_sum / counts

    # error propagation
    err_sum = np.bincount(inv_idx, weights=flux_err**2)
    flux_err_mean = np.sqrt(err_sum) / counts

    return uniq_time, flux_mean, flux_err_mean

    
def plot_data(model, obs, op_name, bin_step=0.004):
    """
    The function read data, processing data with sigma-clipping, merging duplicates,
    binning, time-axis adjusting. Then Plot.               
    """
    # Step1： Read model file
    with open(model, 'r') as f:
        lines = f.readlines()
    #print(lines[:5])

    for i, line in enumerate(lines):
        if not line.startswith("#"):
            break   
        if i == 0:
            title1 = line.lstrip("#").strip()
            continue
        if line.startswith("# Transit midtime"):
            Tc = float(line.split("=")[1])

        if line.startswith("# T1"):
            T1 = float(line.split("=")[1])

        if line.startswith("# T4"):
            T4 = float(line.split("=")[1])
        # save model data to dataframe, skipping first 4 comment rows   
    df1 = pd.read_csv(model, comment = "#")

    # Step2: Read observed data file 
    # Skip the last row which contains a note
    title, column_name = extract_title(obs)
    title_text= f"{title}\n{title1}"
    colnames = column_name.split()
    df2 = pd.read_csv(obs, sep=r"\s+", comment="#", names=colnames)

    # Step3: Data processing for observed data： sigma-clipping
    median = np.median( df2['FluxNorm'] )
    std = np.std( df2['FluxNorm'] )
    mask_good = np.abs( df2['FluxNorm'] - median) < 3 * std
    # Step3.5: we cut the obs to the night in predicted model
    time0 = df1["BJD"].iloc[0]
    time1 = df1["BJD"].iloc[-1]

    margin = 0.1 # days, enough to constrain the obs_lc for one night.
    mask_night = (df2['BJD'] >= (time0 - margin)) & (df2['BJD'] <= (time1+ margin))
    mask_obs = mask_good & mask_night

    obsflux = df2['FluxNorm'][mask_obs]
    obsbjd  = df2['BJD'][mask_obs]
    obsfluxerr = df2['FluxNormErr'][mask_obs]
  
    # Step4: Merge duplicate time points
    time, flux, fluxerr = merge_duplicates(obsbjd, obsflux, obsfluxerr)

    # Step5: calculate binned data
    bin_width = bin_step  # in days
    edges = np.arange(np.min(time), np.max(time), bin_width)
    dig = np.digitize(time, edges)

    time_binned = (edges[1:] + edges[:-1]) / 2
    flux_binned = np.array([np.nan if len(flux[dig == i]) == 0 else flux[dig == i].mean() 
                            for i in range(1, len(edges))])
    err_binned = np.array([np.nan if len(flux[dig == i]) == 0 else np.sqrt(np.sum(fluxerr[dig==i]**2))/len(fluxerr[dig==i]) 
                        for i in range(1, len(edges))])
    time_bin = time_binned[~np.isnan(err_binned)]
    err_bin = err_binned[~np.isnan(err_binned)]
    flux_bin = flux_binned[~np.isnan(err_binned)]

    # Step6: Adjust the time range for plot
    bjd0 = int(df1["BJD"].mean())   # or min()
    t_obs = obsbjd - bjd0
    t_model = df1["BJD"] - bjd0
    t_bin = time_bin - bjd0

    # Step7: Plotting
    plt.figure(figsize=(10, 7))
    plt.plot(t_obs,obsflux,".",label='Observed Data', color ="black",alpha=0.2, zorder=0, markersize=4)
    plt.errorbar(t_bin, flux_bin, yerr=err_bin, fmt='o', markersize=6,  color='blue', ecolor="c", label='Binned Data',zorder=1)
    plt.plot(t_model, df1['Flux'], '-', label='Model Data', linewidth=2, color='red', zorder=2)
    plt.axvline(T1-bjd0, color='orange', linestyle='--', label='T1')
    plt.axvline(Tc-bjd0, color='green', linestyle='--', label='Tc')
    plt.axvline(T4-bjd0, color='orange', linestyle='--', label='T4')
    plt.title(title_text)
    plt.xlabel(f'Time (BJD - {bjd0}) Days', fontsize=14)
    plt.ylabel('Normalised Flux', fontsize=14)
    plt.legend()
    plt.savefig(op_name, dpi = 500)
    plt.show()  
    plt.close()

def moplot_single(logger, outdir, ticid, night, model_file):
    """
    This function works for a given night.
    Read model LC + obs LC, create plot directory, and call plot function. 
    """
    # 1. Unpack the model_file, which is a dict
    plname = model_file["planet"]
    model_csv = model_file["file"]
    tc = model_file["tc"]
    T1 = model_file["T1"]
    T4 = model_file["T4"]

    # 2. check the model_csv
    if not os.path.exists(model_csv):
        logger.error(f"[PLOT] Model file does not exist: {model_csv}")
        return
    else:
        logger.info(f"[PLOT] Found model file: {model_csv}")

    # 3. Locate OBS LC file (obs is in parent directory of outdir)
    # delete the “/" first, the obs lc locates in analyses_outputs, without nightinfo.
    outdir = outdir.rstrip("/")
    obs_search_dir = os.path.dirname(outdir)
    logger.info(f"[PLOT] Searching obs LC in: {obs_search_dir}")
    obs_pattern = os.path.join(obs_search_dir,  f"*{night}*master_apers_bsproc_lc.dat")
    obs_files = glob.glob(obs_pattern)

    if len(obs_files) == 0:
        logger.error(f"[PLOT] No obs LC found in: {outdir}")
        return
    elif len(obs_files) > 1:
        logger.warning(f"[PLOT] Multiple obs LC found, using first: {obs_files[0]}")

    obs_file = obs_files[0]
    logger.info(f"[PLOT] Found obs LC: {obs_file}")

    # 4. Create directory for saving model-related results
    figure_dir = os.path.join(outdir, "model_plots")
    os.makedirs(figure_dir, exist_ok=True)
    output_png = os.path.join(figure_dir, f"{plname}_{night}_model_vs_obs.png")
    logger.info(f"[PLOT] Saving plot to: {figure_dir}")

    # 5. Call plot function
    try:
        bin_step = 0.004
        plot_data(model_csv, obs_file, output_png, bin_step)
        logger.info(f"[PLOT] Saved figure: {output_png}")
        logger.info("[PLOT] Plotting completed.")

    except Exception as e:
        logger.error(f"[PLOT] Plotting failed for TIC {ticid} on night {night}: {e}")

   
# Example usage:
#modelfile= "/Users/urnotlizzy/Downloads/139528693_2025-11-11_model.csv"
#obsfilefile= "/Users/urnotlizzy/Downloads/NGTS_TIC-139528693_2025-11-10_2025-11-11_2025-11-12.txt"
#output_dir = "/Users/urnotlizzy/Documents/code/bsproc"

def moplot(logger, night_outdir_dict, ticid, nights, model_files):
    """
    This function is written in bsproc. 
    """
    # bsproc read nights instead of night, this function expand the single night plot to the multi-night one.
    if isinstance(nights, str):
        nights = [nights]
    for night in nights:
        if night not in night_outdir_dict:
            logger.warning(f"[PLOT] Night {night} not found in night_outdir_dict, skipping.")
            continue

        if night not in model_files:
            logger.warning(f"[PLOT] No model files for night {night}, skipping.")
            continue

        nightly_outdir = night_outdir_dict[night]
        logger.info(f"[PLOT] Processing night {night}, dir={nightly_outdir}")

        for plname, modelfile in model_files[night].items():

            logger.info(f"[PLOT] Plotting planet {plname}")

            moplot_single(
                logger,
                nightly_outdir,
                ticid,
                night,
                modelfile
            )
