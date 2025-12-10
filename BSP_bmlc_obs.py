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
        title_lines = [line.lstrip("#").strip() for line in header_lines[:4]]
        title = "\n".join(title_lines)
        title_lines = [line.lstrip("#").strip() for line in header_lines[:4]]
        columns_line = header_lines[-1].lstrip("#").strip()
    return title, columns_line


def merge_duplicates(time, flux, flux_err=None):
    """
    merge duplictaes according to the mean value of fluxes and flux errors at the same time points. 
    """
    # there are different flux values with same time points for different actions.
    # So merges should be done.
    time = np.array(time)
    flux = np.array(flux)
    if flux_err is not None:
        flux_err = np.array(flux_err)

    uniq_time, inv_idx, counts = np.unique(time, return_inverse=True, return_counts=True)
    flux_sum = np.bincount(inv_idx, weights=flux)
    flux_mean = flux_sum / counts
    
    err_sum = np.bincount(inv_idx, weights=flux_err**2)
    flux_err_mean = np.sqrt(err_sum) / counts
    else:
        flux_err_mean = None

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

    for line in lines:
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
    title_text, column_name = extract_title(obs)
    colnames = column_name.split()
    df2 = pd.read_csv(obs, sep=r"\s+", comment="#", names=colnames)

    # Step3: Data processing for observed data： sigma-clipping
    median = np.median( df2['FluxNorm'] )
    std = np.std( df2['FluxNorm'] )
    mask_good = np.abs( df2['FluxNorm'] - median) < 3 * std
    obsflux = df2['FluxNorm'][mask_good]
    obsbjd  = df2['BJD'][mask_good]
    obsfluxerr = df2['FluxNormErr'][mask_good]

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
    # 1. Check model file
    if not os.path.exists(model_file):
        logger.error(f"[PLOT] Model file does not exist: {model_file}")
        return
    else:
        logger.info(f"[PLOT] Found model file: {model_file}")

    # 2. Locate OBS LC file (obs is in parent directory of outdir)
    # delete the “/" first, the obs lc locates in analyses_outputs, without nightinfo.
    outdir = outdir.rstrip("/")
    obs_search_dir = os.path.dirname(outdir)
    logger.info(f"[PLOT] Searching obs LC in: {obs_search_dir}")
    obs_pattern = os.path.join(obs_search_dir, "*master_apers_bsproc_lc.dat")
    obs_files = glob.glob(obs_pattern)

    if len(obs_files) == 0:
        logger.error(f"[PLOT] No obs LC found in: {outdir}")
        return
    elif len(obs_files) > 1:
        logger.warning(f"[PLOT] Multiple obs LC found, using first: {obs_files[0]}")

    obs_file = obs_files[0]
    logger.info(f"[PLOT] Found obs LC: {obs_file}")

    # 3. Create directory for saving model related results
    figure_dir = os.path.join(outdir, "model_plots")
    os.makedirs(figure_dir, exist_ok=True)
    output_png = os.path.join(figure_dir, f"{ticid}_{night}_model_vs_obs.png")
    logger.info(f"[PLOT] Saving plot to: {figure_dir}")

    # 4. Call plot function
    try:
        bin_step = 0.004
        plot_data(model_file, obs_file, output_png, bin_step)
        logger.info(f"[PLOT] Saved figure: {output_png}")
    except Exception as e:
        logger.error(f"[PLOT] Plotting failed for TIC {ticid} on night {night}: {e}")

    logger.info("[PLOT] Plotting completed.")

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
    if isinstance(model_files, str):
        model_files = [model_files]

    if len(nights) != len(model_files):
        logger.error("[PLOT] nights and model_files length mismatch!")
        return

    for night, modelfile in zip(nights, model_files):

        nightly_outdir = night_outdir_dict[night]

        logger.info(f"[PLOT] Processing night {night}, dir={nightly_outdir}")
        moplot_single(logger, nightly_outdir, ticid, night, modelfile)
