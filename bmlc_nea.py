import batman
import pandas as pd
import numpy as np
import QueryNEA as QNEA
import os
import astropy.io.fits as pyfits

def get_bjd_range(actions):
    all_bjds = []

    for action_id in actions:
        try:
            bjds = get_ngpipe_bjd(action_id)
            all_bjds.append(bjds)
        except Exception as e:
            print(f"[BMLC]Failed in fetching bjd time for actions:{e}")

    if len(all_bjds) == 0:
        print("[BMLC] No valid BJD data found for any action.")
        return None, None
    bjds = np.concatenate(all_bjds)
    t_start = np.min(bjds)
    t_end   = np.max(bjds)

    return t_start, t_end

## read the BJDs using action_id 
def get_ngpipe_bjd(ac_id, ngpipe_op_dir = "/ngts/PAOPhot2/"):
    """
    Get BJD for NGTS action
    """
    phot_file_dir = ngpipe_op_dir + f'photometry/action{ac_id}/'
    if os.path.exists(phot_file_dir):
        print('Found photometry directory')
        phot_file_root = phot_file_dir + f'ACTION_{ac_id}_'
    else:
        print(f'Can\'t find photometry for Action {ac_id}')
        print('Trying "old" directory')
        phot_file_dir = ngpipe_op_dir + f'bs_photometry/action{ac_id}/'
        if os.path.exists(phot_file_dir):
            print('Found "old" photometry directory')
            phot_file_root = phot_file_dir + f'ACTION_{ac_id}_'
        else:
            print(f'No photometry for Action {ac_id}')
            print(f'Skipping Action {ac_id}.')
            return None, None, None, None, None, None, None, None
    # BJD from ngpipe is saved in seconds from a specific time.
    #  The correction applied is to convert it to human understandable units
    try:
        bjds = pyfits.getdata(phot_file_root+'BJD.fits.bz2') / 86400. + 2456658.5
    except:
        bjds = pyfits.getdata(phot_file_root+'BJD.fits') / 86400. + 2456658.5
    try:
        fluxes = pyfits.getdata(phot_file_root+'FLUX.fits.bz2')
    except:
        fluxes = pyfits.getdata(phot_file_root+'FLUX.fits')
    
    #try:
    #    airmass_0 = pyfits.getdata(phot_file_root+'IMAGELIST.fits.bz2')
    airmass_hdu_0 = pyfits.open(phot_file_root+'IMAGELIST.fits')
    airmass_0 = np.array(airmass_hdu_0[1].data['AIRMASS'])

    target_flux0 = np.copy(fluxes[0][:, 0])    # select target array for 1 aperture to check for bad images
    bjd_keep_good_timestamps = (target_flux0 > 0.)
    
    airmass_keep = airmass_0 > 0.99
    if np.sum(airmass_keep) <= 0.2 * len(airmass_0):
        # If an image fails in ngpipe it is given airmass = 0
        # So if more than 80% of the airmass array is < 1 then 80% of the images have failed
        print('Fewer than 20% of the airmass array is above 1.')
        print('Skipping action.')
        return None
    keep =  bjd_keep_good_timestamps & airmass_keep
    target_bjd = np.copy(bjds[0])[keep]
    return target_bjd


def predict_transit_curve(row, t_start, t_end):
    """
    row: a single-row Series from NEA containing pl_orbper, pl_tranmid, pl_ratror, pl_ratdor, pl_imppar ...
    """

    # 1. closest tc = t0 + nP
    P = row['pl_orbper']
    t0 = row['pl_tranmid']
    n = round((t_start - t0) / P)
    tc = t0 + n * P

    # 3. build Batman model based on tc
    params = batman.TransitParams()
    params.t0  = tc
    params.per = P
    params.rp  = row['pl_ratror']
    params.a   = row['pl_ratdor']
    b   = row['pl_imppar']
    aRs = row['pl_ratdor']
    ratio = np.clip(b / aRs, -1, 1)       # avoid strange error in arccos funcion
    params.inc = np.degrees(np.arccos(ratio))
    params.ecc = 0
    params.w   = 90
    params.u   = [0.1, 0.3]
    params.limb_dark = "quadratic"

    # 4. calculate the theoretical model
    t = np.linspace(t_start, t_end, 1000)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)
    return t, flux, tc

def save_transit_csv(t, flux, tc, ticid, night, outdir = None,logger=None):
    if outdir is None:
        logger.info("Can\'t find the output directory")
        outdir = f"./bsproc_outputs/{ticid}/{night}/model/"
        logger.info("[BMLC]Create an output directory:{outdir}")
    os.makedirs(outdir, exist_ok=True)
   
    filename = os.path.join(outdir, f"{ticid}_{night}_model.csv")
    with open(filename, "w") as f:
        f.write(f"# TIC {ticid}, Night {night}\n")
        f.write(f"# Transit midtime (Tc) = {tc:.10f}\n")
        f.write("BJD,FLUX\n")
        for ti, fi in zip(t, flux):
            f.write(f"{ti},{fi}\n")
    

#ticid = "276754403"
#night = "2025-07-15"
#action_id = (get_actionid(ticid,night) or [None])[0]
#bjds = get_ngpipe_bjd(action_id)
#t_start = bjds.min()
#t_end   = bjds.max()
#t, flux, tc= predict_transit_curve(ticid, t_start, t_end)
#save_transit_csv(t, flux, tc, ticid, night)

def tranmodel(actionlist, ticid, nights, night_outdir_dict, logger= None):
    #1. call query and prepare parameters for Batman
    df = QNEA.query_params_NEA(ticid)
    if df is None or len(df) == 0:
        logger.error(f"No NEA parameters found for TIC {ticid}")
        return None
    
    row = df.iloc[0]
    results = []
    for night in nights:
        #2. Find the actionids for each night in actionlist
        actions_onen = actionlist.loc[actionlist["night"]==night,"action_id"].tolist()
        if len(actions_onen)==0:
                logger.info(f"[BMLC]No actions found on night:{night}")
        outdir = night_outdir_dict[night]

        night = str(night)
        # 3. from bspd : find_target_actions, actions have been searched,  
        t_start, t_end = get_bjd_range(actions_onen)

        logger.info(f"FOR ACTION: {actions_onen} ----Start BJD: {t_start}, End BJD: {t_end}")

        # 4. batman prediction
        bjdt, flux, tc = predict_transit_curve(row, t_start, t_end)

        # 5. save output
        save_transit_csv(bjdt, flux, tc, ticid, night, outdir, logger)
        logger.info(f"[BMLC] Saved predicted light curve for {ticid} on {night}")
        results.append((night, bjdt, flux, tc))

    return results

