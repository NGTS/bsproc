import batman
import numpy as np
import QueryNEA as QNEA
import os
import glob
import astropy.io.fits as pyfits

def get_bjd_range(actions,logger):
    all_bjds = []

    for action_id in actions:
        try:
            bjds = get_ngpipe_bjd(action_id,logger)
            all_bjds.append(bjds)

        except Exception as e:
            logger.error(f"[BMLC]Failed in fetching bjd time for actions:{e}")

    if len(all_bjds) == 0:
        logger.info("[BMLC] No valid BJD data found for any action.")
        return None, None
    bjds = np.concatenate(all_bjds)
    t_start = np.min(bjds)
    t_end   = np.max(bjds)

    return t_start, t_end

## read the BJDs using action_id 
def get_ngpipe_bjd(ac_id,logger=None, ngpipe_op_dir = "/ngts/PAOPhot2/"):
    """
    Get BJD for NGTS action
    """
    phot_file_dir = ngpipe_op_dir + f'photometry/action{ac_id}/'
    if os.path.exists(phot_file_dir):
        logger.info(f'Found BJD for action {ac_id} in photometry directory.')
        phot_file_root = phot_file_dir + f'ACTION_{ac_id}_'
    else:
        logger.info(f'Can\'t find photometry for Action {ac_id}')
        logger.info('Trying searching BJD "old" directory')
        phot_file_dir = ngpipe_op_dir + f'bs_photometry/action{ac_id}/'
        if os.path.exists(phot_file_dir):
            logger.info(f'Found BJD for action {ac_id} in "old" photometry directory')
            phot_file_root = phot_file_dir + f'ACTION_{ac_id}_'
        else:
            logger.info(f'No photometry for Action {ac_id}')
            logger.info(f'Skipping Action {ac_id}.')
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

def predict_transit_curve(row, t_start, t_end, logger=None):
    # 0. read the parameters out and set as nan if not exist.
    def get_val(key, default=np.nan):
        return row.get(key, default) if not np.isnan(row.get(key, default)) else default

    P = get_val('pl_orbper')        
    t0_data = get_val('pl_tranmid') 
    k = get_val('pl_ratror', 0.1)   # Rp/Rs，default is 0.1
    b = get_val('pl_imppar', 0.0)   # impact parameter, default is 0.0
    w = get_val('pl_orblper', 90.0)  # argument of periastron
    # stellar parameters
    m_star = get_val('st_mass')     # unit: solar mass
    r_star = get_val('st_rad')      # unit: solar radius
    
    # dealing with t0
    if np.isnan(t0_data):
        if logger: logger.warning("[BMLC] No effective 'pl_tranmid', using the observational midtime as t0")
        t0 = (t_start + t_end) / 2
    else:
        t0 = t0_data

    # 1. calculate closest tc = t0 + nP
    # n is the closest epoch to obs
    n = round((t_start - t0) / P)
    tc = t0 + n * P

    # 2. derive aRs, the priority: 
    #a. read from "pl_ratdor"; 
    #b. calculate with m_star, s_star using Kepler 3rd law; 
    #c. calculate with known tran_duration time (this may be better than b? deserves more consideration)
	#d. set a default value for it to keep the model generating code running.
    aRs = np.nan
    # a.
    aRs = get_val('pl_ratdor')
    # c.
    if np.isnan(aRs) and not np.isnan(get_val('pl_trandur')):
        dur = row['pl_trandur'] / 24.0 # hours to days
        try:
            #sin(delta_phi) = sqrt((1+k)^2 - b^2) / (aRs * sin(i))
            sin_term = np.sin(dur * np.pi / P)
            sin_term = np.clip(np.sin(dur * np.pi / P), 1e-6, 1.0)
            aRs = np.sqrt(((1 + k)**2 - b**2) / sin_term**2 + b**2)
            if logger: logger.info(f"[BMLC] transit duration deriving a/R*: {aRs:.2f}")
        except:
            pass

    # b.
    if np.isnan(aRs) and not np.isnan(m_star) and not np.isnan(r_star):
        # a/Rs = [ (G * M_sun * P^2) / (4 * pi^2) ]^(1/3) / (R_sun * r_star)
        # a/Rs = 4.205 * (m_star)^(1/3) * (P_days)^(2/3) / r_star
        aRs = 4.205 * (m_star**(1/3)) * (P**(2/3)) / r_star
        if logger: logger.info(f"[BMLC] Kepler 3rd law deriving a/R*: {aRs:.2f}")

    # d.
    if np.isnan(aRs):
        aRs = 50.0
        if logger: logger.warning("[BMLC] Cannot derive a/Rs, using default value: 50")

    # 3. calculate the inclination
    # b = aRs * cos(i)
    cos_i = np.clip(b / aRs, 0, 1.0)
    inc_deg = np.degrees(np.arccos(cos_i))
    incl = np.radians(inc_deg)

    # 4. build Batman model based on tc
    params = batman.TransitParams()
    params.t0  = tc
    params.per = P
    params.rp  = k
    params.a   = aRs
    params.inc = inc_deg
    params.ecc = get_val('pl_orbeccen', 0.0)
    params.w   = w 
    params.u   = [0.1, 0.3] 
    params.limb_dark = "quadratic"
		
    t = np.linspace(t_start, t_end, 1000)
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)
    
    # calculate transit duration using the model in_transit time, or read the archive duration
    # Some of params may be nan,then T1,T4 will be returned as nan. 
       
    T14 = get_val('pl_trandur')
    if not np.isnan(T14):
        T14 /= 24.0  # hours to days
        logger.info(f"[BMLC]The T14 = {T14} coming from the transit duration value in archive.")

    else:
        sin_i = np.sin(incl)
        inner_val = (1.0/aRs) * np.sqrt((1 + k)**2 - b**2) / sin_i
        inner_val = np.clip(inner_val, -1, 1) 
        T14 = (P / np.pi) * np.arcsin(inner_val)
        if not np.isnan(T14): logger.info(f"[BMLC]calcating T14 = {T14}")

    dt = T14 / 2 
    return t, flux, tc, tc - dt, tc + dt

def save_transit_csv(t, flux, tc, T1, T4, ticid, night, plname, actions_onen, logger, outdir = None):
    if outdir is None:
        outdir = f"./bsproc_outputs/{ticid}/{night}/models/{plname}"
        logger.info(f"[BMLC] No outdir provided, using default: {outdir}")
    
    os.makedirs(outdir, exist_ok=True)
   
    filename = os.path.join(outdir, f"TIC_{ticid}_{night}_{plname}_model.csv")
    with open(filename, "w") as f:
        # Comment header 
        f.write(f"# Hostname: TIC {ticid}, Night: {night}, Planet: {plname}\n")
        f.write(f"# Transit midtime (Tc) = {tc}\n")
        f.write(f"# Actions: [{actions_onen}]\n")
        f.write(f"# T1 = {T1}\n") 
        f.write(f"# T4 = {T4}\n")

        # Column header
        f.write("BJD,Flux\n")

        for ti, fi in zip(t, flux):
            f.write(f"{ti},{fi}\n")
    return {
    "night": night,
    "planet": plname,
    "file": filename,
    "tc": tc,
    "T1": T1,
    "T4": T4,
}
    
def tranmodel(actionlist, ticid, nights, night_outdir_dict, logger= None):
    #1. call query and prepare parameters for Batman
    df = QNEA.query_params_NEA(ticid)
    if df is None or len(df) == 0:
        logger.error(f"No NEA parameters found for TIC {ticid}")
        return None
    
    row = df.iloc[0]
    model_files = {}
    for _, row in df.iterrows():
        pl_name = row["pl_name"]
        logger.info(f"[BMLC] Processing planet {pl_name}.")
        for night in nights:
            #2. Find the actionids for each night in actionlist
            actions_onen= actionlist['action_id'][ actionlist['night'] == night ].to_numpy()
            outdir = night_outdir_dict[night]
            night = str(night)
            # 3. from bspd : find_target_actions, actions have been searched,  
            t_start, t_end = get_bjd_range(actions_onen, logger)

            logger.info(f"For planet {pl_name}, For actions: {actions_onen} ----Start BJD: {t_start}, End BJD: {t_end}")

            # 4. batman prediction
            bjd, flux, tc, T1, T4 = predict_transit_curve(row, t_start, t_end, logger)

            # 5. save output  
            model_dir = os.path.join(outdir, "models", pl_name)
            os.makedirs(model_dir, exist_ok=True)

            model_file = save_transit_csv(
                bjd, flux, tc, T1, T4,
                ticid, night, pl_name, actions_onen, 
                logger,
                model_dir
            )
            if night not in model_files:
                model_files[night] = {}

            model_files[night][pl_name] = model_file

            logger.info(f"[BMLC] Saved model for {ticid} {pl_name} on night {night}")

    return model_files


def forcemodel(actionlist, ticid, nights, night_outdir_dict, logger= None):
    #1. call query and prepare parameters for Batman
    df = QNEA.get_ephem_from_tess_portal(ticid)
    if df is None or len(df) == 0:
        logger.error(f"No  parameters found in TESS_portal.ephems for TIC {ticid}")
        return None
    
    row = df.iloc[0]
    model_files = {}
    for _, row in df.iterrows():
        pl_name = row["pl_name"]
        logger.info(f"[BMLC] Processing planet {pl_name}.")
        for night in nights:
            #2. Find the actionids for each night in actionlist
            actions_onen= actionlist['action_id'][ actionlist['night'] == night ].to_numpy()
            outdir = night_outdir_dict[night]
            night = str(night)
            # 3. from bspd : find_target_actions, actions have been searched,  
            t_start, t_end = get_bjd_range(actions_onen, logger)

            logger.info(f"For planet {pl_name}, For actions: {actions_onen} ----Start BJD: {t_start}, End BJD: {t_end}")

            # 4. batman prediction
            bjd, flux, tc, T1, T4 = predict_transit_curve(row, t_start, t_end, logger)

            # 5. save output  
            model_dir = os.path.join(outdir, "models", pl_name)
            os.makedirs(model_dir, exist_ok=True)

            model_file = save_transit_csv(
                bjd, flux, tc, T1, T4,
                ticid, night, pl_name, actions_onen, 
                logger,
                model_dir
            )
            if night not in model_files:
                model_files[night] = {}

            model_files[night][pl_name] = model_file

            logger.info(f"[BMLC] Saved model for {ticid} {pl_name} on night {night}")

    return model_files

def collect_model(actionlist, ticid, nights, night_outdir_dict, source=None, logger=None):
    """
    Collect or generate model files for plotting.
    For now, only 'nea' and 'ephem' sources are supported.
    """
    if source == 'nea':
        if logger:
            logger.info(f"[PLOT] Collecting model files from NEA source for TIC {ticid}")
        model_files = tranmodel(actionlist, ticid, nights, night_outdir_dict, logger)
    
    elif source == 'ephem':
        if logger:
            logger.info(f"[PLOT] Collecting model files from Ephemeris source for TIC {ticid}")
        model_files = forcemodel(actionlist, ticid, nights, night_outdir_dict, logger)
    
    else:
        if logger:
            logger.warning(f"[PLOT] Source '{source}' not yet implemented. Returning empty dict.")
        model_files = {}  ## TBC

    return model_files
