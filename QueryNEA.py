
####The file is prepared for final updates on bsproc in deriving parameters for making batman models.
import numpy as np
import pandas as pd
import requests
import io

G_cgs      = 6.67430e-8
Rsun_cm    = 6.957e10
Msun_g     = 1.98847e33
day_to_sec = 86400.0


def to_float(x, name):
    """Safe float conversion with warning."""
    try:
        val = float(x)
    except:
        print(f"[WARNING] {name} invalid: {x}")
        return np.nan
    
    if np.isnan(val):
        print(f"[WARNING] {name} is NaN")
        return np.nan
    return val

def compute_ratror(pl_trandep):
    dep = to_float(pl_trandep, "pl_trandep")
    if np.isnan(dep):
        return np.nan
    if dep < 0:
        print(f"[WARNING] pl_trandep negative: {dep}")
        return np.nan
    return np.sqrt(dep * 1e-6) # convert ppm to fraction

def compute_Mstar_from_logg_rad(st_logg, st_rad):
    """
    Compute stellar mass from logg and radius. 

    Parameters
    ----------              
    st_logg : float
        Stellar surface gravity in cgs (log10(cm/s^2)).         
    st_rad : float
        Stellar radius in solar radii.  
    Returns 
    -------     
    Mstar : float
        Stellar mass in solar masses.   
    """
    logg_val = to_float(st_logg, "st_logg")
    rad_val  = to_float(st_rad,  "st_rad")

    if np.isnan(logg_val) or np.isnan(rad_val):
        return np.nan

    g = 10**logg_val
    R_cm = rad_val * Rsun_cm
    M_g  = g * R_cm**2 / G_cgs
    return M_g / Msun_g

def compute_Mstar_from_aRs(aRs, st_rad, P_days):
    """
    Compute stellar mass from a/Rstar, stellar radius, and orbital period.
    
    Parameters
    ----------
    aRs : float
        a / R_star
    st_rad: float
        stellar radius in units of R_sun
    P_days : float
        orbital period in days

    Returns
    -------
    M_star_Msun : float
        Stellar mass in units of M_sun
    """
    # convert inputs
    Rstar_cm = st_rad * Rsun_cm
    a_cm = aRs * Rstar_cm
    P_sec = P_days * day_to_sec

    # Kepler's 3rd law: a^3 = G M P^2 / (4 pi^2)
    M_star_g = 4 * np.pi**2 * a_cm**3 / (G_cgs * P_sec**2)

    # convert to solar mass
    return M_star_g / Msun_g

def compute_aRs_from_kepler(P_days, Mstar, Rstar):
    """
    Compute a/Rstar from Kepler's 3rd law.
    Parameters
    ----------
    P_days : float
        Orbital period in days. 
    Mstar : float
        Stellar mass in solar masses.
    Rstar : float
        Stellar radius in solar radii.
    Returns 
    ------- 
    aRs : float
        Semi-major axis in units of stellar radii.
    """
    P = to_float(P_days, "pl_orbper")
    M = to_float(Mstar,  "st_mass")
    R = to_float(Rstar,  "st_rad")

    if np.isnan(P) or np.isnan(M) or np.isnan(R):
        return np.nan

    P_sec = P * day_to_sec
    a_cm = (G_cgs * (M*Msun_g) * P_sec**2 / (4*np.pi**2))**(1/3)
    return a_cm / (R * Rsun_cm)

def compute_aRs_from_TPk_b0(T_hours, P_days, k):
    """
    Compute a/Rstar from transit duration, period, and radius ratio.
    Parameters
    ----------
    T_hours : float
        Transit duration in hours.
    P_days : float
        Orbital period in days.
    k : float
        Radius ratio (Rp/Rs).
    Returns
    -------
    aRs : float
        Semi-major axis in units of stellar radii.
    """
    T = to_float(T_hours, "pl_trandurh")
    P = to_float(P_days,  "pl_orbper")
    k = to_float(k,       "ratror")

    if np.isnan(T) or np.isnan(P) or np.isnan(k):
        return np.nan

    T_days = T / 24.0
    x = np.pi * T_days / P
    s = np.sin(x)
    if s <= 0:
        print("[WARNING] sin(x)<=0 in a/R* calculation")
        return np.nan

    return (1 + k) / s

def compute_b_from_TPkaRs(T_hours, P_days, k, aRs):
    """
    Compute impact parameter from transit duration, period, radius ratio, and a/Rstar.
    Parameters
    ----------
    T_hours : float
        Transit duration in hours.
    P_days : float
        Orbital period in days.
    k : float
        Radius ratio (Rp/Rs).
    aRs : float
        Semi-major axis in units of stellar radii.
    Returns 
    -------
    b : float
        Impact parameter.
    """ 
    T = to_float(T_hours, "pl_trandurh")
    P = to_float(P_days,  "pl_orbper")
    k = to_float(k,       "ratror")
    aR = to_float(aRs,    "aRs")

    if np.isnan(T) or np.isnan(P) or np.isnan(k) or np.isnan(aR):
        return np.nan

    T_days = T / 24.0
    X = (T_days * np.pi / P) * aR
    term = (1 + k)**2 - X**2

    if term < 0:
        return np.nan

    return np.sqrt(term)

def compute_P_from_Tk_aRs_b(T_hours, k, aRs, b):
    """
    Compute orbital period from transit duration, radius ratio, a/Rstar, and impact parameter.
    Parameters
    ----------  
    T_hours : float
        Transit duration in hours.
    k : float
        Radius ratio (Rp/Rs).       
    aRs : float
        Semi-major axis in units of stellar radii.
    b : float
        Impact parameter.
    Returns
    -------
    P_days : float
        Orbital period in days.
    """
    T = to_float(T_hours, "pl_trandurh")
    k = to_float(k,       "ratror")
    aR = to_float(aRs,    "aRs")
    b  = to_float(b,      "b")

    if np.isnan(T) or np.isnan(k) or np.isnan(aR) or np.isnan(b):
        return np.nan

    num = np.sqrt((1+k)**2 - b**2)
    arg = num / aR

    if not (0 <= arg <= 1):
        return np.nan

    angle = np.arcsin(arg)
    if angle <= 0:
        return np.nan

    T_days = T / 24.0
    return (np.pi * T_days) / angle

def infer_transit_params(row):
    P_tab = row.get("pl_orbper",   np.nan)
    T_tab = row.get("pl_trandurh", np.nan)
    Rstar = row.get("st_rad",      np.nan)
    logg  = row.get("st_logg",     np.nan)
    dep   = row.get("pl_trandep",  np.nan)

    # ---- k ----
    k = compute_ratror(dep)

    # ---- Rstar ----
    R_val = to_float(Rstar, "st_rad")

    # 1) Mstar from logg (preferred)
    Mstar = compute_Mstar_from_logg_rad(logg, R_val)
    method_Mstar = "logg_rad"


    # 2) a/R* using Kepler (only if Mstar exists)
    aRs = np.nan
    method_aRs = "missing"

    if not np.isnan(Mstar) and not np.isnan(P_tab) and not np.isnan(R_val):
        aRs_kepler = compute_aRs_from_kepler(P_tab, Mstar, R_val)
        if 1.0 < aRs_kepler < 1000:   
            aRs = aRs_kepler
            method_aRs = "kepler"

    # 3) b = 0, a/R* from TPk
    if not np.isnan(k) and not np.isnan(T_tab) and not np.isnan(P_tab):
        aRs_TPk = compute_aRs_from_TPk_b0(T_tab, P_tab, k)
        if 1.0 < aRs_TPk < 1000:
            aRs = aRs_TPk
            method_aRs = "TPk_b0"

    # 4) If Mstar was missing, derive it from a/R*, R_star, P
    if np.isnan(Mstar):
        if all(not np.isnan(x) for x in [aRs, R_val, P_tab]):
            Mstar_est = compute_Mstar_from_aRs(aRs, R_val, P_tab)
            if 0.1 < Mstar_est < 10:  
                Mstar = Mstar_est
                method_Mstar = "kepler"
    if np.isnan(Mstar):
        method_Mstar = "missing"
    
    # 5) impact parameter b from T, P, k, a/R*
    b = 0
    method_b = "ASSUMED_0"

    if all(not np.isnan(x) for x in [T_tab, P_tab, k, aRs]):
        b_calc = compute_b_from_TPkaRs(T_tab, P_tab, k, aRs)
        if 0.0 <= b_calc <= 1.0 + k:
            b = b_calc
            method_b = "TPkaRs"       
    # 6) orbital period P from T, k, a/R*, b
    P_val = to_float(P_tab, "pl_orbper")
    if P_val > 150:
        print("Caution: long period from toi")
    method_P = "given"

    if np.isnan(P_val):
        if all(not np.isnan(x) for x in [T_tab, k, aRs, b]):
            P_est = compute_P_from_Tk_aRs_b(T_tab, k, aRs, b)
            if not np.isnan(P_est):
                P_val = P_est
                method_P = "from_Tk_aRs_b"
        else:
            method_P = "missing" 


    return {
        "pl_orbper": P_tab,
        "pl_trandurh": T_tab,
        "pl_ratror": k,
        "pl_ratdor": aRs,
        "st_mass": Mstar,
        "st_rad": R_val,
        "pl_imppar": b,
        "method_imppar": method_b,
        "method_orbper": method_P,
        "method_ratror": method_aRs,
        "method_stmass": method_Mstar
    }



# -------------------QUERY----------------------

# query all requisite parameters from pscomppars
def query_params_from_pscomppars(ticid):
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    qry = (
        "SELECT pl_name, hostname, tic_id, pl_orbper, pl_tranmid, pl_ratror, pl_ratdor, "
        "pl_imppar, pl_rade, st_rad, st_mass "
        "FROM pscomppars "
        f"WHERE tic_id LIKE '%{ticid}%'"
    )

    params = {
        "query": qry,
        "format": "csv"
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        return df
    except Exception as e:
        print(f"Error querying pscomppars for TIC ID {ticid}: {e}")
        return None

# query all parameters related and available in toi 
def query_params_from_toi(ticid):
    # Query the TESS Object of Interest (TOI) table for a given TIC ID  
    """
        Parameters
        ----------
        ticid : str
            TIC ID of the target star.
        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the queried TOI parameters.
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    qry = (
        "SELECT toi, tid, ctoi_alias, pl_pnum, tfopwg_disp, "
        "pl_tranmid, pl_orbper, pl_trandurh, pl_trandep, "
        "st_tmag, st_rad, st_logg "
        #"SELECT TOP 5 * "
        "FROM toi "
        f"WHERE tid = {ticid}" 
    )
    params = {
        "query": qry,
        "format": "csv"
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        return df
    except Exception as e:
        print(f"Error querying TOI for TIC ID {ticid}: {e}")
        return None

# query latest time parameters (TC , P) in ps table 
def query_latest_time_from_ps(ticid):
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    # order the result with rowupdate
    qry = f"""
    SELECT 
        pl_name, hostname, tic_id, pl_orbper, pl_tranmid,
        rowupdate, pl_pubdate, releasedate
    FROM ps
    WHERE tic_id LIKE '%{ticid}%'
    ORDER BY  pl_pubdate DESC, rowupdate DESC, releasedate DESC
    """  

    try:
        r = requests.get(url, params={"query": qry, "format": "csv"}, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))

        if df.empty:
            print(f"No entry in ps table for TIC {ticid}")
            return None
        
    # find the latest set of valid values of time terms
        valid = df.dropna(subset=["pl_orbper","pl_tranmid"])
        if valid.empty:
            print("No tranmid and orbper satisfied")

        return valid.iloc[0]

    except Exception as e:
        print(f"Error when querying ps: {e}")
        return None    


# Combine the pscomppars and ps tables: replace P, tc in pscomppars
def get_best_params(ticid):
    # 1. pscomppars 
    df_comp = query_params_from_pscomppars(ticid)
    if df_comp is None or len(df_comp)==0:
        print("No entry in pscomppars")
        return None
    comp = df_comp.iloc[0].copy() 

    # 2. ps
    result = query_latest_time_from_ps(ticid)
    if result is not None:
        latest_ps = result

        # replace pscomppars time items with ps ones
        if not pd.isna(latest_ps["pl_orbper"]) and comp["pl_orbper"] != latest_ps["pl_orbper"]:
            comp["pl_orbper"] = latest_ps["pl_orbper"]

        if not pd.isna(latest_ps["pl_tranmid"]) and comp["pl_tranmid"] != latest_ps["pl_tranmid"]:
            comp["pl_tranmid"] = latest_ps["pl_tranmid"]

    return comp

    # organise the structure of the output
    #final = comp[[
    #   "pl_name", "hostname", "tic_id", 
    #    "pl_orbper", "pl_tranmid", "pl_ratror", "pl_ratdor",
    #    "pl_imppar", "pl_rade", "st_rad", "st_mass"
    #   ]].to_frame().T

    #return final

def infer_params_for_df(df):
    results = []
    for _, row in df.iterrows():
        results.append(infer_transit_params(row))
    return pd.DataFrame(results)


def query_params_NEA(ticid):
    """
    Query NEA parameters for a given TIC ID.
    1. Try pscomppars
    2. If empty, fallback to TOI table
    3. Infer missing transit parameters if needed
    Prints formatted output and returns dataframe.
    """

    print("---------------------------------------------------------")
    print(f"Querying NASA Exoplanet Archive for TIC {ticid} ...")

    # Step 1: Try pscomppars first
    df_psc = get_best_params(ticid)
    if df_psc is not None and not df_psc.empty:
        print(f"pscomppars entry FOUND for TIC {ticid}:")
        print(df_psc)
        print("---------------------------------------------------------")
        return df_psc

    print(f"No pscomppars entry found for TIC {ticid}. Trying TOI table...")

    # Step 2: Try TOI
    try:
        df_toi = query_params_from_toi(ticid)

        if df_toi is None or df_toi.empty:
            print(f"No TOI entry found for TIC {ticid}.")
            print("---------------------------------------------------------")
            return None

        # Step 3: Infer parameters

        inferred = infer_params_for_df(df_toi)    # return df
        combined = df_toi.copy()

        for col in inferred.columns:
            if col not in combined.columns:
                combined[col] = inferred[col]
            else:
                combined[col] = combined[col].combine_first(inferred[col])

        print(f"Returning combined TOI + inferred parameters for TIC {ticid}:")
        print(combined)
        print("---------------------------------------------------------")
        return combined

    except Exception as e:
        print(f"Error querying TOI table for TIC {ticid}: {e}")
        print("---------------------------------------------------------")
        return None


#df = query_params_from_toi("211446495")
#query_params_NEA("276754403")
# print(df.columns)