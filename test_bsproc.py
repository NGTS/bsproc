import requests
import pandas as pd
import io

#print(requests.get(schema_url).text)

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?format=csv&query="

def query_params_from_tic(ticid):
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    # Construct the query, be careful with 1. space after each sentence 2. use 'f' to pass tic id
    # The query must be URL encoded, but requests does it for us when we pass params dict
    # we can choose the conditions we want, here we choose default_flag = 1 to get the latest parameters, or we can choose tran_flag = 1 to get only transiting planets
    qry = (
        "SELECT pl_name, hostname, tic_id, pl_orbper, pl_tranmid, pl_ratror, pl_ratdor, "
        "pl_imppar, pl_rade, st_rad, st_mass "
        "FROM pscomppars "
        f"WHERE tic_id ='TIC {ticid}'"
        #" AND default_flag = 1" for table ps nor for pscomppars
    )

    params = {
        "query": qry,
        "format": "csv"
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    return df

#query_params_from_tic("TIC 276754403")
