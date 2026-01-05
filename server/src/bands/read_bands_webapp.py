import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# read individual files
def read_file(filename):
    table = np.loadtxt(filename)
    wvl = table[:, 0] * 1e-4
    through = table[:, 1]

    #print(wvl)
    #print(stop)

    return wvl, through ## wavelength [converted to um] , throughput

# read all files for particular telescope. can be called four times to get data for all four telescopes
def read_bands(telescope_name):

    bands_path = os.path.dirname(__file__) + "/"

    if telescope_name == 'HST':
        bandlist_filenames = ['HST_ACS_WFC.F435W.dat', 'HST_ACS_WFC.F606W.dat', 'HST_ACS_WFC.F775W.dat', 'HST_ACS_WFC.F814W.dat', 'HST_ACS_WFC.F850LP.dat']
        #bandlist_cols = ['HST_F435W', 'HST_F606W', 'HST_F775W', 'HST_F814W', 'HST_F850LP']
        df_wvl = []
        df_through = []
        for i, f in enumerate(bandlist_filenames):
            wvl, through = read_file(bands_path + f)

            df_wvl.append(wvl)
            df_through.append(through)

        #df_wvl = np.concatenate(df_wvl, axis=1)[0]
        #df_through = np.concatenate(df_through, axis=1)[0]

        result = pd.DataFrame({'wavelength':np.array(df_wvl, dtype=object), 'throughput':np.array(df_through, dtype=object)})

        #print(result)
        return result

    if telescope_name == 'JWST':
        bandlist_filenames = ['JWST_NIRCam.F090W.dat', 'JWST_NIRCam.F115W.dat', 'JWST_NIRCam.F150W.dat', 'JWST_NIRCam.F200W.dat', 'JWST_NIRCam.F277W.dat', 'JWST_NIRCam.F356W.dat', 'JWST_NIRCam.F444W.dat', 'JWST_MIRI.F1500W.dat']
        #bandlist_cols = ['JWST_F1500W', 'JWST_F090W', 'JWST_F115W', 'JWST_F150W', 'JWST_F200W', 'JWST_F277W', 'JWST_F356W', 'JWST_F444W']
        df_wvl = []
        df_through = []
        for i, f in enumerate(bandlist_filenames):
            wvl, through = read_file(bands_path + f)

            df_wvl.append(wvl)
            df_through.append(through)

        #df_wvl = np.concatenate(df_wvl, axis=1)[0]
        #df_through = np.concatenate(df_through, axis=1)[0]

        result = pd.DataFrame({'wavelength':np.array(df_wvl, dtype=object), 'throughput':np.array(df_through, dtype=object)})

        return result

    if telescope_name == 'Herschel':
        bandlist_filenames = ['Herschel_Pacs_100um.dat', 'Herschel_Pacs_160um.dat', 'Herschel_SPIRE_250um.dat', 'Herschel_SPIRE_350um.dat', 'Herschel_SPIRE_500um.dat']
        #bandlist_cols = ['JWSTHerschel_100um', 'JWSTHerschel_160um', 'JWSTHerschel_250um', 'JWSTHerschel_350um', 'JWSTHerschel_500um']
        df_wvl = []
        df_through = []
        for i, f in enumerate(bandlist_filenames):
            wvl, through = read_file(bands_path + f)

            df_wvl.append(wvl)
            df_through.append(through)

        #df_wvl = np.concatenate(df_wvl, axis=1)[0]
        #df_through = np.concatenate(df_through, axis=1)[0]

        result = pd.DataFrame({'wavelength':np.array(df_wvl, dtype=object), 'throughput':np.array(df_through, dtype=object)})

        #print(result)
        return result

    if telescope_name == 'ALMA':

        df_wvl = []
        df_through = []

        # generate block functions
        # 2.1micron, 10micron, 100micron, 1mm
        # width = 0.1*wvl_centre
        wvl0 = 2.1
        wvl = np.linspace(wvl0*0.95, wvl0*1.05, 100)#*1e4
        weight = 0.5*np.ones(len(wvl))
        weight[0] = 0
        weight[-1] = 0

        df_wvl.append(wvl)
        df_through.append(weight)


        wvl0 = 10
        wvl = np.linspace(wvl0*0.95, wvl0*1.05, 100)#*1e4
        weight = 0.5*np.ones(len(wvl))
        weight[0] = 0
        weight[-1] = 0

        df_wvl.append(wvl)
        df_through.append(weight)

        wvl0 = 100
        wvl = np.linspace(wvl0*0.95, wvl0*1.05, 100)#*1e4
        weight = 0.5*np.ones(len(wvl))
        weight[0] = 0
        weight[-1] = 0

        df_wvl.append(wvl)
        df_through.append(weight)

        wvl0 = 1000
        wvl = np.linspace(wvl0*0.95, wvl0*1.05, 100)#*1e4
        weight = 0.5*np.ones(len(wvl))
        weight[0] = 0
        weight[-1] = 0

        df_wvl.append(wvl)
        df_through.append(weight)
        
        # 870micron~350GHz with width 7.5GHz
        wvl = np.linspace(819, 898, 100)#*1e4
        weight = 0.5*np.ones(len(wvl))
        weight[0] = 0
        weight[-1] = 0

        df_wvl.append(wvl)
        df_through.append(weight)

        result = pd.DataFrame({'wavelength':df_wvl, 'throughput':df_through})

        #print(result)
        return result

# func to combine bands -- main func to use which tells us which filters to grab and then combines them!
def combine_bands(filter_names):

    # number of rows
    nrows = 23 # number of filters we have
    i_hst = [0, 1, 2, 3, 4]
    i_jwst = [5, 6, 7, 8, 9, 10, 11, 12]
    i_herschel = [13, 14, 15, 16, 17]
    i_alma = [18, 19, 20, 21, 22]

    # mapping instrument to indices
    instrument_indices = {
        "HST": i_hst,
        "JWST": i_jwst,
        "Herschel": i_herschel,
        "ALMA": i_alma
    }
    
    # create empty DataFrame with nrows
    df_all = pd.DataFrame(index=range(nrows), columns=["wavelength", "throughput"])
    
    for filtername in filter_names:

        # determine which instrument this filter belongs to
        for instrument, indices in instrument_indices.items():
            if instrument in filtername:
                rows_to_fill = indices
                break
        else:
            # if filter doesn't match any instrument
            continue
        
        # read data
        x = read_bands(filtername)  # should return a DataFrame with wavelength & throughput

        # make sure number of rows matches
        if len(rows_to_fill) != len(x):
            raise ValueError(f"Filter {filtername}: number of rows {len(x)} does not match target rows {len(rows_to_fill)}")

        # assign each row
        for idx, row_idx in enumerate(rows_to_fill):
            df_all.loc[row_idx, "wavelength"] = x.iloc[idx]["wavelength"]
            df_all.loc[row_idx, "throughput"] = x.iloc[idx]["throughput"]

    return df_all

# func to sort by wvl of filters available. keeps NaNs where they are 
def sort_by_first_wavelength(df):
    df_sorted = df.copy()
    

    valid_mask = df['wavelength'].apply(lambda x: isinstance(x, (list, np.ndarray)))
    
    valid_rows = df[valid_mask]
    
    sorted_valid_rows = valid_rows.sort_values(
        by='wavelength', 
        key=lambda col: col.apply(lambda x: x[0])
    )
    
    df_sorted.loc[valid_mask, :] = sorted_valid_rows.values
    
    return df_sorted

# func to shift wvl of filter
def shift_filter_wvl_with_redshift(wvl_obs, redshift):
    wvl_rest = wvl_obs/(1 + redshift)
    
    return wvl_rest

# example plot
"""

def plot(df, redshift=0.0, draw_fig=False):

    bands = sort_by_first_wavelength(df)

    nrows = len(bands)
    
    colors = ["#800080", "#0000FF", "#00FF00", "#FFA500", "#FF0000", "#8B4513"]  # Purple, Blue, Green, Orange, Red, Dark Brown

    if draw_fig:
        cmap = LinearSegmentedColormap.from_list("purple_brown", colors)
        colors = [cmap(i / (nrows - 1)) for i in range(nrows)]
        plt.figure(figsize=(10, 6))
    
    for idx, row in bands.iterrows():
        wavelength = row["wavelength"]
        throughput = row["throughput"]
        
        if isinstance(wavelength, float) and np.isnan(wavelength):
            continue
        if isinstance(throughput, float) and np.isnan(throughput):
            continue
        if not isinstance(wavelength, (list, np.ndarray)):
            continue
        if not isinstance(throughput, (list, np.ndarray)):
            continue
        if len(wavelength) == 0 or len(throughput) == 0:
            continue
        
        wavelength = shift_filter_wvl_with_redshift(np.array(wavelength, dtype=float), redshift)
        throughput = np.array(throughput, dtype=float)

        if draw_fig:
            plt.fill_between(wavelength, throughput, color=colors[idx], alpha=0.4)
    
    if draw_fig:
        plt.xlabel("Wavelength")
        plt.ylabel("Throughput")
        plt.xscale('log')
        plt.xlim(1e-1, 1e3)
        plt.savefig(f'test_redshift_{redshift}.png', dpi=200, bbox_inches='tight')
        #plt.show()
        plt.close()
    
    return colors,wavelength,throughput

"""

def plot(df, redshift=0.0, draw_fig=False):
    bands = sort_by_first_wavelength(df)
    nrows = len(bands)

    # Generate enough colors to cover all bands
    base_colors = ["#800080", "#0000FF", "#00FF00", "#FFA500", "#FF0000", "#8B4513"]

    if nrows > len(base_colors):
        cmap = LinearSegmentedColormap.from_list("purple_brown", base_colors)
        color_list = [cmap(i / (nrows - 1)) for i in range(nrows)]
        # Convert RGBA floats (0–1) to hex strings for JSON
        color_list = [mcolors.to_hex(c) for c in color_list]
    else:
        color_list = base_colors[:nrows]

    datasets = []

    if draw_fig:
        plt.figure(figsize=(10, 6))

    for idx, row in bands.iterrows():
        wavelength = row["wavelength"]
        throughput = row["throughput"]

        if isinstance(wavelength, float) and np.isnan(wavelength):
            continue
        if isinstance(throughput, float) and np.isnan(throughput):
            continue
        if not isinstance(wavelength, (list, np.ndarray)):
            continue
        if not isinstance(throughput, (list, np.ndarray)):
            continue
        if len(wavelength) == 0 or len(throughput) == 0:
            continue

        wavelength = shift_filter_wvl_with_redshift(np.array(wavelength, dtype=float), redshift)
        throughput = np.array(throughput, dtype=float)

        datasets.append({
            "xs": wavelength.tolist(),
            "ys": throughput.tolist(),
            "color": color_list[idx]
        })

        if draw_fig:
            plt.fill_between(wavelength, throughput, color=color_list[idx], alpha=0.4)

    if draw_fig:
        plt.xlabel("Wavelength")
        plt.ylabel("Throughput")
        plt.xscale("log")
        plt.xlim(1e-1, 1e3)
        plt.savefig(f"test_redshift_{redshift}.png", dpi=200, bbox_inches="tight")
        plt.close()

    return datasets
            
if __name__ == "__main__":
    ######## read bands for each #######
    herschel = read_bands('Herschel')
    jwst = read_bands('JWST')
    hst = read_bands('HST')
    alma = read_bands('ALMA')

    ####### combine any that are shown on webapp ########
    bands = combine_bands(['HST', 'JWST', 'Herschel', 'ALMA'])
    print(bands)

    ####### example plot with colour ##########
    plot(bands, redshift=0)
    plot(bands, redshift=1)
    plot(bands, redshift=2)

