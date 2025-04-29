import numpy as np
import os
from scipy.interpolate import interp1d


def write_departures_for_ts(file_name, tau, depart, abund):
    """
    Writes NLTE departure coefficients into the file compatible
    with TurboSpectrum

    Parameters
    ----------
    file_name : str
        name of the file in which to write the departure coefficients
    tau : np.array
        depth scale in the model atmosphere used to solve for NLTE RT
        (e.g. TAU500nm)
    depart : np.ndarray
        departure coefficients
    abund : float
        chemical element abundance on log 12 scale
    """

    ndep = len(tau)
    nk = len(depart)
    with open(file_name, 'w') as f:
        """  Comment lines below are requested by TS """
        for i in range(8):
            f.write('# parameter 1.0 1.0\n')

        f.write(f"{abund:.3f}\n")
        f.write(f"{ndep:.0f}\n")
        f.write(f"{nk:.0f}\n")
        for t in tau:
            f.write(F"{t:15.8E}\n")

        for i in range(ndep):
            f.write( f"{'  '.join(str(depart[j,i]) for j in range(nk))} \n" )


def read_departures_for_ts(file_name):
    """
    Reads NLTE departure coefficients from the input file compatible
    with TurboSpectrum

    Parameters
    ----------
    file_name : str
        name of the file in which to write the departure coefficients

    Returns
    -------
    abund : float
        abundance of the chemical element on log 12 scale
    tau : np.array
        depth scale in the model atmosphere used to solve for NLTE RT
        (e.g. TAU500nm)
    depart : np.ndarray
        departure coefficients
    """
    with open(file_name, 'r') as f:
        data = [ l for l in f.readlines() if not l.startswith('#') ]

    abund = float( data[0])
    ndep = int( data[1])
    nk = int( data[2] )

    tau = np.loadtxt(file_name, skiprows=11, max_rows = ndep)
    depart = np.loadtxt(file_name, skiprows=11 + ndep).T
    return abund, tau, depart


def read_binary_grid(grid_file, pointer=1):
    """
    Reads a record at specified position from the binary NLTE grid
    (grid of departure coefficients)

    Parameters
    ----------
    grid_file : str
        path to the binary NLTE grid
    pointer : int
        bitwise start of the record as read from the auxiliarly file

    Returns
    -------
    ndep : int
        number of depth points in the model atmosphere used to solve for NLTE
    nk : int
        number of energy levels in the model atom used to solved for NLTE
    depart : array
        NLTE departure coefficients of shape (ndep, nk)
    tau : array
        depth scale (e.g. TAU500nm) in the model atmosphere
        used to solve for NLTE of shape (ndep)
    """
    with open(grid_file, 'rb') as f:
        # -1 since Python stars at 0
        pointer = pointer - 1

        f.seek(pointer)
        atmos_str = f.readline(500)#.decode('utf-8', 'ignore').strip()
        ndep = int.from_bytes(f.read(4), byteorder='little')
        nk = int.from_bytes(f.read(4), byteorder='little')
        tau  = np.log10(np.fromfile(f, count = ndep, dtype='f8'))
        depart = np.fromfile(f, dtype='f8', count=ndep*nk).reshape(nk, ndep)

    return ndep, nk, depart, tau


def read_full_nlte_grid(bin_file, aux_file, rescale=False, depth_scale=None, safe_memory = np.inf):
    """
    Reads the full binary NLTE grid of departure coefficients
    Note: binary grids can reach up to 100 Gb in size, therefore it is
    possible to only read half of the records
    randomly distributed across the grid

    Parameters
    ----------
    bin_file : str
        path to the binary NLTE grid
    aux_file : str
        path to the complementary auxilarly file
    rescale : boolean
        whether or not to bring the departure coefficients onto a depth
        scale different from the depth scale in the model atmosphere
        used in solving for NLTE. False by default
    depth_scale : np.array
        depth scale to which departure coefficients will be rescaled
        if rescale is set to True
    safe_memory : boolean
        whether or not to save the memory by only reading random
        half of the records in the NLTE grid

    Returns
    -------
    data : dict
        contains NLTE departure coefficients as well as the depth scale of the
        model atmosphere and parameters used in RT
        (e.g. Teff, log(g) of model atmosphere)
    """
    if rescale and isinstance(depth_scale, type(None)):
            raise Warning(f"To re-scale NLTE departure coefficients, please supply new depth scale.")

    aux = np.genfromtxt(aux_file, dtype = [('atmos_id', 'S500'), ('teff','f8'), ('logg','f8'),
                                           ('feh', 'f8'), ('alpha', 'f8'), ('mass', 'f8'),
                                           ('vturb', 'f8'), ('abund', 'f8'), ('pointer', 'i8')])

    data = {}
    grid_size = os.path.getsize(bin_file) / (1024**3)
    if  grid_size > safe_memory:
        print(f"size of the NLTE grid in {bin_file} is {grid_size:.2f} Gb, which is bigger than "
              f"available memory of {safe_memory:.2f}")
        radn_select = np.random.random(size=len(aux['atmos_id']))
        mask_select = np.full(len(aux['atmos_id']), False)
        mask_select[radn_select < (safe_memory / grid_size)] = True
        print(f"Will read { (np.sum(mask_select) / len(aux['atmos_id']) ) *100:.0f} % of the records.")
        for k in aux.dtype.names:
            data.update( { k : aux[k][mask_select] })
    else:
        for k in aux.dtype.names:
            data.update( { k : aux[k] })
    """
    Get size of records from the first records
    assuming same size across the grid (i.e. model atmospheres had the same
    depth dimension and same model atom was used consistently)
    """
    p = data['pointer'][0]
    lev_subst = []
    depth_subst = []
    test = []
    # TODO: read size separately for each record
    ndep, nk, depart, tau = read_binary_grid(bin_file, pointer=p)
    if rescale:
        depart_shape = ( len(data['pointer']), nk, len(depth_scale))
    else:
        depart_shape = ( len(data['pointer']), nk, ndep)
    data.update( {
                'depart' : np.full(depart_shape, np.nan),
                'depthScale' : np.full((depart_shape[0],depart_shape[-1]), np.nan)
                } )
    ## TODO: move replacing nans and inf to preparation for interpolation
    for i in range(len( data['pointer'])):
        p = data['pointer'][i]
        ndep, nk, depart, tau = read_binary_grid(bin_file, pointer=p)
        if np.isnan(depart).any():
            nan_mask = np.where(np.isnan(depart))
            depart[nan_mask] = 1.
            lev_subst.extend(np.unique(nan_mask[1]))
            depth_subst.extend(np.unique(nan_mask[0]))
        if np.isinf(depart).any():
            inf_mask = np.where(np.isinf(depart))
            depart[inf_mask] = 1.
            lev_subst.extend(np.unique(inf_mask[1]))
            depth_subst.extend(np.unique(inf_mask[0]))
        if (depart < 0).any():
            neg_mask = np.where( depart < 0 )
            lev_subst.extend(np.unique(neg_mask[1]))
            depth_subst.extend(np.unique(neg_mask[0]))
            depart[neg_mask] = 1e-20
        if rescale:
            depart = depart + 1e-20
            f_int = interp1d(tau, np.log10(depart), fill_value='extrapolate')
            depart = 10**f_int(depth_scale)
            if np.isnan(depart).any():
                print('NaN at ', p)
            tau = depth_scale
# questinable, but I tested, and they somethimes go from 0.01 to -0.1, so...
# I don't know, Maria doesn't give me time to think or do things properly, I am so tired
        data['depart'][i] = depart
        data['depthScale'][i] = tau

    lev_subst   = np.unique(lev_subst)
    depth_subst = np.unique(depth_subst)
    if len(lev_subst):
        data['comment'] = (f" Found NaN/inf or negative value in the departure coefficients for some of the models "
                           f"at levels {lev_subst} at depth {depth_subst}, changed to 1 (==LTE) \n")
    else: data['comment'] = ""
    return data


def find_distance_to_point(point, grid):
    """

    Find the closest record in the NLTE grid to the supplied point
    based on quadratic distance.
    If several records are at the same distance
    (might happen if e.g. one abudnance was included twice),
    the first one is picked

    Parameters
    ----------
    point : dict
        coordinates of the input point. Only coordinates provided here will
        be used to compute the distance
    grid : dict
        NLTE grid of departure coefficients as read by read_fullNLTE_grid()

    Returns
    -------
    pos : int
        position of the closest point found in the grid
    comment : str
        notifies if more than one point at the minimum distance was found
        and which one was picked
    """
    dist = 0
    for k in point:
        dist += ((grid[k] - point[k])/max(grid[k]))**2
    dist = np.sqrt(dist)
    pos = np.where(dist == min(dist) )[0]
    if len(pos) > 1:
        comment = f"Found more than one 'closets' points to: \n"
        comment += '\n'.join(f"{k} = {point[k]}" for k in point) + '\n'
        comment += f"{grid['atmos_id'][pos]}\n"
        comment += f"Adopted departure coefficients at pointer = {grid['pointer'][pos[0]]}\n"
        return pos[0], comment
    else: return pos[0], ''


def restore_depart_scaling(depart, el):
    """
    Departure coefficients are normalised and brought to the log scale
    for the ease of interpolation.
    This functions brings them back to the initial units

    Parameters
    ----------
    depart : np.ndarray
        normalised departure coefficients
    el : ChemElement
        chemical element corresponding to the departure coeffcicients
        (scaling is the same for all departure coefficients of the same
        chemical element)

    Returns
    -------
    np.ndarray
        Departure coefficient in original units
        as read from the binary NLTE grid
    """
    return 10**(depart * el.DepartScaling)
