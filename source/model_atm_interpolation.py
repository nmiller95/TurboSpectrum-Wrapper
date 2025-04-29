# external
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.spatial import Delaunay
import pickle
import warnings
# local
from atmos_package import ModelAtmosphere
from read_nlte import read_full_nlte_grid, find_distance_to_point,  write_departures_for_ts, restore_depart_scaling


def gradient3rd_order(f):
    for i in range(3):
        f = np.gradient(f, edge_order=2)
    return f


def in_hull(p, hull):
    """
    Is triangulation-based interpolation to this point possible?

    Parameters
    ----------
    p : dict or np.array
        point to test, contains coordinate names as keys and their values, e.g.
        p['teff'] = 5150
    hull :

    Returns
    -------
    whether point is inside the hull
    """
    return hull.find_simplex(p) >= 0


def get_all_ma_parameters(models_path, depth_scale_new, fmt='m1d', debug = False):
    """
    Gets a list of all available model atmopsheres and their parameters
    for interpolation later on.
    If no list is available, creates one by scanning through all available
    models in the specified input directory.

    Parameters
    ----------
    models_path : str
        input directory contatining all available model atmospheres
    depth_scale_new : array or np.ndarray
        depth scale (e.g. TAU500nm) to be used uniformly for model
        atmospheres and departure coefficients
        required to ensure homogenious interpolation and can be provided
        in the config file
    fmt : str
        format of model atmosphere, options: 'm1d' for MULTI formatted input,
        'marcs' for standard MARCS format
    debug : boolean
        switch detailed print out

    Returns
    -------
    magrid : dict
        dictionary containing grid of model atmospheres including both
        the parameters (like Teff, log(g), etc)
        and structure (density as a function of depth, etc)
    """

    save_file = f"{models_path}/all_models_save.pkl"

    if os.path.isfile(save_file) and os.path.getsize(save_file) > 0:
        if debug:
            print(f"reading pickled grid of model atmospheres from {save_file}")
        with open(save_file, 'rb') as f:
            magrid = pickle.load(f)
        depth_scale_new = magrid['structure'][:, np.where(magrid['structure_keys'][0] == 'tau500')[0][0]]
        if np.shape(depth_scale_new) != np.shape(np.unique(depth_scale_new, axis=1)):
            print(f"depth scale is not uniform in the model atmosphere grid read from {save_file}")
            print(f"try removing file {save_file} and run the code again")
            exit()
        else:
            depth_scale_new = np.array(depth_scale_new[0])
    else:
        print(f"Checking all model atmospheres under {models_path}")

        magrid = {'teff':[], 'logg':[], 'feh':[], 'vturb':[], 'file':[], 'structure':[], 'structure_keys':[], 'mass':[]}

        with os.scandir(models_path) as all_files:
            for entry in all_files:
                if not entry.name.startswith('.') and entry.is_file():
                    # try:
                    file_path = models_path + entry.name
                    ma = ModelAtmosphere()

                    ma.read(file_path, format=fmt)

                    if ma.mass <= 1.0:

                        magrid['teff'].append(ma.teff)
                        magrid['logg'].append(ma.logg)
                        magrid['feh'].append(ma.feh)
                        magrid['vturb'].append(ma.vturb[0])
                        magrid['mass'].append(ma.mass)

                        magrid['file'].append(entry.name)

                        ma.temp = np.log10(ma.temp)
                        ma.ne = np.log10(ma.ne)

                        # bring all values to the same depth_scale (tau500)
                        for par in ['temp', 'ne', 'vturb']:
                            f_int = interp1d(ma.depth_scale, ma.__dict__[par], fill_value='extrapolate')
                            ma.__dict__[par] = f_int(depth_scale_new)
                        ma.depth_scale = depth_scale_new

                        magrid['structure'].append( np.vstack( (ma.depth_scale, ma.temp, ma.ne, ma.vturb )  ) )
                        magrid['structure_keys'].append( ['tau500', 'temp', 'ne', 'vturb'])

                    # except: # if it's not a model atmosphere file, or format is wrong
                    #         if debug:
                    #             print(f"Cound not read model file {entry.name} for model atmosphere")

        for k in magrid:
            magrid[k] = np.array(magrid[k])

        " Check if any model atmosphere was successfully read "
        if len(magrid['file']) == 0:
            raise Exception(f"no model atmosphere parameters were retrived from files under {models_path}. "
                            f"Try setting debug = 1 in config file. "
                            f"Check that expected format of model atmosphere is set correctly.")
        else:
            print(f"{len(magrid['file'])} model atmospheres in the grid")

        "Print UserWarnings about any NaN in parameters"
        for k in magrid:
            try: # check for NaNs in numeric values:
                if np.isnan(magrid[k]).any():
                    pos = np.where(np.isnan(magrid[k]))
                    for p in pos:
                        message = f"NaN in parameter {k} from model atmosphere {magrid['path'][p]}"
                        warnings.warn(message, UserWarning)
            except TypeError: # ignore other [non-numerical] keys, such as path, name, etc
                pass
        "Dump all in one file (only done once)"
        with open(save_file, 'wb') as f:
            pickle.dump(magrid, f)
    return magrid


def pre_interpolation_tests(data, interpol_coords, value_key, data_label =''):
    """
    Run multiple tests to catch possible exceptions
    that could affect the performance of the underlying
    Qnull math engine during Delaunay triangulation
    Parameters
    ----------
    data : str or np.ndarray or dict
        input directory contatining all available model atmospheres - this seems to be incorrect
    interpol_coords : array or iterable
        depth scale (e.g. TAU500nm) to be used uniformly for model
        atmospheres and departure coefficients
        required to ensure homogenious interpolation and can be provided
        in the config file
    value_key : str
        format of model atmosphere, options: 'm1d' for MULTI formatted input,
        'marcs' for standard MARCS format
    data_label : str or boolean
        switch detailed print out

    Returns
    -------
    boolean
    """

    " Check for degenerate parameters (aka the same for all grid points) "
    for k in interpol_coords:
        if max(data[k]) == min(data[k]):
            print(f"Grid {data_label} is degenerate in parameter {k}")
            print(F"Values: {np.unique(data[k])}")
            return False

    " Check for repetitive points within the requested coordinates "
    test = [ data[k] for k in interpol_coords]
    if len(np.unique(test, axis=1)) != len(test):
        print(f"Grid {data_label} with coordinates {interpol_coords} has repetitive points")
        return False

    "Any coordinates correspond to the same value? e.g. [Fe/H] and A(Fe) "
    for k in interpol_coords:
        for k1 in interpol_coords:
            if k != k1:
                diff = 100 * ( np.abs( data[k] - data[k1]) ) / np.mean(np.abs( data[k] - data[k1]))
                if np.max(diff) < 5:
                    print(f"Grid {data_label} is only {np.max(diff)} % different in parameters {k} and {k1}")
                    return False

    for k in interpol_coords:
        if np.isnan(data[k]).any():
                print(f"Warning: found NaN in coordinate {k} in grid '{data_label}'")
    if np.isnan(data[value_key]).any():
        print(f"Found NaN in {value_key} array of {data_label} grid")
    return True


def nd_interpolate_grid(input_grid, interpol_par, value_key ='structure'):
    """
    Creates the function that interpolates provided grid.
    Coordinates of the grid are normalised and normalisation vector
    is returned for future reference.

    Parameters
    ----------
    input_grid : dict or np.array
        contains data for interpolation and its coordinates
    interpol_par : np.array
        depth scale in the model atmosphere used to solve for NLTE RT
        (e.g. TAU500nm)
    value_key : str
        key of the inputGrid that subset contains data for interpolation,
        e.g. 'departure'

    Returns
    -------
    interp_f : scipy.interpolate.LinearNDInterpolator
        returns interpolated data
    norm_coord : dict
        contains normalisation applied to coordinates of interpolated data
        should be used to normalised the labels provided in the call to interp_f
    """

    points = []
    norm_coord = {}
    for k in interpol_par:
            points.append(input_grid[k] / max(input_grid[k]))
            norm_coord.update({k :  max(input_grid[k])})
    points = np.array(points).T
    values = np.array(input_grid[value_key])
    interp_f = LinearNDInterpolator(points, values)

    #from scipy.spatial import Delaunay
    #print('preparing triangulation...')
    #tri = Delaunay(points)

    return interp_f, norm_coord#, tri


def prep_interpolation_ma(setup):
    """
    Read grid of model atmospheres and NLTE grids of departures
    and prepare interpolating functions
    Store for future use
    """

    " Over which parameters (== coordinates) to interpolate?"
    interpol_coords = ['teff', 'logg', 'feh'] # order should match input file!
    if 'vturb' in setup.inputParams:
        interpol_coords.append('vturb')

    "Model atmosphere grid"
    if setup.debug: print("preparing model atmosphere interpolator...")
    model_atm_grid = get_all_ma_parameters(setup.atmos_path, setup.depthScaleNew,
                                           fmt= setup.atmos_format, debug=setup.debug)
    passed  = pre_interpolation_tests(model_atm_grid, interpol_coords,
                                      value_key='structure', data_label='model atmosphere grid')
    if not passed:
        exit()
    interp_function, normalised_coord = nd_interpolate_grid(model_atm_grid, interpol_coords,
                                                            value_key='structure')
    """
    Create hull object to test whether each of the requested points
    are within the original grid
    Interpolation outside of hull returns NaNs, therefore skip those points
    """
    hull = Delaunay(np.array([ model_atm_grid[k] / normalised_coord[k] for k in interpol_coords ]).T)

    setup.interpolator['modelAtm'] = {'interpFunction' : interp_function,
                                    'normCoord' : normalised_coord,
                                    'hull': hull}
    del model_atm_grid
    return setup, interpol_coords


def interpolate_all_points_ma(setup):
    """
    Python parallelisation libraries can not send more than X Gb of data between processes
    To avoid that, interpolation at each requested point is done before the start of computations
    """
    if setup.debug: print(f"Interpolating to each of {setup.inputParams['count']} requested points...")

    "Model atmosphere grid"
    setup.inputParams.update({'modelAtmInterpol' : np.full(setup.inputParams['count'], None) })

    count_outside_hull = 0
    for i in range(setup.inputParams['count']):
        point = [ setup.inputParams[k][i] / setup.interpolator['modelAtm']['normCoord'][k] \
                for k in setup.interpolator['modelAtm']['normCoord'] ]
        if not in_hull(np.array(point).T, setup.interpolator['modelAtm']['hull']):
            count_outside_hull += 1
        else:
            values =  setup.interpolator['modelAtm']['interpFunction'](point)[0]
            setup.inputParams['modelAtmInterpol'][i] = values
    if count_outside_hull > 0 and setup.debug:
        print(f"{count_outside_hull}/{setup.inputParams['count']}requested points are outside of the model "
              f"atmosphere grid. No computations will be done for those")
    return setup


def prep_interpolation_nlte(setup, el, interpol_coords, rescale = False, depth_scale = None):
    """
    Read grid of departure coefficients
    in nlteData 0th element is tau, 1th--Nth are departures for N levels
    """
    if setup.debug:
        print(f"reading grid {el.nlteGrid}...")

    el.nlteData = read_full_nlte_grid(
                                el.nlteGrid, el.nlteAux,
                                rescale=rescale, depth_scale= depth_scale,
                                safe_memory= setup.safeMemory
                                )
    del el.nlteData['comment'] # to avoid confusion with dict keys

    """ Scaling departure coefficients for the most efficient interpolation """

    el.nlteData['depart'] = np.log10(el.nlteData['depart']+ 1.e-20)
    pos = np.isnan(el.nlteData['depart'])
    if setup.debug and np.sum(pos) > 0:
        print(f"{np.sum(pos)} points become NaN under log10") # none should become NaN
    el.DepartScaling = np.max(np.max(el.nlteData['depart'], axis=1), axis=0)
    el.nlteData['depart'] = el.nlteData['depart'] / el.DepartScaling

    """
    If element is Fe, than [Fe/H] == A(Fe) with an offset,
    so one of the parameters needs to be excluded to avoid degeneracy
    Here we omit [Fe/H] dimension but keep A(Fe)
    """
    if len(np.unique(el.nlteData['feh'])) == len(np.unique(el.nlteData['abund'])):
        # it is probably Fe
        if el.isFe:
            interpol_coords_el = [c for c in interpol_coords if c != 'feh']
            indiv_abund = np.unique(el.nlteData['abund'])
        else:
            print(f"abundance of {el.ID} is coupled to metallicity, but element is not Fe "
                  f"(for Fe A(Fe) == [Fe/H] is acceptable)")
            exit()
    elif len(np.unique(el.nlteData['abund'])) == 1 :
    # it is either H or no iteration ovr abundance was included in computations of NLTE grids
            interpol_coords_el = interpol_coords.copy()
            indiv_abund = np.unique(el.nlteData['abund'])
    else:
        interpol_coords_el = interpol_coords.copy()
        indiv_abund = np.unique(el.nlteData['abund'] - el.nlteData['feh'])

    """
    Here we use Delaunay triangulation to interpolate over
    fund. parameters like Teff, log(g), [Fe/H], etc,
    and direct linear interpolation for abundance,
    since it is regularly spaced by construction.
    This saves a lot of time.
    """
    el.interpolator = {
            'abund' : [], 'interpFunction' : [], 'normCoord' : [], 'tri':[]
    }

    """ Split the NLTE grid into chuncks of the same abundance """
    sub_grids = {'abund':np.zeros(len(indiv_abund)), 'nlteData':np.empty(len(indiv_abund), dtype=dict) }
    for i in range(len(indiv_abund)):
        sub_grids['abund'][i] = indiv_abund[i]
        if el.isFe or el.isH:
            mask = np.where( np.abs(el.nlteData['abund'] - sub_grids['abund'][i]) < 1e-3)[0]
        else:
            mask = np.where( np.abs(el.nlteData['abund'] - el.nlteData['feh'] - sub_grids['abund'][i]) < 1e-3)[0]
        sub_grids['nlteData'][i] = { k: el.nlteData[k][mask] for k in el.nlteData }

    """
    Run pre-interpolation tests and eventually build an interpolating function
    for each sub-grid of constant abundance
    Grid is divided into sub-grids of constant abundance to speed-up building
    Delaunay triangulation, which is very sensitive to regular spacing
    (e.g. in abundance dimension)

    Delete intermediate sub-grids
    """
    for i in range(len(sub_grids['abund'])):
        ab = sub_grids['abund'][i]
        passed = pre_interpolation_tests(sub_grids['nlteData'][i], interpol_coords_el, value_key='depart',
                                         data_label=f"NLTE grid {el.ID}")
        if passed:
            interp_function, normalised_coord  = nd_interpolate_grid(sub_grids['nlteData'][i], interpol_coords_el,
                                                                     value_key='depart')

            el.interpolator['abund'].append(ab)
            el.interpolator['interpFunction'].append(interp_function)
            el.interpolator['normCoord'].append(normalised_coord)
        else:
            print("Failed pre-interpolation tests, see above")
            print(f"NLTE grid: {el.ID}, A({el.ID}) = {ab}")
            exit()
    del sub_grids
    return setup


def interpolate_all_points_nlte(setup, el):
    """
    Interpolate to each requested abundance of element (el)
    Write departure coefficients to a file
    that will be used as input to TS later
    """
    el.departFiles = np.full(setup.inputParams['count'], None)
    for i in range(len(el.abund)):
        depart_file = el.departDir + \
                f"/depCoeff_{el.ID}_{el.abund[i]:.3f}_{i}.dat"
        x, y = [], []
        # TODO: introduce class for nlte grid and set exceptions if grid wasn't rescaled
        tau = setup.depthScaleNew
        for j in range(len(el.interpolator['abund'])):
            point = [ setup.inputParams[k][i] / el.interpolator['normCoord'][j][k] \
                     for k in el.interpolator['normCoord'][j] if k !='abund']
            ab = el.interpolator['abund'][j]
            depart_ab = el.interpolator['interpFunction'][j](point)[0]
            if not np.isnan(depart_ab).all():
                x.append(ab)
                y.append(depart_ab)
        x, y = np.array(x), np.array(y)
        """
        Now interpolate linearly along abundance axis
        If only one point is present (e.g. A(H) is always 12),
        take departure coefficient at that abundance
        """
        if len(x) >= 2:
            if not el.isFe or el.isH:
                ab_scale = el.abund[i] - setup.inputParams['feh'][i]
            else:
                ab_scale = el.abund[i]
            if min(x) < ab_scale < max(x):
                depart = interp1d(x, y, axis=0)(ab_scale)
                depart = restore_depart_scaling(depart, el)
            else:
                depart = np.nan
        elif len(x) == 1 and el.isH:
            print(f'only one point at abundandance={x} found, will accept depart coeff.')
            depart = y[0]
            depart = restore_depart_scaling(depart, el)
        else:
            print(f"Found no departure coefficients at A({el.ID}) = {el.abund[i]}, "
                  f"[Fe/H] = {setup.inputParams['feh'][i]} at i = {i}")
            depart = np.nan

        """
        Check that no non-linearities are present
        """
        non_lin = False
        if not np.isnan(depart).all():
            for ii in range(np.shape(depart)[0]):
                if (gradient3rd_order(depart[ii]) > 0.01).any():
                    depart = np.nan
                    non_lin = True
                    setup.inputParams['comments'][i] += (f"Non-linear behaviour in the interpolated departure "
                                                         f"coefficients of {el.ID} found. Will be using the closest "
                                                         f"data from the grid instead of interpolated values.\n")
                    break
        if not non_lin:
            print(f'no weird behaviour encountered for {el.ID} at abund={ el.abund[i]:.2f}')
        else:
            print(f"non-linearities for {el.ID} at abund={el.abund[i]:.2f}")
        """
        If interpolation failed e.g. if the point is outside of the grid,
        find the closest point in the grid and take a departure coefficient
        for that point
        """
        if np.isnan(depart).all():
            if setup.debug:
                print(f"attempting to find the closest point the in the grid of departure coefficients")
            # TODO: move the four routines below into model_atm_interpolation
            point = {}
            for k in el.interpolator['normCoord'][0]:
                point[k] = setup.inputParams[k][i]
            if 'abund' not in point:
                point['abund'] = el.abund[i]
            pos, comment = find_distance_to_point(point, el.nlteData)
            depart = el.nlteData['depart'][pos]
            depart = restore_depart_scaling(depart, el)
            tau = el.nlteData['depthScale'][pos]

            for k in el.interpolator['normCoord'][0]:
                if ( np.abs(el.nlteData[k][pos] - point[k]) / point[k] ) > 0.5:
                    for k in el.interpolator['normCoord'][0]:
                        setup.inputParams['comments'][i] += (f"{k} = {el.nlteData[k][pos]} (off by "
                                                             f"{point[k] - el.nlteData[k][pos] }) \n")

        write_departures_for_ts(depart_file, tau, depart, el.abund[i])
        el.departFiles[i] = depart_file
        setup.inputParams['comments'][i] += el.comment
    return setup
