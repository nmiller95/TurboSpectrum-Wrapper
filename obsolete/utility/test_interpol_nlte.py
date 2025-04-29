# external
import sys
sys.path.append('/Users/semenova/phd/projects/TurboSpectrum-Wrapper/')
# local
from source.model_atm_interpolation import nd_interpolate_grid, pre_interpolation_tests
from source.read_nlte import *


def test_interpol_nlte(nlte_data, interpol_parameters, i):
    depart_orig = nlte_data['depart'][i]

    try:
        tau = nlte_data['depthScale'][i]
    #     nlte_data['departNew'] = np.full((np.shape(nlte_data['depart'])[0], np.shape(nlte_data['depart'])[1]+1, np.shape(nlte_data['depart'])[2]), np.nan)
    #     for i in range(len(nlte_data['pointer'])):
    #         nlte_data['departNew'][i] = np.vstack([nlte_data['depthScale'][i], nlte_data['depart'][i]])
    #     nlte_data['depart'] = nlte_data['departNew'].copy()
    #     del nlte_data['departNew']
    #     del nlte_data['depthScale']

        if 'comment' in nlte_data:
            if len(nlte_data['comment'].strip()) > 0:
                print(nlte_data['comment'])
            del nlte_data['comment']

        passed = pre_interpolation_tests(nlte_data, interpol_parameters, value_key='depart', data_label=f"")

        if not passed:
            print('Pre-interpolation tests failed.')
            return exit()

        else:
            for k in interpol_parameters:
                interpol_parameters[k] = nlte_data[k]

            mask = np.full(len(nlte_data['pointer']), True)
            mask[i] = False

            nlte_data_copy = {}
            for k in nlte_data:
                nlte_data_copy[k] = nlte_data[k][mask]

            interp_f, params_to_interpolate = nd_interpolate_grid(nlte_data_copy, interpol_parameters,
                                                                  value_key='depart')

            point = np.array([nlte_data[k][i] / params_to_interpolate[k] for k in params_to_interpolate])
            depart_interpol = interp_f(point)[0]  # [1:]

            return tau, depart_orig, depart_interpol

    except KeyError:
        print('no depth scale provided in nlte grid. stopped')
        return exit()


if __name__ == '__main__':
    bin_file = './NLTEgrid_H_MARCS_May-10-2021.bin'
    aux_file = './auxData_H_MARCS_May-10-2021.txt'
    nlte_dat = read_full_nlte_grid(bin_file, aux_file)

    atmos_path = '/Users/semenova/phd/projects/ts-wrapper/input/atmos/MARCS/all/'
    interpol_params = { 'teff':None, 'logg':None, 'feh':None, 'vturb':None}
    j = 100

    test_interpol_nlte(nlte_dat, interpol_params, j)
