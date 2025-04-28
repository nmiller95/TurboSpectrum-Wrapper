# external
import os
from sys import argv
import shutil
import subprocess
import numpy as np
import pickle
import glob
import time
import datetime
# local
from atmos_package import read_atmos_marcs, model_atmosphere
from configure_setup import setup


def compute_babsma(ts_input, atmos, model_opac_file, quite=True):
    """
    Creates input for the babsma.f routine and executes it
    babsma.f routine computes opacities for the give model atmosphere
    which are then used by the bsyn.f routine

    Parameters
    ----------
    ts_input : dict
        contains TS input flags
        must include the following flags:
            'MARCS-FILE'('.true.' or '.false.'),
            'ts_root' (path to TS executables bsyn.f and babsma.f)
    atmos : model_atmosphere
        for which model atmosphere to compute the opacities
    model_opac_file : str
        where to store computed opacities
    quite : boolean
        controls details printout of the progress info
    """

    babsma_conf = F""" \
'MODELINPUT:'    '{atmos.path}'
'LAMBDA_MIN:'    '{ts_input['LAMBDA_MIN']:.3f}'
'LAMBDA_MAX:'    '{ts_input['LAMBDA_MAX']:.3f}'
'LAMBDA_STEP:'   '{ts_input['LAMBDA_STEP']:.3f}'
'MARCS-FILE:' '{ts_input['MARCS-FILE']}'
'MODELOPAC:' '{model_opac_file}'
'METALLICITY:'    '{atmos.feh:.3f}'
'HELIUM     :'    '0.00'
'R-PROCESS  :'    '0.00'
'S-PROCESS  :'    '0.00'
    """

    time0 = time.time()
    cwd = os.getcwd()
    os.chdir(ts_input['ts_root'])
    pr = subprocess.Popen(['./exec-gf/babsma_lu'], stdin=subprocess.PIPE,
                          stdout=open(cwd + '/babsma.log', 'w'), stderr=subprocess.STDOUT )
    pr.stdin.write(bytes(babsma_conf, 'utf-8'))
    pr.communicate()
    pr.wait()
    os.chdir(cwd)
    if not quite:
        print(F"babsma: {time.time()-time0} seconds")


def compute_bsyn(ts_input, elemental_abundances, atmos, model_opac_file, spec_result_file, nlte_info_file=None, quite = True):
    """
    Creates input for the bsyn.f routine and executes it
    bsyn.f runs spectral synthesis based on the opacities
    computed previsously by babsma.f

    Parameters
    ----------
    ts_input : dict
        contains TS input flags
        must include the following flags:
            'NLTE' ('.true.' or '.false.'),
            'LAMBDA_MIN', 'LAMBDA_MAX', 'LAMBDA_STEP',
            'MARCS-FILE'('.true.' or '.false.'),
            'NFILES' (how many linelists provided, integer),
            'LINELIST' (separated with new line),
            'ts_root' (path to TS executables bsyn.f and babsma.f)
    elemental_abundances : list
        contains atomic numbers and abundances of the elements requested for spectral synthesis,
        e.g.
        [ [26, 7.5], [8, 8.76] ]
    atmos : model_atmosphere
        for which model atmosphere to compute the opacities
    model_opac_file : str
        path to the file storing opacities computed by babsma
        returned by compute_babsma
    spec_result_file : str
        where to save computed spectrum
    nlte_info_file : str
        path to the configuration file that controls inclusion of NLTE to TS
        returned by create_NlteInfoFile
        if None, spectrum will be computed in LTE
    quite : boolean
        controls details printout of the progress info
    """

    bsyn_config = F""" \
'NLTE :'          '{ts_input['NLTE']}'
'LAMBDA_MIN:'    '{ts_input['LAMBDA_MIN']:.3f}'
'LAMBDA_MAX:'    '{ts_input['LAMBDA_MAX']:.3f}'
'LAMBDA_STEP:'   '{ts_input['LAMBDA_STEP']:.3f}'
'INTENSITY/FLUX:' 'Flux'
'MARCS-FILE:' '{ts_input['MARCS-FILE']}'
'MODELOPAC:'        '{model_opac_file}'
'RESULTFILE :'    '{spec_result_file}'
'HELIUM     :'    '0.00'
'NFILES   :' '{ts_input['NFILES']}'
{ts_input['LINELIST']}
"""
    if atmos.spherical:
        bsyn_config = bsyn_config + f" 'SPHERICAL:'  '.true.' \n"
    else:
        bsyn_config = bsyn_config + f" 'SPHERICAL:'  '.false.' \n"

    bsyn_config = bsyn_config + f"""\
30
300.00
15
1.30
"""

    if not isinstance(nlte_info_file, type(None)):
        bsyn_config = bsyn_config + f"'NLTEINFOFILE:' '{nlte_info_file}' \n"

    bsyn_config = bsyn_config +\
            f"'INDIVIDUAL ABUNDANCES:'   '{len(elemental_abundances)}' \n"
    for i in range(len(elemental_abundances)):
        z, abund = elemental_abundances[i]
        bsyn_config = bsyn_config + f" {z:.0f} {abund:5.3f} \n"

    """ Run bsyn """
    time0 = time.time()
    cwd = os.getcwd()
    os.chdir(ts_input['ts_root'])
    pr = subprocess.Popen(['./exec-gf/bsyn_lu'], stdin=subprocess.PIPE,
                          stdout=open(cwd + '/bsyn.log', 'w'), stderr=subprocess.STDOUT )
    pr.stdin.write(bytes(bsyn_config, 'utf-8'))
    pr.communicate()
    pr.wait()
    os.chdir(cwd)
    if not quite:
        print(F"bsyn: {time.time()-time0} seconds")


def create_nlte_info_file(elemental_config, model_atoms_path='', departure_files_path='', file_path='./nlteinfofile.txt'):
    """
    Creates configuration file that controls inclusion of NLTE
    for requsted elements into spectral synthesis

    Parameters
    ----------
    elemental_config : list
        contains IDs, atomic number, abundances, NLTE flag,
        and departure coefficient file + model atom ID if NLTE is True,
        for the elements requested for spectral synthesis
        e.g.
        [
            ['Fe', 26, 7.5, True, './depart_Fe.dat', 'atom.fe607c'],
            ['O', 8, 8.76, False, '', '']
        ]
    model_atoms_path : str
        path to the model atoms, since TS requires all the model atoms
        to be provided in the same directory
        can be symbolic links
    departure_files_path : str
        path to directory containing departure files in TS format
        can be set to empty string, then paths to individual departure files
        have to be absolute
    file_path : str
        where to write the file
    """
    with open(file_path, 'w') as nlte_info_file:
        nlte_info_file.write('# created on \n')
        nlte_info_file.write('# path for model atom files ! this comment line has to be here !\n')
        nlte_info_file.write(F"{model_atoms_path} \n")

        nlte_info_file.write('# path for departure files ! this comment line has to be here !\n')
        nlte_info_file.write(F"{departure_files_path} \n")
        nlte_info_file.write('# atomic (non)LTE setup \n')
        for i in range(len(elemental_config)):
            i_d, z, abund, nlte, depart_file, model_atom = elemental_config[i]
            if nlte:
                model_atom_id = model_atom
                nlte_info_file.write(F"{z}  '{i_d}'  'nlte' '{model_atom}'  '{depart_file}' 'ascii' \n")
            else:
                nlte_info_file.write(F"{z}  '{i_d}'  'lte' ''  '' 'ascii' \n")


def parallel_worker(arg):
    """
    Responsible for organising computations and talking to TS
    Creates model atmosphers, opacity file (by running babsma.f),
    NLTE control file if NLTE is requested fot at least one element,
    computes the spectrum ( by calling bsyn.f),
    and finally cleans up by removing temporary files

    Parameters
    ----------
    arg: tuple
        setup_config:
            requested setup configuration
        ind : list or np.array of int
            positional indexes of stellar labels and individual abundances
            compuations will be done consequently for each index
    """
    setup_config, ind = arg
    temp_dir = f"{setup_config.cwd}/output/job_{setup_config.jobID}_{min(ind)}_{max(ind)}/"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    today = datetime.date.today().strftime("%b-%d-%Y")

    elements = setup_config.inputParams['elements'].values()

    for i in ind:
        # TODO: move writing intrpolated model somewhere else, maybe even right after intrpolation
        atmos = model_atmosphere()
        if not isinstance(setup_config.inputParams['modelAtmInterpol'][i], type(None)):
            """ Compute the spectrum """
            spec_result_file = f"{temp_dir}/spec_{i:.0f}_{['NLTE' if setup_config.nlte else 'LTE'][0]}"
            if not os.path.isfile(spec_result_file):

                atmos.depth_scale, atmos.temp, atmos.ne, atmos.vturb = \
                    setup_config.inputParams['modelAtmInterpol'][i]
                setup_config.inputParams['modelAtmInterpol'][i] = None
                atmos.temp, atmos.ne = 10 ** atmos.temp, 10 ** atmos.ne
                atmos.depth_scale_type = 'TAU500'
                atmos.feh, atmos.logg = setup_config.inputParams['feh'][i], setup_config.inputParams['logg'][i]
                atmos.spherical = False
                atmos.id = f"interpol_{i:05d}_{setup_config.jobID}"
                atmos.path = f"{temp_dir}/atmos.{atmos.id}"
    
                atmos.write(atmos.path, format = 'ts')
    
                """ Compute model atmosphere opacity with babsma.f"""
                model_opac_file = F"{setup_config.ts_input['ts_root']}/opac_{atmos.id}_{setup_config.jobID}"
                compute_babsma(setup_config.ts_input, atmos, model_opac_file, True)
                if not os.path.isfile(model_opac_file):
                    print("It seems that babsma.f failed and did not produce a spectrum, check ./babsma.log")


                header = (f"computed with TS NLTE v.20 \nby E.Magg (emagg at mpia dot de) \n"
                          f"Date: {today} \nInput parameters: \n ")
                header += '\n'.join( f"{k} = {setup_config.inputParams[k][i]}" for  k in setup_config.freeInputParams)
                header += '\n'
                header += '\n'.join(f"A({el.ID}) = {el.abund[i]} {['NLTE' if el.nlte else 'LTE']}" for el in elements)
                header += '\n'
                header += '\n'
    
                "Create NLTE info file"
                if setup_config.nlte:
                    elemental_config = []
                    nlte_info_file   = f"{temp_dir}/NLTEinfoFile_{setup_config.jobID}.txt"
                    for el in elements:
                        if el.nlte:
                            if not isinstance(el.departFiles[i], type(None)):
                                cnfg = [
                                        el.ID, el.Z, el.abund[i],
                                        el.nlte, el.departFiles[i],
                                        el.modelAtom.split('/')[-1]
                                        ]
                            else:
                                cnfg = [ el.ID, el.Z, el.abund[i], False, '', '']
                                setup_config.inputParams['comments'][i] += (f"failed to create departure file for "
                                                                            f"{el.ID} at A({el.ID}) = {el.abund[i]}. "
                                                                            f"Treated in LTE instead.")
                        else:
                            cnfg = [ el.ID, el.Z, el.abund[i], False, '', '']
                        elemental_config.append( cnfg )
    
                    create_nlte_info_file(elemental_config, setup_config.modelAtomsPath, '', nlte_info_file)
                else:
                    nlte_info_file =  None
    
                "Run bsyn.f for spectral synthesis"
                elemental_config = [ [el.Z, el.abund[i]] for el in setup_config.inputParams['elements'].values() ]
                compute_bsyn(
                            setup_config.ts_input, elemental_config,
                            atmos, model_opac_file, spec_result_file,
                            nlte_info_file, setup_config.debug
                )
    
                """ Add header, comments and save the spectrum to the common output directory """
                if os.path.isfile(spec_result_file) and os.path.getsize(spec_result_file) > 0:
                    with open(f"{setup_config.spectraDir}/{spec_result_file.split('/')[-1]}", 'w') as moveSpec:
                        for l in header.split('\n'):
                            moveSpec.write('#' + l + '\n')
                        for l in setup_config.inputParams['comments'][i].split('\n'):
                            moveSpec.write('#' + l + '\n')
                        moveSpec.write('#\n')
                        for l in open(spec_result_file, 'r').readlines():
                            moveSpec.write(l)
                    os.remove(spec_result_file)
                else:
                    print("It seems that bsyn.f failed and did not produce a spectrum, check ./bsyn.log")
              
    
                """ Clean up """
                #os.remove(atmos.path)
                os.remove(model_opac_file)
                os.remove(model_opac_file+'.mod')


if __name__ == '__main__':
    if len(argv) > 1:
        conf_file = argv[1]
    else:
        print("Usage: ./run_ts.py ./configFile.txt")
        exit()
    setup_file = setup(file = conf_file)
    parallel_worker((setup_file, np.arange(len(setup_file))))
    exit(0)
