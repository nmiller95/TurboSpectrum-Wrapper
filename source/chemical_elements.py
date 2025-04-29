import numpy as np
from sys import exit


def atomic_z(el):
    """
    Finds atomic number (Z) of the chemical element by comparing its name
    to the list stored in ./atomic_numbers.dat

    Parameters
    ----------
    el : str
        element name e.g. 'Mg'
    """
    try:
        el_z = np.loadtxt('./atomic_numbers.dat', usecols=0, dtype=int)
        el_id = np.loadtxt('./atomic_numbers.dat', usecols=1, dtype=str)

        for i in range(len(el_id)):
            if el.lower() == el_id[i].lower():
                return el_z[i]

        print(f"Caution: element {el} not found in atomic_numbers.dat")
        return 0

    except FileNotFoundError:
        print("Can not find './atomic_numbers.dat' file. Stopped.")
        return exit(1)


class ChemElement(object):
    """
    Class for handling individual chemical elements. Gets atomic number
    and checks whether element is Fe or H when initialised

    Parameters
    ----------
    i_d : str
        element name e.g. 'Fe'
    """
    def __init__(self, i_d =''):

        self.ID = i_d.strip().capitalize()
        self.Z = atomic_z(self.ID)
        self.nlte = False
        self.comment = ""

        # TODO: If you find a nicer way to figure this out, please change here and the rest of the code will manage
        if i_d.strip().lower() == 'fe' and self.Z == 26:
            self.isFe = True
        else: self.isFe = False

        if i_d.strip().lower() == 'h' and self.Z == 1:
            self.isH = True
        else: self.isH = False
