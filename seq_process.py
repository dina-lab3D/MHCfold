from Bio.PDB import *
from Bio import pairwise2
import subprocess
import os
import numpy as np
import pandas as pd


MHC_MAX_LENGTH = 415
A_MAX_LENGTH = 290
B_MAX_LENGTH = 110
C_MAX_LENGTH = 15

FEATURE_NUM = 25
AA_DICT = {"A": 0, "C": 1, "D": 2, "E": 3, "F": 4, "G": 5, "H": 6, "I": 7, "K": 8, "L": 9, "M": 10, "N": 11, "P": 12,
           "Q": 13, "R": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19,  "X": 20, "-": 21, "1": 22, "2": 23, "3": 24}


def get_seq(chain, flag=False):
    """
    returns the sequence (String) and a list of all the aa residue objects of the given nanobody chain.
    :param chain: BioPython chain object
    :return: sequence, [aa objects]
    """
    seq = ""
    aa_list = []
    for residue in chain.get_residues():
        aa = residue.get_resname()
        if not is_aa(aa) or not residue.has_id('CA'):
            continue
        elif (aa == "UNK") or (aa == "CDE"):
            seq += "X"
        else:
            try:
                seq += Polypeptide.three_to_one(residue.get_resname())
            except:
                seq += "X"
        aa_list.append(residue)
    if flag:
        return seq, aa_list
    return seq


def pad_seq(seq):
    """
    pads a Nb sequence with "-" to match the length required for NanoNet (NB_MAX_LENGTH)
    :param seq: Nb sequence (String) with len =< 140
    :return: Nb sequence (String) with len == 140 (with insertions)
    """
    seq_len = len(seq)
    end_pad = (MHC_MAX_LENGTH - seq_len + 3)

    # pad the sequence with '-'
    seq = seq + end_pad * "-"
    return seq


def concat_and_pad_seq(pdb_name):
    """
    given a pdb return the seq with separater $ and padding
    :param pdb_name:
    :return:
    """
    model = PDBParser(QUIET=True).get_structure(pdb_name, pdb_name)[0]
    seq = ""
    for id in ['A', 'B', 'C']:
        seq += get_seq(model[id]) +'$'
    # seq = seq[:-1]
    seq = pad_seq(seq)
    return seq


def generate_input(seq):
    """
    receives a Nb sequence and returns its  sequence in a one-hot encoding matrix (each row is an aa in the sequence, and
    each column represents a different aa out of the 20 aa + 2 special columns).
    :param seq: sequence (string)
    :return: numpy array of size (NB_MAX_LENGTH * FEATURE_NUM)
    """
    # if "X" in seq:
        # print("Warning, sequence: {}, has unknown aa".format(seq))
    n = 1
    # turn in to one-hot encoding matrix
    seq_matrix = np.zeros((MHC_MAX_LENGTH, FEATURE_NUM))
    for i in range(MHC_MAX_LENGTH + 3):
        s = seq[i]
        if s not in AA_DICT.keys():
            s = 'X'
        if seq[i] == '$':
            n = n + 1
            continue
        seq_matrix[i - n + 1][AA_DICT[s]] = 1
        if n < 4:
            seq_matrix[i - n + 1][AA_DICT[str(n)]] = 1
    return seq_matrix


def create_label(pdb_file):
    # load model
    model = PDBParser(QUIET=True).get_structure(pdb_file, pdb_file)[0]

    labels_matrix = np.zeros((MHC_MAX_LENGTH, 15))
    past_len = 0
    for id in ['A', 'B', 'C']:
        # get seq and aa residues
        seq, aa_residues = get_seq(model[id], True)

        # get the coordinates
        for i in range(len(seq)):
            for j, atom in enumerate(["N", "CA", "C", "O", "CB"]):
                if aa_residues[i].has_id(atom):
                    labels_matrix[i + past_len][j*3:(j*3)+3] = aa_residues[i][atom].get_coord()
        #todo +1
        past_len += len(seq)

    return labels_matrix


def pad_seq2(a, b, c):
    """
    pads a Nb sequence with "-" to match the length required for NanoNet (NB_MAX_LENGTH)
    :param seq: Nb sequence (String) with len =< 140
    :return: Nb sequence (String) with len == 140 (with insertions)
    """
    seq = ''
    end_a = A_MAX_LENGTH - len(a)
    seq = seq + a +end_a * "-"
    end_b = B_MAX_LENGTH - len(b)
    seq = seq + b + end_b * "-"
    end_c = C_MAX_LENGTH - len(c)
    seq = seq + c + end_c * "-"
    return seq


def concat_and_pad_seq2(pdb_name):
    """
    given a pdb return the seq with separater $ and padding
    :param pdb_name:
    :return:
    """
    model = PDBParser(QUIET=True).get_structure(pdb_name, pdb_name)[0]
    a = get_seq(model['A'])
    b = get_seq(model['B'])
    c = get_seq(model['C'])
    seq = pad_seq2(a, b, c)
    return seq


def generate_input2(seq):
    """
    receives a Nb sequence and returns its  sequence in a one-hot encoding matrix (each row is an aa in the sequence, and
    each column represents a different aa out of the 20 aa + 2 special columns).
    :param seq: sequence (string)
    :return: numpy array of size (NB_MAX_LENGTH * FEATURE_NUM)
    """
    # turn in to one-hot encoding matrix
    seq_matrix = np.zeros((MHC_MAX_LENGTH, FEATURE_NUM))
    for i in range(MHC_MAX_LENGTH):
        s = seq[i]
        if s not in AA_DICT.keys():
            s = 'X'
        seq_matrix[i][AA_DICT[s]] = 1
        if not s == '-':
            if i < 290:
                seq_matrix[i][22] = 1
            elif i < 400:
                seq_matrix[i][23] = 1
            else:
                seq_matrix[i][24] = 1
    return seq_matrix


lens = {"A": A_MAX_LENGTH, "B": B_MAX_LENGTH, "C": C_MAX_LENGTH}


def create_label2(pdb_file):
    model = PDBParser(QUIET=True).get_structure(pdb_file, pdb_file)[0]
    labels_matrix = np.zeros((MHC_MAX_LENGTH, 15))
    past_len = 0
    for id in ['A', 'B', 'C']:
        # get seq and aa residues
        seq, aa_residues = get_seq(model[id], True)

        # get the coordinates
        for i in range(len(seq)):
            for j, atom in enumerate(["N", "CA", "C", "O", "CB"]):
                if aa_residues[i].has_id(atom):
                    labels_matrix[i + past_len][j*3:(j*3)+3] = aa_residues[i][atom].get_coord()
        past_len = past_len + lens[id]

    return labels_matrix


# pdb = "/cs/labs/dina/alon.aronson/project/3D_structure/pdbs/1A1M/1A1M.pdb"
# a = "GSHSMRYFYTAMSRPGRGEPRFIAVGYVDDTQFVRFDSDAASPRTEPRPPWIEQEGPEYWDRNTQIFKTNTQTYRENLRIALRYYNQSEAGSHIIQRMYGCDLGPDGRLLRGHDQSAYDGKDYIALNEDLSSWTAADTAAQITQRKWEAARVAEQLRAYLEGLCVEWLRRYLENGKETLQRADPPKTHVTHHPVSDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDRTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWEPHH"
# b = "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
# c = "TPYDINQML"
#
# seq = pad_seq2(a, b, c)
# print(seq)
# b = generate_input2(seq)
# c = create_label2(pdb)
# aa = 5
