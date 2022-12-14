import argparse
import os
import sys
import numpy as np
import pandas as pd
import logging
import subprocess
from Bio import SeqIO
from Bio.PDB import Polypeptide
from timeit import default_timer as timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tqdm import tqdm

from seq_process import *
from create_pdb import *


def seq_iterator(fasta_file_path):
    """
    iterates over a fasta file
    :param fasta_file_path: path to fasta file
    :return:yields sequence, name
    """
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        seq = str(seq_record.seq)
        name = str(seq_record.name)
        yield seq, name


def make_alignment_file(pdb_name, sequence):
    """
    makes alignment file for modeller
    """
    with open("temp_alignment.ali", "w") as ali_file:
        ali_file.write(">P1;{}\n".format(pdb_name))
        ali_file.write("sequence:{}:::::::0.00: 0.00\n".format(pdb_name))
        ali_file.write("{}*\n".format(sequence))

    pdb_file = "{}_mhcfold_backbone_cb".format(pdb_name)

    env = environ()
    aln = alignment(env)
    mdl = model(env, file=pdb_file)
    aln.append_model(mdl, align_codes=pdb_file, atom_files=pdb_file)
    aln.append(file="temp_alignment.ali", align_codes=pdb_name)
    aln.align2d()
    aln.write(file="alignment_for_modeller.ali", alignment_format='PIR')


def relax_pdb(pdb_name, sequence):
    """
    reconstruct side chains using modeller
    """
    log.none()
    log.level(output=0, notes=0, warnings=0, errors=0, memory=0)
    make_alignment_file(pdb_name, sequence.replace(":", "/"))

    pdb_file = "{}_mhcfold_backbone_cb".format(pdb_name)

    # log.verbose()
    env = environ()

    # directories for input atom files
    env.io.atom_files_directory = ['.', '../atom_files']

    a = MyLoopModel(env, alnfile='alignment_for_modeller.ali', knowns=pdb_file, sequence=pdb_name)
    a.starting_model = 1
    a.ending_model = 1
    a.make()

#     # clean temp files
#     for file in os.listdir(os.getcwd()):
#         if file[-3:] in ['001', 'rsr', 'csh', 'ini', 'ali', 'sch']:
#             os.remove(file)
    os.rename("{}.B99990001.pdb".format(pdb_name), "{}_mhcfold_full_relaxed.pdb".format(pdb_name))


def run_mhcnet(fasta_path, structure_model_path, classification_model_path, task, output_dir, modeller, scwrl):
    """
    runs NanoNet structure predictions
    """
    # make input for NanoNet
    sequences = []
    s = []
    names = []
    #     i = 0
    for sequence, name in seq_iterator(fasta_path):
        sequences.append(sequence)
        s.append(sequence)
        names.append(name)
    #         i += 1
    input_matrix = np.zeros((len(sequences), MHC_MAX_LENGTH, FEATURE_NUM))
    for i in range(len(input_matrix)):
        a, b, p = sequences[i].split(':')
        seq = pad_seq2(a, b, p)
        sequences[i] = seq
        one_hot = generate_input2(seq)
        input_matrix[i] = one_hot

    # load model
    logging.getLogger('tensorflow').disabled = True
    structure_module = tf.keras.models.load_model(structure_model_path, compile=False)
    classification_module = tf.keras.models.load_model(classification_model_path, compile=False)

    # predict MHC ca coordinates
    input_matrix = np.array(input_matrix)

    # change to output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    if task != 1:
      binary_classification =  classification_module.predict(input_matrix).flatten()
      classification_df = pd.DataFrame({"name": names, "score": binary_classification})
      classification_df.to_csv("classification_results")

    if task == 1 or task == 3:
      backbone_coords = structure_module.predict(input_matrix)
      for coords, sequence, name, modeller_seq in (zip(backbone_coords, sequences, names, s)):
          backbone_file_path = "{}_mhcfold_backbone_cb.pdb".format(name)
          with open(backbone_file_path, "w") as file:
              matrix_to_pdb_bone2(file, sequence, coords)
          if modeller:
              relax_pdb(name, modeller_seq)
          if scwrl:
              subprocess.run("{} -i {} -o {}_mhcfold_full.pdb".format(scwrl, backbone_file_path, name), shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("fasta", help="fasta file with Nbs sequences")
    parser.add_argument("-t", "--task", help="prediction type: structure-1, binary classification-2, both-3. default=3",
                        type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("-o", "--output_dir",
                        help="Directory to put the predicted PDB models, (default: ./MHCNetResults)", type=str)
    parser.add_argument("-m", "--modeller", help="Side chains reconstruction using modeller (default: False)",
                        action="store_true")
    parser.add_argument("-c", "--scwrl", help="Side chains reconstruction using scwrl, path to Scwrl4 executable",
                        type=str)
    args = parser.parse_args()

    # check arguments
    MHCfold_dir_path = os.path.abspath(os.path.dirname(sys.argv[0]))
    structure_model = os.path.join(MHCfold_dir_path, 'lean_model_for_classifier_new_padding2')
    classification_model = os.path.join(MHCfold_dir_path,'v7_date_5_9_2022')

    scwrl_path = os.path.abspath(args.scwrl) if args.scwrl else None
    output_directory = args.output_dir if args.output_dir else os.path.join(".", "MHCNetResults")

    if args.modeller:
        from modeller import *
        from modeller.automodel import *
        class MyLoopModel(automodel):
            def special_patches(self, aln):
                # Rename both chains and renumber the residues in each
                self.rename_segments(segment_ids=['A', 'B', 'C'], renumber_residues=[1, 1, 1])
                        
    if not os.path.exists(args.fasta):
        print("Can't find fasta file '{}', aborting.".format(args.fasta), file=sys.stderr)
        exit(1)
    if not os.path.exists(structure_model):
        print("Can't find trained MHCfold '{}', aborting.".format(structure_model), file=sys.stderr)
        exit(1)
    if scwrl_path and not os.path.exists(scwrl_path):
        print("Can't find Scwrl4 '{}', aborting.".format(scwrl_path), file=sys.stderr)
        exit(1)

    start = timer()
    run_mhcnet(args.fasta, structure_model, classification_model, args.task, output_directory, args.modeller, scwrl_path)
    end = timer()

    print("MHCNet ended successfully, models are located in directory:'{}', total time : {}.".format(output_directory,
                                                                                                      end - start))
