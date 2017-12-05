import sys
sys.path.append("../deepfold/") # append path to deepfold library
from deepfold_predict_ddg import predict_ddg, read_ddg_csv, MissingResidueError
from CNN import CNNCubedSphereModel
import scipy.stats
import pickle
import argparse
import os
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument("--markov-model-filename", dest="markov_model_filename",
                    help="Parameter file for Markov model")
parser.add_argument("--ddg-csv-filename", dest="ddg_csv_filename",
                    help="CSV file containing ddG data", default="data/ddgs/guerois.csv")
parser.add_argument("--step", type=int, default=None,
                    help="Which checkpoint file to use (default: %(default)s)")
parser.add_argument("--checkpoint_path", type=str, default='model/', help="Path to specific model checkpoint path")

options = parser.parse_args()


mutations = read_ddg_csv(options.ddg_csv_filename)

for mutation in mutations:
    print(mutation[1])


model = CNNCubedSphereModel(checkpoint_path=options.checkpoint_path, step=options.step)

    values = {
        'pred_wt': [],
        'pred_mutant': [],
        'ddg': [],
        'wt': [],
        'res_id': [],
        'mutant': []
    }

for mutation in mutations:

    try:
        values = predict_ddg(model=model,
                    high_res_features_input_dir = "data/atomistic_features_cubed_sphere_ddg",
                    low_res_features_input_dir = None,
                    pdb_dir = "data/PDB",
                    values = values,
                    pdb_id = mutation[0],
                    mutations = mutation[1],
                    ddg = mutation[2])
    except MissingResidueError as e:
        sys.stderr.write("SKIPPING DUE TO MissingResidueError: "+ e + "\n")
        continue

    except Exception as e:
        print("AN ERROR!")
        continue


path = 'data/'
if not os.path.exists(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path))
with open(os.path.join(path, 'pred_values.p'), 'wb') as f:
    pickle.dump(values, f)
# pcorr, pvalue  = scipy.stats.pearsonr(*ddg_list)
# print("correlation: ", pcorr)
