import os

import numpy as np
import tensorflow as tf
import torch
from absl import app
from absl import flags
from absl import logging
import contextlib

import utils  # local file import

# Data load / output flags.
# TODO: update with uncertainty wrappers
flags.DEFINE_string(
    'results_dir', None,  # eg. 'gs://drd-final-results/all-ensembles/'
    'The directory where model outputs (e.g., predictions, uncertainty '
    'estimates, ground truth values, retention curves are stored).'
    'We expect that subdirectories in this dir will be named with the format '
    '{model_type}_k{k}_{tuning_domain}_mc{n_samples} where `model_type` is '
    'the method used, `k` is the size of the ensemble, `tuning_domain` '
    'specifies if tuning was done on ID or ID+OOD metrics, and `n_samples` is '
    'the number of MC samples. Each of those directories should follow the '
    'format output by `eval_model_backup.py`.')
flags.mark_flag_as_required('results_dir')
flags.DEFINE_string(
    'output_dir',
    '/tmp/diabetic_retinopathy_detection/plots',
    'The directory where the plots are stored.')

# OOD Dataset flags.
flags.DEFINE_string(
  'distribution_shift', None,
  ("Specifies distribution shift to use, if any."
   "aptos: loads APTOS (India) OOD validation and test datasets. "
   "  Kaggle/EyePACS in-domain datasets are unchanged."
   "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
   "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
   "  as OOD."))
flags.mark_flag_as_required('distribution_shift')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info(
    'Saving robustness and uncertainty plots to %s', FLAGS.output_dir)

  distribution_shift = FLAGS.distribution_shift
  num_bins = FLAGS.num_bins
  results_dir = FLAGS.results_dir

  logging.info(f'Plotting for distribution shift {distribution_shift}.')
  from collections import defaultdict

  # Contains a defaultdict for each dataset
  # Each dataset has a map from (model_type, k, tuning_domain, num_mc_samples)
  # to a final dict.
  # This dict has the below keys. The values are lists of np.arrays, one
  # array for each random seed.

  dataset_to_model_results = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list)))

  model_dirs = tf.io.gfile.listdir(results_dir)
  for model_dir in model_dirs:
    try:
      model_type, ensemble_str, tuning_domain, mc_str = model_dir.split('_')
    except:
      raise ValueError('Expected model directory in format '
                       '{model_type}_k{k}_{tuning_domain}_mc{n_samples}')

    k = int(ensemble_str[1:])  # format f'k{k}'
    # Tuning domain is either `indomain`, `joint` in our implementation.
    num_mc_samples = mc_str[2:][:-1]  # format f'mc{num_mc_samples}/'
    is_deterministic = model_type == 'deterministic' and k == 1
    print(model_type, ensemble_str, tuning_domain, mc_str)

    model_dir_path = os.path.join(results_dir, model_dir)
    dataset_subdirs = [
      file_or_dir for file_or_dir in tf.io.gfile.listdir(model_dir_path)
      if tf.io.gfile.isdir(os.path.join(model_dir_path, file_or_dir))]
    for dataset_subdir in dataset_subdirs:
      dataset_name = dataset_subdir[:-1]
      print(dataset_name)
      dataset_subdir_path = os.path.join(model_dir_path, dataset_subdir)
      random_seed_dirs = tf.io.gfile.listdir(dataset_subdir_path)
      seeds = [int(random_seed_dir.split('_')[-1].split('/')[0])
               for random_seed_dir in random_seed_dirs]
      seeds = sorted(seeds)
      for seed in seeds:
        key = (model_type, k, is_deterministic, tuning_domain, num_mc_samples)
        eval_results = utils.load_eval_results(
          eval_results_dir=dataset_subdir_path, epoch=seed)

        for arr_name, arr in eval_results.items():
          if arr.ndim > 0 and arr.shape[0] > 1:
            dataset_to_model_results[dataset_name][key][arr_name].append(arr)

  # use this
  # dataset_to_model_results
  # utils.plot_retention_curves(
  #   distribution_shift_name=distribution_shift,
  #   dataset_to_model_results=dataset_to_model_results, plot_dir='.')

  utils.plot_roc_curves(
    distribution_shift_name=distribution_shift,
    dataset_to_model_results=dataset_to_model_results, plot_dir='roc-plots')


if __name__ == '__main__':
  app.run(main)
