# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for Introspective Active Sampling.

Library of utilies for the Introspecive Active Sampling method. Includes a
function to generate a table mapping example ID to bias label, which can be
used to train the bias output head.
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf


def get_example_id_to_bias_label_table(
    train_splits: tf.data.Dataset, val_splits: tf.data.Dataset,
    combos: List[List[int]], trained_models: List[tf.keras.Model],
    threshold: float, num_splits: int,
    save_dir: Optional[str] = None,
    save_table: Optional[bool] = True) -> tf.lookup.StaticHashTable:
  """Generates a lookup table mapping example ID to bias label.

  Args:
    train_splits: Splits of training dataset.
    val_splits: Splits of validation dataset.
    combos: Lists of indices indicating the in-domain splits used to train
      models.
    trained_models: List of trained models.
    threshold: Float representing the bias value threshold, above which examples
      will receive a bias label of 1 (and 0 if below).
    num_splits: Total number of slices that data was split into.
    save_dir: Directory in which bias table will be saved as CSV.
    save_table: Boolean for whether or not to save table.

  Returns:
    A lookup table mapping example ID to bias label.
  """
  example_ids_all = []
  bias_values_all = []
  bias_labels_all = []
  for split_idx in range(num_splits):
    # For each split of data,
    # 1. Get the models that included this split (as in-domain training data).
    # 2. Get the models that excluded this split (as out-of-distribution data).
    # 3. Calculate the bias value and, using the threshold, bias label.
    id_predictions_all = []
    ood_predictions_all = []
    labels = list(train_splits[split_idx].map(
        lambda feats, label, example_id: label).as_numpy_iterator())
    labels += list(val_splits[split_idx].map(
        lambda feats, label, example_id: label).as_numpy_iterator())
    labels = np.concatenate(labels)
    for combo_idx, combo in enumerate(combos):
      if split_idx in combo:
        model = trained_models[combo_idx]
        id_predictions_train = model.predict(train_splits[split_idx])
        id_predictions_val = model.predict(val_splits[split_idx])
        id_predictions = tf.concat(
            [id_predictions_train['main'], id_predictions_val['main']], axis=0)
        id_predictions = tf.gather_nd(
            id_predictions, tf.expand_dims(labels, axis=1), batch_dims=1)
        id_predictions_all.append(id_predictions)
      else:
        model = trained_models[combo_idx]
        ood_predictions_train = model.predict(train_splits[split_idx])
        ood_predictions_val = model.predict(val_splits[split_idx])
        ood_predictions = tf.concat(
            [ood_predictions_train['main'], ood_predictions_val['main']],
            axis=0)
        ood_predictions = tf.gather_nd(
            ood_predictions, tf.expand_dims(labels, axis=1), batch_dims=1)
        ood_predictions_all.append(ood_predictions)

    example_ids = list(train_splits[split_idx].map(
        lambda feats, label, example_id: example_id).as_numpy_iterator())
    example_ids += list(val_splits[split_idx].map(
        lambda feats, label, example_id: example_id).as_numpy_iterator())
    example_ids = np.concatenate(example_ids)
    example_ids_all.append(example_ids)
    id_predictions_avg = np.average(np.stack(id_predictions_all), axis=0)
    ood_predictions_avg = np.average(np.stack(ood_predictions_all), axis=0)
    bias_values = np.absolute(
        np.subtract(id_predictions_avg, ood_predictions_avg))
    bias_labels = tf.math.greater(bias_values, threshold)
    bias_values_all.append(bias_values)
    bias_labels_all.append(bias_labels)

  example_ids_all = np.concatenate(example_ids_all)
  bias_values_all = np.squeeze(np.concatenate(bias_values_all))
  bias_labels_all = np.squeeze(np.concatenate(bias_labels_all))
  print(f'# of examples: {example_ids_all.shape}')
  print(f'# of bias labels: {bias_labels_all.shape}')
  print(f'# of non-zero bias labels: {tf.math.count_nonzero(bias_labels_all)}')

  if save_table:
    df = pd.DataFrame({
        'example_id': example_ids_all,
        'bias': bias_values_all,
        'bias_label': bias_labels_all
    })
    df.to_csv(
        os.path.join(save_dir, 'bias_table.csv'),
        index=False)

  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(example_ids_all, dtype=tf.string),
      values=tf.convert_to_tensor(bias_labels_all, dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)
