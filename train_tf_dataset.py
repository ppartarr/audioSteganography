#!/usr/bin/env python3
# coding: utf-8

import tensorflow_datasets as tfds
import apache_beam as beam

from tensorflow_datasets.core.utils import gcs_utils
gcs_utils.gcs_dataset_info_files = lambda *args, **kwargs: None
gcs_utils.is_dataset_on_gcs = lambda *args, **kwargs: False


ds_dir = '/opt/tensorflow_datasets'

(ds_train, ds_test), ds_info = tfds.load(
    'librispeech',
    split=['train', 'test'],
    data_dir=ds_dir,
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    download=True,
    try_gcs=False,
    download_and_prepare_kwargs={
        'download_dir': ds_dir,
        'download_config': tfds.download.DownloadConfig(
            beam_options=beam.options.pipeline_options.PipelineOptions()
        )}
)

print(ds_train)
print(ds_test)
print(ds_info)
