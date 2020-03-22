# Trained models with params
name | epochs | samples | batch-size | mel filters | frame_length | neurons | kernel size | notes
--- | --- | --- | --- | --- | --- | --- | --- | ---
gcloud-20200321_0609-n1001.hdf5 | 100 | 6300 | 64 | 512 | 160 | 64 | 3 -> 4 -> 5 | na
gcloud-20200321_2243-n1001.hdf5 | 100 | 6300 | 64 | 128 | 160 | 128 | 3  | 4 layers on hide and 6 on reveal
gcloud-xxx | 100 | 6300 | 64 | 128 | 160 | 128 -> 64 -> 32 -> 32 | 3, 4, 5 | 1st run with datagenerator and full 6300 secret & cover instead of 3150