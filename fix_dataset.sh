#!/bin/bash

length="02.50"
path_data="data"
path_trans_train="TRAIN:TRAIN_CUT"
path_trans_test="TEST:TEST_CUT"

for subset in "${path_trans_train}" "${path_trans_test}"; do
    src="$(awk -F':' '{print $1}' <<< "${subset}")"
    dst="$(awk -F':' '{print $2}' <<< "${subset}")"
    subset_samples="$(find "${path_data}/${src}" -type f -name '*.WAV.wav' | xargs)"

    for src_wav_file in ${subset_samples}; do
        dst_wav_file="${src_wav_file/${src}/${dst}}"
        mkdir -p "$(dirname "${dst_wav_file}")"

        echo "Adapting ${src_wav_file} to ${dst_wav_file}"
        ffmpeg -y -stream_loop -1 -i "${src_wav_file}" \
            -ss 00:00:00.00 -to 00:00:${length} "${dst_wav_file}" > /dev/null 2>&1
    done
done
