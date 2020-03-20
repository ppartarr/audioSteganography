#!/bin/bash

length="02.50"
path_data="data"
train_train_path_src="TRAIN"
train_train_path_dst="TRAIN_CUT"

test_path_src="TEST"
test_path_dst="TEST_CUT"

train_samples="$(find "${path_data}/${train_path_src}" -type f -name '*.WAV.wav' | xargs)"
test_samples="$(find "${path_data}/${train_path_src}" -type f -name '*.WAV.wav' | xargs)"

for src_wav_file in ${train_samples}; do
    dst_wav_file="${src_wav_file/${train_path_src}/${train_path_dst}}"
    mkdir -p "$(dirname "${dst_wav_file}")"

    echo "Adapting ${src_wav_file} to ${dst_wav_file}"
    ffmpeg -y -stream_loop -1 -i "${src_wav_file}" \
        -ss 00:00:00.00 -to 00:00:${length} "${dst_wav_file}" > /dev/null 2>&1
done

for src_wav_file in ${test_samples}; do
    dst_wav_file="${src_wav_file/${test_path_src}/${test_path_dst}}"
    mkdir -p "$(dirname "${dst_wav_file}")"

    echo "Adapting ${src_wav_file} to ${dst_wav_file}"
    ffmpeg -y -stream_loop -1 -i "${src_wav_file}" \
        -ss 00:00:00.00 -to 00:00:${length} "${dst_wav_file}" > /dev/null 2>&1
done
