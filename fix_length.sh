#!/bin/bash

length="02.50"
path_data="data"
path_src="TRAIN"
path_dst="TRAIN_CUT"

samples="$(find "${path_data}/${path_src}" -type f -name '*.WAV.wav' | xargs)"

for src_wav_file in ${samples}; do
    dst_wav_file="${src_wav_file/${path_src}/${path_dst}}"
    mkdir -p "$(dirname "${dst_wav_file}")"

    echo "Adapting ${src_wav_file} to ${dst_wav_file}"
    ffmpeg -y -stream_loop -1 -i "${src_wav_file}" \
        -ss 00:00:00.00 -to 00:00:${length} "${dst_wav_file}" > /dev/null 2>&1
done
