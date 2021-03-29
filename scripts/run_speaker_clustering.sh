#!/bin/bash

export LC_ALL=C
source ~/anaconda3/bin/activate ~/anaconda3/envs/inaseg
#python_path=~/anaconda3/bin
stage=2


#DATA_PATH=$PWD'/sample_data/segmented_data'
DATA_PATH=$1
echo $DATA_PATH
num_spk=$2
TOTAL_SPLIT=1
#Split data folder into multiple chunk
#if [ $stage -eq 1 ]; then
#CUDA_VISIBLE_DEVICES=1 $python_path/python ./src/split_data_segments.py --source $DATA_PATH --split $TOTAL_SPLIT

#fi

#echo $stage

#Extract speaker embedding using Tensorflow
if [ $stage -le 2 ]; then
mkdir -p exp/embeddings
for (( split=1; split<=$TOTAL_SPLIT; split++ )); do
  python ./src/extract_embedding_from_model.py \
      --feat_type mfcc \
      --feat_dim 40 \
      --nfft 512 \
      --win_len 400 \
      --hop 160 \
      --vad True \
      --cmvn m \
      --data_folder $DATA_PATH \
      --total_split $TOTAL_SPLIT \
      --current_split $split \
      --save_folder exp/embeddings \
      --model_name spk2vec_aug \
      --softmax_num 1211 \
      --resume_startpoint 6992000 \
      --segments_format True \
      --embeddings_layer softmax/fc2

done

fi

#Clustering and write seg2spk label file in data folder. 
# The last column in seg2spk file represent speaker ID.
if [ $stage -le 3 ]; then

python ./src/spk_clustering.py \
      --max_spks $num_spk \
      --data_folder $DATA_PATH \
      --total_split $TOTAL_SPLIT \
      --embedding_folder exp/embeddings \
      --model_name spk2vec_aug \
      --embedding_layer softmax/fc2

fi
