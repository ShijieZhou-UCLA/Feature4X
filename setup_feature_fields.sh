DATA_NAME=car

# Your absolute path to the data, should contain a subfolder 'images' for image sequence or a mp4 video file
DATA_PATH=/usr/project/Feature4X/data/$DATA_NAME # must be abs path

# Preprocess input data
python prepare.py --config ./configs/wild/prepare_davis.yaml --src $DATA_PATH --video # please remove --video if input is image sequence

# Extract features from different models
cd internvideo_chat_feature
python internvideo_extract_feat.py --video_path $DATA_PATH/preprocess
cd ..

cd sam2
python sam2_extract_feat.py --video_path $DATA_PATH/preprocess
cd ..

cd lseg_encoder
CUDA_VISIBLE_DEVICES=0 python -u encode_images.py --backbone clip_vitl16_384 --weights demo_e200.ckpt --widehead --no-scaleinv --outdir $DATA_PATH/preprocess/semantic_features/rgb_feature_langseg --test-rgb-dir $DATA_PATH/preprocess/images --workers 0
cd ..

# Run training to distill features into 4D feature field
CUDA_VISIBLE_DEVICES=0 python run.py --config ./configs/wild/davis.yaml --src $DATA_PATH
