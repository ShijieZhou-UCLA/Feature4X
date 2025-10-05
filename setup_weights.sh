#### Preprocessing ####
mkdir weights
cd weights
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt
wget https://www.dropbox.com/s/4j4z58wuv8o0mfz/models.zip
unzip models.zip
mv models raft_models
mv models.zip raft_models.zip
cd ..

#### SAM2 ####
cd sam2
cd checkpoints && \
./download_ckpts.sh && \
cd ../..

# If this doesn't work when you are using docker on Windows:
# cd checkpoints
# sed -i 's/\r$//' download_ckpts.sh
# bash download_ckpts.sh
# cd ..


#### CLIP-LSeg ####
FILE_ID="1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb"
DEST="lseg_encoder/demo_e200.ckpt"
gdown --id $FILE_ID -O $DEST