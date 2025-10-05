conda remove -n feature4x --all -y
conda create -n feature4x python=3.10 -y

conda activate feature4x

which python
which pip

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip3 install -U xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu124
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+cu124.html 
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html 
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+cu124.html 
pip install torch_geometric

pip install lib_render/gs3d/simple-knn
pip install lib_render/gs3d/gof-diff-gaussian-rasterization
pip install lib_render/gs3d/diff_gaussian_rasterization-alphadep-feature

pip install transformers==4.46.0 #git+https://github.com/huggingface/transformers.git
pip install lpips pytorch_msssim
pip install matplotlib==3.9.2

conda install -c "nvidia/label/cuda-12.4" libcusolver-dev -y
pip install mistral_inference
pip install --extra-index-url https://pypi.nvidia.com cuml-cu12
pip install viser
pip install loguru
pip install roma
pip install cupy-cuda12x
# if bug with numpy when running prepare.py
pip uninstall numpy
pip install numpy==1.26.4

# for lseg
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install pytorch-lightning==2.4.0
pip install lightning
pip install git+https://github.com/openai/CLIP.git