
# cuda11 暂未成功也是cpu训练
# open3d-ml
3d 深度学习框架，支持tf,pytorch。

升级了构建系统，增加了对 CUDA 11 的支持。我们的 pip 包现在包括对 CUDA 11.0、PyTorch 1.7.1 和 TensorFlow 2.4.1 的支持，以启用 RTX 3000 系列设备。请注意，我们为 Linux 提供了自定义 PyTorch WHEEL，以解决 CUDA 11、PyTorch 和扩展模块（如 Open3D-ML）之间的不兼容问题。

提供可视化工具，数据集下载等。

还提供了预训练模型（查看以下链接）。

https://github.com/isl-org/Open3D-ML/tree/r0.13.0

# insatll
RTX3090 470.42.01 CUDA 11.4 cudatookit11.0  tf  2.4.3 torch 1.7.1
git clone https://github.com/luckyluckydadada/lucky-Open3D-ML-0.13.git
cd lucky-Open3D-ML-0.13
conda create -n open3d-ml python=3.7
conda activate open3d-ml
conda install cudatookit=11.0
pip install open3d -i https://pypi.tuna.tsinghua.edu.cn/simple  --timeout=120
pip install -r requirements-tensorflow.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple  --timeout=120
pip install -r requirements-torch-cuda.txt  

# train
conda activate open3d-ml
## ranla+torch fail
python scripts/run_pipeline.py torch -c ml3d/configs/randlanet_s3dis.yml --dataset.dataset_path /home/lucky/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/  --pipeline SemanticSegmentation --dataset.use_cache True --dataset.ckpt_path /home/lucky/pretrain-model/randlanet_s3dis_202010091238.pth  
## ranla+tf success;CPU only;no pretrain-model
python scripts/run_pipeline.py tf    -c ml3d/configs/randlanet_s3dis.yml --dataset.dataset_path /home/lucky/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/  --pipeline SemanticSegmentation --dataset.use_cache True 
## kpconv+torch fail
python scripts/run_pipeline.py torch -c ml3d/configs/kpconv_s3dis.yml --dataset.dataset_path /home/lucky/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/  --pipeline SemanticSegmentation --dataset.use_cache True --dataset.ckpt_path /home/lucky/pretrain-model/kpconv_s3dis_202010091238.pth  
## kpconv+tf success;CPU only 
python scripts/run_pipeline.py tf    -c ml3d/configs/kpconv_s3dis.yml --dataset.dataset_path /home/lucky/data/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/  --pipeline SemanticSegmentation --dataset.use_cache True --dataset.ckpt_path /home/lucky/pretrain-model/kpconv_s3dis_202010091238