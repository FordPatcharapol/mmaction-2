# install env

python -m venv .venv
activate env

# install python package

pip install -r requirements.txt

# install mm package

pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmpose
