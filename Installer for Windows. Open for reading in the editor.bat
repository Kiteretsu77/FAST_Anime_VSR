python -m venv venv
call .\venv\Scripts\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
pip install tensorrt_dispatch-8.6.0-cp310-none-win_amd64.whl
pip install tensorrt_lean-8.6.0-cp310-none-win_amd64.whl
pip install tensorrt-8.6.0-cp310-none-win_amd64.whl
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
call python setup.py install
