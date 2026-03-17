# TruFor Docker image — inference only
# Based on pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
# Tested with TruFor_train_test/trufor_conda.yaml

FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

WORKDIR /trufor

# Copy the TruFor code (already cloned into trufor_repo/)
COPY trufor_repo/TruFor_train_test/ ./

# Install mmcv-full with pre-compiled wheels for torch 1.11 + CUDA 11.3
RUN pip install --no-cache-dir \
    mmcv-full==1.5.3 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install mmsegmentation and other dependencies
RUN pip install --no-cache-dir \
    mmsegmentation==0.25.0 \
    timm==0.5.4 \
    torch-dct==0.1.5 \
    jpegio==0.2.2 \
    albumentations==1.2.1 \
    scikit-image==0.16.2 \
    yacs==0.1.8 \
    tqdm==4.64.0 \
    seaborn==0.11.2 \
    pillow==9.0.1

# Default entry point: run test.py
ENTRYPOINT ["python", "test.py"]
