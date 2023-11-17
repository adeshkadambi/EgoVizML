# Common Issues and Errors

### AssertionError: Torch not compiled with CUDA enabled

This usually occurs as a result of running `poetry install` which ends up installing torch without cuda.

1. Uninstall torch and torchvision.

```bash
pip uninstall torch torchvision torchaudio
```

2. Re-install torch and torchvision with CUDA enabled.

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
