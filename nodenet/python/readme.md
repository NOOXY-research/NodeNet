## Requirement
- Python
- Python package: NumPy
- Python package: Matplotlib
- Python package: CuPy(Optional)

to install Python packages
```bash
#!/bin/bash
sudo python3 -m pip install numpy matplotlib
```
to use Nvdia GPU (example: ArchLinux)
```bash
#!/bin/bash
sudo pacman -S cuda
# Install CUDA
sudo vim /etc/profile
# In vim add /opt/cuda/bin to your path
# finally
sudo python3 -m pip install cupy
# if something wrong try
source /etc/profile
# or
sudo reboot now
```

If you need Nvidia GPU support(for really large size of neural network) simply change nodenet/imports/commons.py
```python
import numpy as np
```
to
```python
import cupy as np
```
