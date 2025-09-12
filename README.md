        PLUGIN_LIBRARY = "packages/libmyplugins_from11.4.so"
        ctypes.CDLL(PLUGIN_LIBRARY)

          File "/home/ma-user/work/copy/files/video-deal-search/video-deal-service/customize_service.py", line 34, in __init__
    ctypes.CDLL(PLUGIN_LIBRARY)
  File "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: libcudart.so.11.0: cannot open shared object file: No such file or directory
