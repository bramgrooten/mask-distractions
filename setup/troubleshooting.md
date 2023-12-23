
## Troubleshooting

If you get an error like `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`,
you may need to install `libgl1-mesa-glx`.

If you get an error like `ImportError: libGLEW.so.2.1: cannot open shared object file: No such file or directory`,
you may need to install `libglew-dev`.

If you get an error like `ImportError: libjpeg.so.8: cannot open shared object file: No such file or directory`,
you may need to install `libjpeg-dev`.

If you get an error like `ImportError: libGLU.so.1: cannot open shared object file: No such file or directory`,
you may need to install `libglu1-mesa-dev`.

If you get an error like `ImportError: libSM.so.6: cannot open shared object file: No such file or directory`,
you may need to install `libsm6`.

If you get an error like
```
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
```
you may need to install `libgl1-mesa-dri` or use the solution from https://stackoverflow.com/a/72200748/4230999
which is: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`

Packages that you may need to install if you still get errors:
```
conda activate madi
conda install -c conda-forge glew
```

If you get an error like 
```
KeyError: 'DMCGB_DATASETS'
Segmentation fault
```
then make sure you've set the environment variable `DMCGB_DATASETS` to the path where you store the external datasets.
(See the Dataset section of the README.)

Loading the `places365_standard` dataset can take some time, up to 15 minutes.

If you get an error that just says `Killed`, you may need to increase the memory limit for your process.
(You can try closing other programs first, and restart your computer if that doesn't work.)

