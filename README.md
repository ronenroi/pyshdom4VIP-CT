
:warning::warning::warning::warning::warning:
This project is built on the pyshdom project by Aviad Levis & Jesse Loveridge from
https://github.com/aviadlevis/pyshdom
:warning::warning::warning::warning::warning:

---

# pyshdom

Pyshdom performs 3D reconstruction of cloud microphysical properties from multi-angle, multi-spectral solar reflected radiation using a non-linear optimization procedure [[1],[2]]. The core radiative transfer routines are sourced from the Fortran SHDOM (Spherical Harmonic Discrete Ordinate Method for 3D Atmospheric Radiative Transfer) code by Frank K. Evans [[3]]. The python package was created by Aviad Levis, Amit Aides (Technion - Israel Institute of Technology) and Jesse Loveridge (University of Illinois).

[1]: http://openaccess.thecvf.com/content_iccv_2015/html/Levis_Airborne_Three-Dimensional_Cloud_ICCV_2015_paper.html
[2]: http://openaccess.thecvf.com/content_cvpr_2017/html/Levis_Multiple-Scattering_Microphysics_Tomography_CVPR_2017_paper.html
[3]: http://coloradolinux.com/~evans/shdom.html

&nbsp;

## Installation 
Installation using using anaconda package management

Start a clean virtual environment
```
conda create -n pyshdom python=3
source activate pyshdom
```

Install required packages
```
conda install anaconda dill tensorflow tensorboard pillow joblib
```

Install pyshdom distribution with (either install or develop flag)
```
python setup.py develop
```

&nbsp;


## Main scripts
For generating cloud data for training VIP-CT see the list below. 
  - VIP-CT_scripts/generate_data_fixed_imagers.py
  - VIP-CT_scripts/generate_data_varying_imagers.py

Change the corresponding config files in VIP-CT_scripts/configs. 

- CloudFieldFile: Cloud data raw txt folder.
- satellites_images_path: Output folder.
- GSD: The image ground spatial resolution.
- dx,dy,dz: The imager perturbation magnitude.
- n_formations: Number of imagers 


&nbsp;

## Usage and Contact
If you find this package useful please contact aviad.levis@gmail.com.
If you use this package in an academic publication please acknowledge the appropriate publications (see LICENSE file). 

