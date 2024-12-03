# Python Implementation of Recommendation ITU-R P.1812

This code repository contains a python software implementation of  [Recommendation ITU-R P.1812-6](https://www.itu.int/rec/R-REC-P.1812/en) with a path-specific propagation prediction method for point-to-area terrestrial services in the frequency range 30 MHz to 6000 MHz.  

This code is functionally in line with the original reference [MATLAB/Octave Implementation of Recommendation ITU-R P.1812](https://github/eeveetza/p1812) approved by ITU-R Working Party 3M and published on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx).

<!-- This is a development code and it is not necessarily in line with the original reference [MATLAB/Octave Implementation of Recommendation ITU-R P.1812](https://github/eeveetza/p1812) approved by ITU-R Working Party 3M and published on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx).
-->

The package can be downloaded and installed using:
~~~
python -m pip install "git+https://github.com/eeveetza/Py1812#egg=Py1812"  
~~~
<!--
~~~
python -m pip install "git+https://github.com/eeveetza/Py1812@dev#egg=Py1812"  
~~~
--> 

and imported as follows
~~~
from Py1812 import P1812
~~~

| File/Folder               | Description                                                         |
|----------------------------|---------------------------------------------------------------------|
|`/src/Py1812/P1812.py`                | python implementation of Recommendation ITU-R P.1812-6   |
|`/src/Py1812/initiate_digital_maps.py`| python script that processes the ITU-R maps and generates the necessary `.npz` file. It needs to be run prior to using this software implementation. For details, see [Integrating ITU Digital Products](#integrating-itu-digital-products). |
|`/tests/validateP1812.py`          | python script used to validate the implementation of Recommendation ITU-R P.1812-6 in `P1812.bt_loss()`             |
|`/tests/validation_profiles/`    | Folder containing a proposed set of terrain profiles and inputs for validation of software implementations of this Recommendation |
|`/tests/validation_results/`	   | Folder containing all the results written during the transmission loss computations for the set of terrain profiles defined in the folder `./validation_profiles/` |

## Integrating ITU Digital Products

This software uses ITU digital products that are integral part of Recommendations. These products must not be reproduced or distributed without explicit written permission from the ITU.

### Setup Instructions

1. **Download and extract the required maps** to `./src/Py1812/maps`:

   - From ITU-R P.1812-7:
     - `N050.TXT`
     - `DN50.TXT`
   
2. **Run the script** `initiate_digital_maps.py` to generate the necessary file `P1812.npz`.

### Notes

- Ensure all files are placed in `./src/Py1812/maps` before running the script.
- The script processes the maps, which are critical for the software’s functionality.
- The resulting `*.npz` file is placed in the folder `./src/Py1812`.

## Function Call

The function `P1812.bt_loss` can be called

1. by invoking only the required input arguments:
~~~ 
Lb, Ep = P1812.bt_loss (f, p, d, h, R, Ct, zone, htg, hrg, pol, phi_t,  phi_r,  lam_t,  lam_r)
~~~
1. by invoking both the required and optional input arguments (the latter can be invoked in any order as key = value pairs):
~~~
Lb, Ep = P1812.bt_loss(f, p, d, h, R, Ct, zone, htg, hrg, pol, phi_t, phi_r, lam_t, lam_r, \
            pL = val_pL, sigmaL = val_sigmaL, Ptx = val_Ptx, DN = val_DN, N0 = val_N0 \
            dct = val_dct, dcr = val_dcr, flag4 = val_flag4, debug = val_debug, fid_log = val_fid_log)
~~~ 

## Required input arguments of function `tl_p1812`

| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `f`               | scalar double | GHz   | 0.03 ≤ `f` ≤ 6 | Frequency   |
| `p         `      | scalar double | %     | 1 ≤ `p` ≤ 50 | Time percentage for which the calculated basic transmission loss is not exceeded |
| `d`               | array double | km    | ~0.25 ≤ `max(d)` ≤ ~3000 | Terrain profile distances (in the ascending order from the transmitter)|
| `h`          | array double | m (asl)   |   | Terrain profile heights |
| `R`           | array double    | m      |              |  Representative clutter heights |
| `Ct`           | array int    |       |  1 - Water/sea, 2 - Open/rural, 3 - Suburban, 4 - Urban/trees/forest, 5 - Dense urban             |  Array of representative clutter types. If empty or all zeros, the default clutter type used is Open/rural |
| `zone`           | array int    |       | 1 - Sea, 3 - Coastal land, 4 - Inland             |  Radio-climatic zone types |
| `htg`           | scalar double    | m      |   1 ≤ `htg`  ≤ 3000          |  Tx antenna height above ground level |
| `hrg`           | scalar double    | m      |   1 ≤ `hrg`  ≤ 3000          |  Rx antenna height above ground level |
| `pol`           | scalar int    |       |   `pol`  = 1, 2          |  Polarization of the signal: 1 - horizontal, 2 - vertical |
| `phi_t`           | scalar double    | deg      |   -80 ≤ `phi_t`  ≤ 80          |  Latitude of Tx station |
| `phi_r`           | scalar double    | deg      |   -80 ≤ `phi_r`  ≤ 80          |  Latitude of Rx station |
| `lam_t`           | scalar double    | deg      |   -180 ≤ `lam_t`  ≤ 180          |  Longitude of Tx station |
| `lam_r`           | scalar double    | deg      |   -180 ≤ `lam_r`  ≤ 180          |  Longitude of Rx station |



## Optional input arguments of function `tl_p1812`
| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `pL`           | scalar double    | %      |   1 ≤ `pL`  ≤ 99          |  Location percentage for which the calculated basic transmission loss is not exceeded. Default is 50%. |
| `sigmaL`           | scalar double    | dB      |             |  location variability standard deviations computed using stdDev.m according to §4.8 and §4.10; the value of 5.5 dB used for planning Broadcasting DTT; Default: 0 dB. |
| `Ptx`           | scalar double    | kW      |   `Ptx` > 0          |  Tx power; Default: 1. |
| `DN`            | scalar double    | N-units/km      | `DN`> 0           | The average radio-refractivity lapse-rate through the lowest 1 km of the atmosphere at the path-center. It can be derived from an appropriate map. Default: 45. |
| `N0`           | scalar double    | N-units      |             | The sea-level surface refractivity at the path-centre. It can be derived from an appropriate map. Default: 325.|
| `dct`           | scalar double    | km      |   `dct` ≥ 0          |  Distance over land from the Tx antenna to the coast along the great-circle interference path. Default: 500 km. Set to zero for a terminal on a ship or sea platform.|
| `dcr`           | scalar double    | km      |   `dcr` ≥ 0          |  Distance over land from the Rx antenna to the coast along the great-circle interference path. Default: 500 km. Set to zero for a terminal on a ship or sea platform.|
| `flag4`           | scalar int    |       |             |  If `flag4`= 1, the alternative method from Attachment 4 to Annex 1 is used to calculate `Lbulls` without using terrain profile. Default: 0. |
| `debug`           | scalar int    |       |             |  If `debug`= 1, the results are written in log files. Default: 0. |
| `fid_log`           | scalar int    |       |     Only used if `debug`= 1        |  File identifier of the log file opened for writing outside the function. If not provided, a default file with a filename containing a timestamp will be created. |


 
## Outputs ##

| Variable   | Type   | Units | Description |
|------------|--------|-------|-------------|
| `Lb`    | double | dB    | Basic transmission loss |
| `Ep`    | double | dB(uV/m)    | Electric field strength |



## Software Versions
The code was tested and runs on:
* python3.9

## References

* [Recommendation ITU-R P.1812](https://www.itu.int/rec/R-REC-P.1812/en)

* [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx)

* [MATLAB/Octave Implementation of Recommendation ITU-R P.1812](https://github/eeveetza/p1812)
