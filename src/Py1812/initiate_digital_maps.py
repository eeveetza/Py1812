# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,line-too-long,too-many-lines,too-many-arguments,too-many-locals,too-many-statements
"""
This Recommendation uses integral digital products. They form an integral
part of the recommendation and may not be reproduced, by any means whatsoever,
without written permission of ITU.

The user should download the necessary digital maps that are used by this
implementation directly from ITU-R web site and place the files in the
folder ./maps. After that, the user should execute the script
contained in this file "initiate_digital_maps.py". The script produces the
necessary maps in .npz format
interpolations.

The following maps should be extracted in the folder ./private/maps:
From ITU-R P.1812: DN50.TXT, N050.TXT

Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
Revision History:
Date            Revision
03DEC2024       Initial version (IS)  
"""

import os
import numpy as np

# Necessary maps
filepath = "./maps/"
filenames = ["DN50.TXT", "N050.TXT"]

# begin code
maps = dict();
failed = False
for filename in filenames:

      # Load the file into a NumPy array
      try:
            print(f"Processing file {filename}");
            matrix = np.loadtxt(filepath + filename)
            # Print the NumPy array (matrix)
            key = filename[0:-4]
            maps[key] = matrix
                        
      except OSError:
            print(f"Error: {filename} does not exist or cannot be opened.")
            failed = True
      except ValueError:
            print(f"Error: The file {filename} contains invalid data for a float matrix.")
            failed = True
      

if (not failed):
      # Save matrices using dynamically provided names
      np.savez('P1812.npz', **maps)

      print("P1812.npz file created successfully.")
      
else:
      print("The process failed.")


