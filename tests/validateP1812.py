# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:50:46 2014
  This script is used to validate an implementation of Recommendation ITU-R P.1812
  as defined in the function Py1812.bt_loss()
  
  Author: Ivica Stevanovic (IS), Federal Office of Communications, Switzerland
  Revision History:
  Date            Revision
  29SEP22         Initial impolementation
"""
import csv
import sys
import os
from traceback import print_last
import numpy as np
import matplotlib.pyplot as pl


from Py1812 import P1812

tol = 1e-8
hit = 0
total = 0


# path to the folder containing test profiles
pathname = "./validation_profiles/"

# path to the folder where the resulting log files will be saved
out_dir = "./validation_results/"

# format of the test profile (measurement) files
fileformat = "Fryderyk_csv"

# Clutter code type
ClutterCode = "GlobCover"

#     ClutterCode='default';  # default clutter code assumes land, rural area with R1 = R2 = 10;
#     ClutterCode='TBD'
#     ClutterCode='OFCOM'
#     ClutterCode='NLCD'
#     ClutterCode='LULC'
#     ClutterCode='GlobCover'
#     ClutterCode='DNR1812'
#     ClutterCode = 'P1546'

# set to 1 if the csv log files need to be produced (together with stdout)
flag_debug = 1

# set to 1 if the plots of the height profile are to be shown
flag_plot = 0

# pathprofile is available (=1), not available (=0)
flag_path = 1

# set to 1 if Attachment 4 to Annex 1 is to be used for computation of
# the spherical earth diffraction Lbs w/o terrain profile analysis

flag4 = 0

# set variabilities to zero and location percentage to 50
pL = 50
sigmaL = 0

# begin code
# Collect all the filenames .csv in the folder pathname that contain the profile data
try:
    filenames = [f for f in os.listdir(pathname) if f.endswith(".csv")]
except:
    print("The system cannot find the given folder " + pathname)


# create the output directory

try:
    os.makedirs(out_dir)
except OSError:
    if not os.path.isdir(out_dir):
        raise

if flag_debug == 1:
    fid_all = open(out_dir + "combined_results.csv", "w")
    if fid_all == -1:
        raise IOError("The file combined_results.csv could not be opened")

    fid_all.write("# # %s, %s, %s, %s, %s, %s\n" % ("Folder", "Filename", "Dataset #", "Reference", "Predicted", "Deviation: Predicted-Reference"))
if len(filenames) < 1:
    raise IOError("There are no .csv files in the test profile folder " + pathname)

# figure counter
fig_cnt = 0

for filename1 in filenames:
    print("***********************************************\n")
    print("Processing file " + pathname + filename1 + "\n")
    print("***********************************************\n")

    # read the file and populate sg3db input data structure
    sg3db = P1812.read_sg3_measurements2(pathname + filename1, fileformat)

    # collect intermediate results in log files (=1), or not (=0)
    sg3db.debug = flag_debug

    # pathprofile is available (=1), not available (=0)
    sg3db.pathinfo = flag_path

    # update the data structure with the Tx Power (kW)
    for kindex in range(0, sg3db.Ndata):
        PERP = sg3db.ERPMaxTotal[kindex]
        HRED = sg3db.HRPred[kindex]
        PkW = 10.0 ** (PERP / 10.0) * 1e-3  # kW

        if np.isnan(PkW):
            # use complementary information from Basic Transmission Loss and
            # received measured strength to compute the transmitter power + gain
            E = sg3db.MeasuredFieldStrength[kindex]
            PL = sg3db.BasicTransmissionLoss[kindex]
            f = sg3db.frequency[kindex]
            PdBkW = -137.2217 + E - 20 * np.log10(f) + PL
            PkW = 10 ** (PdBkW / 10.0)

        sg3db.TransmittedPower = np.append(sg3db.TransmittedPower, PkW)

    sg3db.ClutterCode = []

    x = sg3db.x
    h_gamsl = sg3db.h_gamsl

    # # plot the profile
    if flag_plot:
        fig_cnt = fig_cnt + 1
        newfig = pl.figure(fig_cnt)
        h_plot = pl.plot(x, h_gamsl, linewidth=2, color="k")
        pl.xlim(np.min(x), np.max(x))
        hTx = sg3db.hTx
        hRx = sg3db.hRx

        pl.title("Tx: " + sg3db.TxSiteName + ", Rx: " + sg3db.RxSiteName + ", " + sg3db.TxCountry + sg3db.MeasurementFileName)
        pl.grid(True)
        pl.xlabel("distance [km]")
        pl.ylabel("height [m]")

    # # plot the position of transmitter/receiver

    hTx = sg3db.hTx
    hRx = sg3db.hRx

    if flag_plot:
        ax = pl.gca()

    for measID in range(0, len(hRx)):
        if measID != []:
            if measID > len(hRx) or measID < 0:
                raise ValueError("The chosen dataset does not exist.")
            # print('Computing the fields for Dataset #%d\n', %(dataset))
            sg3db.userChoiceInt = measID
            hhRx = hRx[measID]
            hhTx = hTx[0]
            # this will be a separate function
            # Transmitter
            if flag_plot:
                if sg3db.first_point_transmitter == 1:
                    pl.plot(np.array([x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0] + hhTx]), linewidth=2, color="b")
                    pl.plot(x[0], h_gamsl[0] + hTx[0], marker="v", color="b")
                    pl.plot(np.array([x[-1], x[-1]]), np.array([h_gamsl[-1], h_gamsl[-1] + hhRx]), linewidth=2, color="r")
                    pl.plot(x[-1], h_gamsl[-1] + hhRx, marker="v", color="r")
                else:
                    pl.plot(np.array([x[-1], x[-1]]), np.array([h_gamsl[-1], h_gamsl[-1] + hhTx]), linewidth=2, color="b")
                    pl.plot(x[-1], h_gamsl[0] + hTx[0], marker="v", color="b")
                    pl.plot(np.array([x[0], x[0]]), np.array([h_gamsl[0], h_gamsl[0] + hhRx]), linewidth=2, color="r")
                    pl.plot(x[0], h_gamsl[0] + hhRx, marker="v", color="r")

                ax = pl.gca()

        if not P1812.isempty(sg3db.coveragecode):
            # fill in the  missing fields in Rx clutter
            i = sg3db.coveragecode[-1]
            RxClutterCode, RxP1546Clutter, R2external = P1812.clutter(i, ClutterCode)
            i = sg3db.coveragecode[0]
            TxClutterCode, TxP1546Clutter, R1external = P1812.clutter(i, ClutterCode)

            sg3db.RxClutterCodeP1546 = RxP1546Clutter

            if not P1812.isempty(sg3db.h_ground_cover):
                if not np.isnan(sg3db.h_ground_cover[-1]):
                    if sg3db.h_ground_cover[-1] > 3:
                        sg3db.RxClutterHeight = sg3db.h_ground_cover[-1]
                    else:
                        sg3db.RxClutterHeight = R2external

                else:
                    sg3db.RxClutterHeight = R2external

                if not np.isnan(sg3db.h_ground_cover[0]):
                    sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                    if sg3db.h_ground_cover[0] > 3:
                        sg3db.TxClutterHeight = sg3db.h_ground_cover[0]
                    else:
                        sg3db.TxClutterHeight = R1external

                else:
                    sg3db.TxClutterHeight = R1external

            else:
                sg3db.RxClutterHeight = R2external
                sg3db.TxClutterHeight = R1external

        # Execute P.1812
        fid_log = -1
        if flag_debug == 1:
            filename2 = out_dir + filename1[0:-4] + "_" + str(measID) + "_log.csv"
            fid_log = open(filename2, "w")
            if fid_log == -1:
                error_str = filename2 + " cannot be opened."
                raise IOError(error_str)

        sg3db.fid_log = fid_log

        sg3db.dct = 500
        sg3db.dcr = 500

        if sg3db.radio_met_code[0] == 1:  # Tx at sea
            sg3db.dct = 0

        if sg3db.radio_met_code[-1] == 1:  # Rx at sea
            sg3db.dcr = 0

        sg3db.Lb, sg3db.PredictedFieldStrength = P1812.bt_loss(
            sg3db.frequency[measID] / 1e3,
            sg3db.TimePercent[measID],
            sg3db.x,
            sg3db.h_gamsl,
            sg3db.h_ground_cover,
            sg3db.coveragecode,
            sg3db.radio_met_code,
            sg3db.hTx[measID],
            sg3db.hRx[measID],
            sg3db.polHVC[measID],
            sg3db.TxLAT,
            sg3db.RxLAT,
            sg3db.TxLON,
            sg3db.RxLON,
            pL=pL,
            sigmaL=sigmaL,
            Ptx=sg3db.TransmittedPower[measID],
            DN=sg3db.DN,
            N0=sg3db.N0,
            dct=sg3db.dct,
            dcr=sg3db.dcr,
            flag4=flag4,
            debug=flag_debug,
            fid_log=sg3db.fid_log
        )

        delta = sg3db.PredictedFieldStrength - sg3db.MeasuredFieldStrength[measID]

        if abs(delta) <= tol:
            hit = hit + 1
        else: 
            print("Validation failed, deviation from the reference results %g is larger than %g.\n" % (abs(delta), tol))

        total = total + 1

        if flag_debug:
            fid_log.close()

            # print the deviation of the predicted from the measured value,
            # double check this line
            # Measurement folder | Measurement File | Dataset | Measured Field Strength | Predicted Field Strength | Deviation from Measurement
            fid_all.write(" %s, %s, %d, %.8f, %.8f, %.8f\n" % (sg3db.MeasurementFolder, sg3db.MeasurementFileName, measID, sg3db.MeasuredFieldStrength[measID], sg3db.PredictedFieldStrength, delta))
            # print(' %s, %s, %d, %.2f, %.2f, %.2f\n' % (sg3db.MeasurementFolder,sg3db.MeasurementFileName,measID, sg3db.MeasuredFieldStrength[measID], sg3db.PredictedFieldStrength, sg3db.PredictedFieldStrength - sg3db.MeasuredFieldStrength[measID]))

if flag_debug == 1:
    fid_all.close()

print("Validation results: %d out of %d tests passed successfully.\n" % (hit, total))
if hit == total:
    print("The deviation from the reference results is smaller than %g.\n" % (tol))
