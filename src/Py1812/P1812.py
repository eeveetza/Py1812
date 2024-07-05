# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,line-too-long,too-many-lines,too-many-arguments,too-many-locals,too-many-statements
"""
Created on Tue 27 Sep 2022

@author: eeveetza
"""

import os
import datetime
import numpy as np
from importlib.resources import files

DigitalMaps = {}
with np.load(files("Py1812").joinpath("P1812.npz")) as DigitalMapsNpz:
    for k in DigitalMapsNpz.files:
        DigitalMaps[k] = DigitalMapsNpz[k].copy()


def bt_loss(f, p, d, h, R, Ct, zone, htg, hrg, pol, phi_t, phi_r, lam_t, lam_r, **kwargs):
    """
    P1812.bt_loss basic transmission loss according to P.1812-6
    Lb = P1812.bt_lossbt_loss(f, p, d, h, R, Ct, zone, htg, hrg, pol, phi_t, phi_r, lam_t, lam_r)

    This is the MAIN function that computes the basic transmission loss not exceeded for p% time
    and pL% locations, including additional losses due to terminal surroundings
    and the field strength exceeded for p% time and pL% locations
    as defined in ITU-R P.1812-6.
    This function:
    does not include the building entry loss (only outdoor scenarios implemented)

    Other functions called from this function are in ./private/ subfolder.

    Input parameters:
    f       -   Frequency (GHz)
    p       -   Required time percentage for which the calculated basic
                transmission loss is not exceeded
    d       -   vector of distances di of the i-th profile point (km)
    h       -   vector of heights hi of the i-th profile point (meters
                above mean sea level.
    R       -   vector of representative clutter height Ri of the i-th profile point (m)
    Ct      -   vector of representative clutter type Cti of the i-th profile point
                Water/sea (1), Open/rural (2), Suburban (3),
                Urban/trees/forest (4), Dense urban (5)
                if empty or all zeros, the default clutter used is Open/rural
    zone    -   vector of radio-climatic zone types: Inland (4), Coastal land (3), or Sea (1)
    htg     -   Tx Antenna center heigth above ground level (m)
    hrg     -   Rx Antenna center heigth above ground level (m)
    pol     -   polarization of the signal (1) horizontal, (2) vertical
    phi_t    - latitude of Tx station (degrees)
    phi_r    - latitude of Rx station (degrees)
    lam_t    - longitude of Tx station (degrees)
    lam_r    - longitude of Rx station (degrees)

    Optional input parameters (using keywords):
    pL      -   Required time percentage for which the calculated basic
                transmission loss is not exceeded (1% - 99%)
    sigmaL  -   location variability standard deviations computed using
                stdDev.m according to §4.8 and §4.10
                the value of 5.5 dB used for planning Broadcasting DTT
    Ptx     -   Transmitter power (kW), default value 1 kW
    DN      -   The average radio-refractive index lapse-rate through the
                lowest 1 km of the atmosphere (it is a positive quantity in this
                procedure) (N-units/km)
    N0      -   The sea-level surface refractivity, is used only by the
                troposcatter model as a measure of location variability of the
                troposcatter mechanism. The correct values of DN and N0 are given by
                the path-centre values as derived from the appropriate
                maps (N-units)
    dct     -   Distance over land from the transmit and receive
    dcr         antennas to the coast along the great-circle interference path (km).
                default values dct = 500 km, dcr = 500 km, or
                set to zero for a terminal on a ship or sea platform
    flag4   -   Set to 1 if the alternative method is used to calculate Lbulls
                without using terrain profile analysis (Attachment 4 to Annex 1)
    debug   -   Set to 1 if the log files are to be written,
                otherwise set to default 0
    fid_log  -   if debug == 1, a file identifier of the log file can be
                provided, if not, the default file with a file
                containing a timestamp will be created


    Output parameters:
    Lb     -   basic  transmission loss according to P.1812-6

    Example:
    1) Call with required input parameters
    Lb = bt_loss(f,p,d,h,R,Ct,zone,htg,hrg,pol,phi_t,phi_r,lam_t,lam_r)
    2) Call with required input parameters and optional input parameters as keyword
    Lb = bt_loss(f,p,d,h,R,Ct,zone,htg,hrg,pol,phi_t,phi_r,lam_t,lam_r,DN = 50, N0 = 400)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    28SEP22     Ivica Stevanovic, OFCOM         Initial version
    v1    28MAR24     Ivica Stevanovic, OFCOM         Introduced datamaps for DN and N0
                                                      Fixed a bug with timestamp as proposed by https://github.com/drcaguiar


    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
    OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
    OTHER DEALINGS IN THE SOFTWARE.
    THE AUTHOR(S) AND OFCOM (CH) DO NOT PROVIDE ANY SUPPORT FOR THIS SOFTWARE

    """

    # Set default values for optional arguments

    pL = kwargs.get("pL", 50.0)
    sigmaL = kwargs.get("sigmaL", 0.0)
    Ptx = kwargs.get("Ptx", 1.0)
    DN = kwargs.get("DN", [])
    N0 = kwargs.get("N0", [])
    dct = kwargs.get("dct", 500.0)
    dcr = kwargs.get("dcr", 500.0)
    flag4 = kwargs.get("flag4", 0)
    debug = kwargs.get("debug", 0)
    fid_log = kwargs.get("fid_log", [])

    # Ensure that vector d is ascending
    if not issorted(d):
        raise ValueError("The array of path profile points d(i) must be in ascending order.")

    # Ensure that d[0] = 0 (Tx position)
    if d[0] > 0.0:
        raise ValueError("The first path profile point d[0] = " + str(d[0]) + " must be zero.")

    # verify input argument values and limits

    if not (f >= 0.03 and f <= 6.0):
        print("Warning: frequency must be in the range [0.03, 6] GHz. ")
        print("Computation will continue but the parameters are outside of the valid domain.")

    if not (p >= 1 and p <= 50):
        raise ValueError("The time percentage must be in the range [1, 50]%")

    if not (htg >= 1 and htg <= 3000):
        raise ValueError("The Tx antenna height must be in the range [1, 3000] m")

    if not (hrg >= 1 and hrg <= 3000):
        raise ValueError("The Rx antenna height must be in the range [1, 3000] m")

    if not (pol == 1 or pol == 2):
        raise ValueError("The polarization pol can be either 1 (horizontal) or 2 (vertical).")

    # make sure that there is enough points in the path profile
    if len(d) <= 4:
        raise ValueError("The number of points in path profile should be larger than 4")

    xx = np.logical_or(zone == 1, np.logical_or(zone == 3, zone == 4))
    if np.any(xx == False):
        raise ValueError("The vector of radio-climatic zones zone may only contain integers 1, 3, or 4.")

    if not (pL > 0 and pL < 100):
        raise ValueError("The location percentage must be in the range (0, 100)%")

    if not (Ptx > 0):
        raise ValueError("The Tx power must be positive.")

    if dct < 0 or dcr < 0:
        raise ValueError("Distances dct and dcr must be positive.")

    if sigmaL < 0:
        raise ValueError("Standard deviation in location variability must be positive.")

    if not (flag4 == 0 or flag4 == 1):
        raise ValueError("The parameter flag4 can be either 0 or 1.")

    NN = len(d)

    # the number of elements in d and path need to be the same
    if not (len(h) == NN):
        raise ValueError("The number of elements in the array d and the array h must be the same.")

    if isempty(R):
        R = np.zeros(h.shape)  # default is clutter height zero
    elif not (len(R) == NN):
        raise ValueError("The number of elements in the array d and the array R must be the same.")

    if isempty(Ct):
        Ct = 2 * np.ones(h.shape)  # default is Open/rural clutter type

    elif Ct.any() == 0:
        Ct = 2 * np.ones(h.shape)
        # default is Open/rural clutter type
    else:
        if not (len(Ct) == NN):
            raise ValueError("The number of elements in the array d and the array Ct must be the same.")

    if isempty(zone):
        zone = 4 * np.ones(h.shape)  # default is Inland radio-meteorological zone
    else:
        if not (len(zone) == NN):
            raise ValueError("The array d and the array zone must be of the same size.")

    if zone[0] == 1:  # Tx at sea
        dct = 0

    if zone[-1] == 1:  # Rx at sea
        dcr = 0
        
    # Path center latitude
    Re = 6371
    dpnt = 0.5 * (d[-1] - d[0])
    lam_path, phi_path, Bt2r, dgc = great_circle_path(lam_r, lam_t, phi_r, phi_t, Re, dpnt)

    if isempty(DN):
        # Find radio-refractivity lapse rate dN 
        # using the digital maps at phim_e (lon), phim_n (lat) - as a bilinear interpolation
        DN50 = DigitalMaps["DN50"]
        DN = interp2(DN50, lam_path, phi_path, 1.5, 1.5)

    if isempty(N0):
        # Find radio-refractivity 
        # using the digital maps at phim_e (lon), phim_n (lat) - as a bilinear interpolation
        N050 = DigitalMaps["N050"]
        N0 = interp2(N050, lam_path, phi_path, 1.5, 1.5)

    # handle number fidlog is reserved here for writing the files
    # if fidlog is already open outside of this function, the file needs to be
    # empty (nothing written), otherwise it will be closed and opened again
    # if fid is not open, then a file with a name corresponding to the time
    # stamp will be opened and closed within this function.

    inside_file = 0
    if debug:
        if isempty(fid_log):
            fid_log = open('P1812_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '_log.csv', 'w')
            inside_file = 1
            if fid_log == -1:
                raise IOError("The log file could not be opened")

        else:
            inside_file = 0

    floatformat = "%.10g,\n"

    if debug:
        fid_log.write("# Parameter,Ref,,Value,\n")
        fid_log.write("Ptx (kW),,," + floatformat % (Ptx))
        fid_log.write("f (GHz),,," + floatformat % (f))
        fid_log.write("p (%),,," + floatformat % (p))
        fid_log.write("pL (%),,," + floatformat % (pL))
        fid_log.write("sigmaL (dB),,," + floatformat % (sigmaL))
        fid_log.write("phi_t (deg),,," + floatformat % (phi_t))
        fid_log.write("phi_r (deg),,," + floatformat % (phi_r))
        fid_log.write("lam_t (deg),,," + floatformat % (lam_t))
        fid_log.write("lam_r (deg),,," + floatformat % (lam_r))
        fid_log.write("htg (m),,," + floatformat % (htg))
        fid_log.write("hrg (m),,," + floatformat % (hrg))
        fid_log.write("pol,,," "%d,\n" % (pol))
        fid_log.write("DN ,,," + floatformat % (DN))
        fid_log.write("N0 ,,," + floatformat % (N0))
        fid_log.write("dct (km) ,,," + floatformat % (dct))
        fid_log.write("dcr (km) ,,," + floatformat % (dcr))
        fid_log.write("R2 (m) ,,," + floatformat % (R[1]))
        fid_log.write("Rn-1 (m) ,,," + floatformat % (R[-2]))
        fid_log.write("Ct Tx ,Table 2,," + floatformat % (Ct[1]))
        fid_log.write("Ct Rx ,Table 2,," + floatformat % (Ct[-2]))

    # Compute the path profile parameters

    # Compute  dtm     -   the longest continuous land (inland + coastal =34) section of the great-circle path (km)
    zone_r = 34
    dtm = longest_cont_dist(d, zone, zone_r)

    # Compute  dlm     -   the longest continuous inland section (4) of the great-circle path (km)
    zone_r = 4
    dlm = longest_cont_dist(d, zone, zone_r)

    # Compute b0
    b0 = beta0(phi_path, dtm, dlm)

    ae, ab = earth_rad_eff(DN)

    # Compute the path fraction over see Eq (1)

    omega = path_fraction(d, zone, 1)

    # Derive parameters for the path profile analysis

    hst_n, hsr_n, hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta, pathtype = smooth_earth_heights(d, h, R, htg, hrg, ae, f)

    dtot = d[-1] - d[0]

    # Tx and Rx antenna heights above mean sea level amsl (m)
    hts = h[0] + htg
    hrs = h[-1] + hrg

    # Modify the path by adding representative clutter, according to Section 3.2
    # excluding the first and the last point
    g = h + R
    g[0] = h[0]
    g[-1] = h[-1]

    # Compute htc and hrc as defined in Table 5 (P.1812-6)
    htc = hts
    hrc = hrs

    if debug:
        fid_log.write(",,,,\n")
        fid_log.write("d (km),,," + floatformat % (dtot))
        fid_log.write("dlt (km),Eq (78),," + floatformat % (dlt))
        fid_log.write("dlr (km),Eq (81a),," + floatformat % (dlr))
        fid_log.write("th_t (mrad),Eqs (76-78),," + floatformat % (theta_t))
        fid_log.write("th_r (mrad),Eqs (79-81),," + floatformat % (theta_r))
        fid_log.write("th (mrad),Eq (82),," + floatformat % (theta))
        fid_log.write("hts (m),,," + floatformat % (hts))
        fid_log.write("hrs (m),,," + floatformat % (hrs))
        fid_log.write("htc (m),Table 5,," + floatformat % (htc))
        fid_log.write("hrc (m),Table 5,," + floatformat % (hrc))
        fid_log.write("w,Table 5,," + floatformat % (omega))
        fid_log.write("dtm (km),Sec 3.6,," + floatformat % (dtm))
        fid_log.write("dlm (km),Sec 3.6,," + floatformat % (dlm))
        fid_log.write("phi (deg),Eq (4),," + floatformat % (phi_path))
        fid_log.write("b0 (%),Eq (5),," + floatformat % (b0))
        fid_log.write("ae (km),Eq (7a),," + floatformat % (ae))
        fid_log.write("hst (m),Eq (85),," + floatformat % (hst_n))
        fid_log.write("hsr (m),Eq (86),," + floatformat % (hsr_n))
        fid_log.write("hst (m),Eq (90a),," + floatformat % (hst))
        fid_log.write("hsr (m),Eq (90b),," + floatformat % (hsr))
        fid_log.write("hstd (m),Eq (89),," + floatformat % (hstd))
        fid_log.write("hsrd (m),Eq (89),," + floatformat % (hsrd))
        fid_log.write("htc" " (m),Eq (37a),," + floatformat % (htc - hstd))
        fid_log.write("hrc" " (m),Eq (37b),," + floatformat % (hrc - hsrd))
        fid_log.write("hte (m),Eq (92a),," + floatformat % (hte))
        fid_log.write("hre (m),Eq (92b),," + floatformat % (hre))
        fid_log.write("hm (m),Eq (93),," + floatformat % (hm))
        fid_log.write("\n")

    # Calculate an interpolation factor Fj to take account of the path angular
    # distance Eq (57)

    THETA = 0.3
    KSI = 0.8

    Fj = 1.0 - 0.5 * (1.0 + np.tanh(3.0 * KSI * (theta - THETA) / THETA))

    # Calculate an interpolation factor, Fk, to take account of the great
    # circle path distance:

    dsw = 20
    kappa = 0.5

    Fk = 1.0 - 0.5 * (1.0 + np.tanh(3.0 * kappa * (dtot - dsw) / dsw))  # eq (58)

    Lbfs, Lb0p, Lb0b = pl_los(dtot, hts, hrs, f, p, b0, dlt, dlr)

    Ldp, Ldb, Ld50, Lbulla50, Lbulls50, Ldsph50 = dl_p(d, g, htc, hrc, hstd, hsrd, f, omega, p, b0, DN, flag4)

    # The median basic transmission loss associated with diffraction Eq (42)

    Lbd50 = Lbfs + Ld50

    # The basic tranmission loss associated with diffraction not exceeded for
    # p% time Eq (43)

    Lbd = Lb0p + Ldp

    # A notional minimum basic transmission loss associated with LoS
    # propagation and over-sea sub-path diffraction

    Lminb0p = Lb0p + (1 - omega) * Ldp

    # eq (40a)
    Fi = 1

    if p >= b0:
        Fi = inv_cum_norm(p / 100.0) / inv_cum_norm(b0 / 100.0)

        Lminb0p = Lbd50 + (Lb0b + (1 - omega) * Ldp - Lbd50) * Fi  # eq (59)

    # Calculate a notional minimum basic transmission loss associated with LoS
    # and transhorizon signal enhancements

    eta = 2.5

    Lba = tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, omega, ae, b0)

    Lminbap = eta * np.log(np.exp(Lba / eta) + np.exp(Lb0p / eta))  # eq (60)

    # Calculate a notional basic transmission loss associated with diffraction
    # and LoS or ducting/layer reflection enhancements

    Lbda = Lbd

    if Lminbap <= Lbd[0]:
        Lbda[0] = Lminbap + (Lbd[0] - Lminbap) * Fk

    if Lminbap <= Lbd[1]:
        Lbda[1] = Lminbap + (Lbd[1] - Lminbap) * Fk

    # Calculate a modified basic transmission loss, which takes diffraction and
    # LoS or ducting/layer-reflection enhancements into account

    Lbam = Lbda + (Lminb0p - Lbda) * Fj  # eq (62)

    # Calculate the basic transmission loss due to troposcatter not exceeded
    # for any time percantage p

    Lbs = tl_tropo(dtot, theta, f, p, N0)

    # Calculate the final transmission loss not exceeded for p% time

    Lbc_pol = -5 * np.log10(10 ** (-0.2 * Lbs) + 10 ** (-0.2 * Lbam))  # eq (63)

    Lbc = Lbc_pol[int(pol - 1)]

    Lloc = 0.0  # outdoors only (67a)

    # Location variability of losses (Section 4.8)
    if not (zone[-1] == 1):  # Rx not at sea
        Lloc = -inv_cum_norm(pL / 100.0) * sigmaL

    # Basic transmission loss not exceeded for p% time and pL% locations
    # (Sections 4.8 and 4.9) not implemented

    Lb = max(Lb0p, Lbc + Lloc)  #  eq (69)

    # The field strength exceeded for p% time and pL% locations

    Ep = 199.36 + 20 * np.log10(f) - Lb  # eq (70)

    # Scale to the transmitter power

    EpPtx = Ep + 10 * np.log10(Ptx)

    if debug:
        fid_log.write("Fi,Eq (40),," + floatformat % (Fi))
        fid_log.write("Fj,Eq (57),," + floatformat % (Fj))
        fid_log.write("Fk,Eq (58),," + floatformat % (Fk))
        fid_log.write("Lbfs,Eq (8),," + floatformat % (Lbfs))
        fid_log.write("Lb0p,Eq (10),," + floatformat % (Lb0p))
        fid_log.write("Lb0b,Eq (11),," + floatformat % (Lb0b))
        fid_log.write("Lbulla (dB),Eq (21),," + floatformat % (Lbulla50))
        fid_log.write("Lbulls (dB),Eq (21),," + floatformat % (Lbulls50))
        fid_log.write("Ldsph (dB),Eq (27),," + floatformat % (Ldsph50[int(pol - 1)]))
        fid_log.write("Ld50 (dB),Eq (39),," + floatformat % (Ld50[int(pol - 1)]))
        fid_log.write("Ldb (dB),Eq (39),," + floatformat % (Ldb[int(pol - 1)]))
        fid_log.write("Ldp (dB),Eq (41),," + floatformat % (Ldp[int(pol - 1)]))
        fid_log.write("Lbd50 (dB),Eq (42),," + floatformat % (Lbd50[int(pol - 1)]))
        fid_log.write("Lbd (dB),Eq (43),," + floatformat % (Lbd[int(pol - 1)]))

        fid_log.write("Lminb0p (dB),Eq (59),," + floatformat % (Lminb0p[int(pol - 1)]))

        fid_log.write("Lba (dB),Eq (46),," + floatformat % (Lba))
        fid_log.write("Lminbap (dB),Eq (60),," + floatformat % (Lminbap))

        fid_log.write("Lbda (dB),Eq (61),," + floatformat % (Lbda[int(pol - 1)]))
        fid_log.write("Lbam (dB),Eq (62),," + floatformat % (Lbam[int(pol - 1)]))
        fid_log.write("Lbs (dB),Eq (44),," + floatformat % (Lbs))

        fid_log.write("Lbc (dB),Eq (63),," + floatformat % (Lbc))
        fid_log.write("Lb (dB),Eq (69),," + floatformat % (Lb))
        fid_log.write("Ep (dBuV/m),Eq (70),," + floatformat % (Ep))
        fid_log.write("Ep (dBuV/m) w.r.t. Ptx,,," + floatformat % (EpPtx))
        
    Ep = EpPtx

    return Lb, Ep


def tl_tropo(dtot, theta, f, p, N0):
    """
    tl_tropo Basic transmission loss due to troposcatterer to P.1812-6
    Lbs = tl_tropo(dtot, theta, f, p, N0)

    This function computes the basic transmission loss due to troposcatterer
    not exceeded for p% of time
    as defined in ITU-R P.1812-6 (Section 4.4)

        Input parameters:
        dtot    -   Great-circle path distance (km)
        theta   -   Path angular distance (mrad)
        f       -   frequency expressed in GHz
        p       -   percentage of time
        N0      -   path centre sea-level surface refractivity derived from Fig. 6

        Output parameters:
        Lbs    -   the basic transmission loss due to troposcatterer
                    not exceeded for p% of time

        Example:
        Lbs = tl_tropo(dtot, theta, f, p, N0)


        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    29SEP22    Ivica Stevanovic, OFCOM         Initial version
    """

    # Frequency dependent loss

    Lf = 25 * np.log10(f) - 2.5 * (np.log10(f / 2.0)) ** 2  # eq (45)

    # the basic transmission loss due to troposcatter not exceeded for any time
    # percentage p, below 50# is given

    Lbs = 190.1 + Lf + 20 * np.log10(dtot) + 0.573 * theta - 0.15 * N0 - 10.125 * (np.log10(50.0 / p)) ** (0.7)

    return Lbs


def tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, omega, ae, b0):
    """
    tl_anomalous Basic transmission loss due to anomalous propagation according to P.452-17
    Lba = tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, omega, ae, b0)

    This function computes the basic transmission loss occuring during
    periods of anomalous propagation (ducting and layer reflection)
    as defined in ITU-R P.1812-6 (Section 4.5)

        Input parameters:
        dtot         -   Great-circle path distance (km)
        dlt          -   interfering antenna horizon distance (km)
        dlr          -   Interfered-with antenna horizon distance (km)
        dct, dcr     -   Distance over land from the transmit and receive
                         antennas tothe coast along the great-circle interference path (km).
                         Set to zero for a terminal on a ship or sea platform
        dlm          -   the longest continuous inland section of the great-circle path (km)
        hts, hrs     -   Tx and Rx antenna heights aobe mean sea level amsl (m)
        hte, hre     -   Tx and Rx terminal effective heights for the ducting/layer reflection model (m)
        hm           -   The terrain roughness parameter (m)
        theta_t      -   Interfering antenna horizon elevation angle (mrad)
        theta_r      -   Interfered-with antenna horizon elevation angle (mrad)
        f            -   frequency expressed in GHz
        p            -   percentage of time
        omega        -   fraction of the total path over water
        ae           -   the median effective Earth radius (km)
        b0           -   the time percentage that the refractivity gradient (DELTA-N) exceeds 100 N-units/km in the first 100m of the lower atmosphere

        Output parameters:
        Lba    -   the basic transmission loss due to anomalous propagation
                (ducting and layer reflection)

        Example:
        Lba = tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, omega, b0)


        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    29SEP22     Ivica Stevanovic, OFCOM         Initial version


    """

    # empirical correction to account for the increasing attenuation with
    # wavelength inducted propagation (47a)

    Alf = 0

    if f < 0.5:
        Alf = 45.375 - 137.0 * f + 92.5 * f * f

    # site-shielding diffraction losses for the interfering and interfered-with
    # stations (48)

    theta_t2 = theta_t - 0.1 * dlt  # eq (48a)
    theta_r2 = theta_r - 0.1 * dlr

    Ast = 0
    Asr = 0

    if theta_t2 > 0:
        Ast = 20 * np.log10(1 + 0.361 * theta_t2 * np.sqrt(f * dlt)) + 0.264 * theta_t2 * f ** (1.0 / 3.0)

    if theta_r2 > 0:
        Asr = 20 * np.log10(1 + 0.361 * theta_r2 * np.sqrt(f * dlr)) + 0.264 * theta_r2 * f ** (1.0 / 3.0)

    # over-sea surface duct coupling correction for the interfering and
    # interfered-with stations (49) and (49a)

    Act = 0
    Acr = 0

    if dct <= 5:
        if dct <= dlt:
            if omega >= 0.75:
                Act = -3 * np.exp(-0.25 * dct * dct) * (1 + np.tanh(0.07 * (50 - hts)))

    if dcr <= 5:
        if dcr <= dlr:
            if omega >= 0.75:
                Acr = -3 * np.exp(-0.25 * dcr * dcr) * (1 + np.tanh(0.07 * (50 - hrs)))

    # specific attenuation (51)

    gamma_d = 5e-5 * ae * f ** (1.0 / 3.0)

    # angular distance (corrected where appropriate) (52-52a)

    theta_t1 = theta_t
    theta_r1 = theta_r

    if theta_t > 0.1 * dlt:
        theta_t1 = 0.1 * dlt

    if theta_r > 0.1 * dlr:
        theta_r1 = 0.1 * dlr

    theta1 = 1e3 * dtot / ae + theta_t1 + theta_r1

    dI = min(dtot - dlt - dlr, 40)  # eq (56a)

    mu3 = 1

    if hm > 10:
        mu3 = np.exp(-4.6e-5 * (hm - 10) * (43 + 6 * dI))  # eq (56)

    tau = 1 - np.exp(-(4.12e-4 * dlm**2.41))  # eq (3)

    epsilon = 3.5

    alpha = -0.6 - epsilon * 1e-9 * dtot ** (3.1) * tau  # eq (55a)

    if alpha < -3.4:
        alpha = -3.4

    # correction for path geometry:

    mu2 = (500 / ae * dtot**2 / (np.sqrt(hte) + np.sqrt(hre)) ** 2) ** alpha  # eq (55)

    if mu2 > 1:
        mu2 = 1

    beta = b0 * mu2 * mu3  # eq (54)

    # beta = max(beta, eps)      # to avoid division by zero

    Gamma = 1.076 / (2.0058 - np.log10(beta)) ** 1.012 * np.exp(-(9.51 - 4.8 * np.log10(beta) + 0.198 * (np.log10(beta)) ** 2) * 1e-6 * dtot ** (1.13))  # eq (53a)

    # time percentage variablity (cumulative distribution):

    Ap = -12 + (1.2 + 3.7e-3 * dtot) * np.log10(p / beta) + 12 * (p / beta) ** Gamma  # eq (53)

    # time percentage and angular-distance dependent losses within the
    # anomalous propagation mechanism

    Adp = gamma_d * theta1 + Ap  # eq (50)

    # total of fixed coupling losses (except for local clutter losses) between
    # the antennas and the anomalous propagation structure within the
    # atmosphere (47)

    Af = 102.45 + 20 * np.log10(f) + 20 * np.log10(dlt + dlr) + Alf + Ast + Asr + Act + Acr

    # total basic transmission loss occuring during periods of anomalaous
    # propagation (46)

    Lba = Af + Adp

    return Lba


def smooth_earth_heights(d, h, R, htg, hrg, ae, f):
    """
    smooth_earth_heights smooth-Earth effective antenna heights according to ITU-R P.1812-6
    hst_n, hsr_n, hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta_tot, pathtype = smooth_earth_heights(d, h, R, htg, hrg, ae, f)
    This function derives smooth-Earth effective antenna heights according to
    Sections 4 and 5 of the Attachment 1 to Annex 1 of ITU-R P.1812-6

    Input parameters:
    d         -   vector of terrain profile distances from Tx [0,dtot] (km)
    h         -   vector of terrain profile heigths amsl (m)
    R         -   vector of representative clutter heights (m)
    htg, hrg  -   Tx and Rx antenna heights above ground level (m)
    ae        -   median effective Earth's radius (c.f. Eq (6a))
    f         -   frequency (GHz)

    Output parameters:

    hst_n, hsr_n -   Not corrected Tx and Rx antenna heigts of the smooth-Earth surface amsl (m)
    hst, hsr     -   Tx and Rx antenna heigts of the smooth-Earth surface amsl (m)
    hstd, hsrd   -   Tx and Rx effective antenna heigts for the diffraction model (m)
    hte, hre     -   Tx and Rx terminal effective heights for the ducting/layer reflection model (m)
    hm           -   The terrain roughness parameter (m)
    dlt          -   interfering antenna horizon distance (km)
    dlr          -   Interfered-with antenna horizon distance (km)
    theta_t      -   Interfering antenna horizon elevation angle (mrad)
    theta_r      -   Interfered-with antenna horizon elevation angle (mrad)
    theta_tot    -   Angular distance (mrad)
    pathtype     -   1 = 'los', 2 = 'transhorizon'

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22    Ivica Stevanovic, OFCOM         First implementation in python
    """
    n = len(d)

    dtot = d[-1]

    # Tx and Rx antenna heights above mean sea level amsl (m)
    hts = h[0] + htg
    hrs = h[-1] + hrg

    g = h + R
    g[0] = h[0]
    g[-1] = h[-1]

    htc = hts
    hrc = hrs

    # Section 5.6.1 Deriving the smooth-Earth surface
    ii = np.arange(1,n)
    v1 = ((d[ii] - d[ii - 1]) * (h[ii] + h[ii - 1])).sum()   # Eq (85)
    v2 = ((d[ii] - d[ii - 1]) * (h[ii] * (2 * d[ii] + d[ii - 1]) + h[ii - 1] * (d[ii] + 2 * d[ii - 1]))).sum()  # Eq (86)

    hst = (2 * v1 * dtot - v2) / dtot**2  # Eq (87)
    hsr = (v2 - v1 * dtot) / dtot**2  # Eq (88)

    hst_n = hst
    hsr_n = hsr

    # Section 5.6.2 Smooth-surface heights for the diffraction model

    HH = h - (htc * (dtot - d) + hrc * d) / dtot  # Eq (89d)

    hobs = max(HH[1:-1])  # Eq (89a)

    alpha_obt = max(HH[1:-1] / d[1:-1])  # Eq (89b)

    alpha_obr = max(HH[1:-1] / (dtot - d[1:-1]))  # Eq (89c)

    # Calculate provisional values for the Tx and Rx smooth surface heights

    gt = alpha_obt / (alpha_obt + alpha_obr)  # Eq (90e)
    gr = alpha_obr / (alpha_obt + alpha_obr)  # Eq (90f)

    if hobs <= 0:
        hstp = hst  # Eq (90a)
        hsrp = hsr  # Eq (90b)
    else:
        hstp = hst - hobs * gt  # Eq (90c)
        hsrp = hsr - hobs * gr  # Eq (90d)

    # calculate the final values as required by the diffraction model

    if hstp >= h[0]:
        hstd = h[0]  # Eq (91a)
    else:
        hstd = hstp  # Eq (91b)

    if hsrp > h[-1]:
        hsrd = h[-1]  # Eq (91c)
    else:
        hsrd = hsrp  # Eq (91d)

    # Interfering antenna horizon elevation angle and distance

    ii = range(1, n - 1)

    theta = 1000 * np.arctan((h[ii] - hts) / (1000 * d[ii]) - d[ii] / (2 * ae))  # Eq (77)

    # theta(theta < 0) = 0  # condition below equation (152)

    theta_td = 1000 * np.arctan((hrs - hts) / (1000 * dtot) - dtot / (2 * ae))  # Eq (78)
    theta_rd = 1000 * np.arctan((hts - hrs) / (1000 * dtot) - dtot / (2 * ae))  # Eq (81)

    theta_max = max(theta)  # Eq (76)
    if theta_max > theta_td:  # Eq (150): test for the trans-horizon path
        pathtype = 2  # transhorizon
    else:
        pathtype = 1  # los

    theta_t = max(theta_max, theta_td)  # Eq (79)

    if pathtype == 2:  # transhorizon
        (kindex,) = np.where(theta == theta_max)

        lt = kindex[0] + 1  # in order to map back to path d indices, as theta takes path indices 2 to n-1,

        dlt = d[lt]  # Eq (80)

        # Interfered-with antenna horizon elevation angle and distance

        theta = 1000 * np.arctan((h[ii] - hrs) / (1000 * (dtot - d[ii])) - (dtot - d[ii]) / (2 * ae))  # Eq (82a)

        theta_r = max(theta)

        (kindex,) = np.where(theta == theta_r)

        lr = kindex[-1] + 1  # in order to map back to path d indices, as theta takes path indices 2 to n-1,

        dlr = dtot - d[lr]  # Eq (83)

    else:  # pathtype == 1 (LoS)
        theta_r = theta_rd  # Eq (81)

        ii = range(1, n - 1)

        # speed of light as per ITU.R P.2001
        lam = 0.2998 / f
        Ce = 1.0 / ae  # Section 4.3.1 supposing median effective Earth radius

        nu = (h[ii] + 500 * Ce * d[ii] * (dtot - d[ii]) - (hts * (dtot - d[ii]) + hrs * d[ii]) / dtot) * np.sqrt(0.002 * dtot / (lam * d[ii] * (dtot - d[ii])))  # Eq (81)
        numax = max(nu)

        (kindex,) = np.where(nu == numax)
        lt = kindex[-1] + 1  # in order to map back to path d indices, as theta takes path indices 2 to n-1,
        dlt = d[lt]  # Eq (80)
        dlr = dtot - dlt  # Eq  (83a)
        lr = lt

    # Angular distance

    theta_tot = 1e3 * dtot / ae + theta_t + theta_r  # Eq (84)

    # Section 5.6.3 Ducting/layer-reflection model

    # Calculate the smooth-Earth heights at transmitter and receiver as
    # required for the roughness factor

    hst = min(hst, h[0])  # Eq (92a)
    hsr = min(hsr, h[-1])  # Eq (92b)

    # Slope of the smooth-Earth surface

    m = (hsr - hst) / dtot  # Eq (93)

    # The terminal effective heigts for the ducting/layer-reflection model

    hte = htg + h[0] - hst  # Eq (94a)
    hre = hrg + h[-1] - hsr  # Eq (94b)

    ii = range(lt, lr + 1)

    hm = max(h[ii] - (hst + m * d[ii]))  # Eq (95)

    return hst_n, hsr_n, hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta_tot, pathtype


def pl_los(d, hts, hrs, f, p, b0, dlt, dlr):
    """
    pl_los Line-of-sight transmission loss according to ITU-R P.1812-6
    This function computes line-of-sight transmission loss (including short-term effects)
    as defined in ITU-R P.1812-67.

    Input parameters:
    d       -   Great-circle path distance (km)
    f       -   Frequency (GHz)
    hts     -   Tx antenna height above sea level (masl)
    hrs     -   Rx antenna height above sea level (masl)
    p       -   Required time percentage(s) for which the calculated basic
                transmission loss is not exceeded (%)
    b0      -   Point incidence of anomalous propagation for the path
                central location (%)
    w       -   Fraction of the total path over water (%)
    temp    -   Temperature (degrees C)
    press   -   Dry air pressure (hPa)
    dlt     -   For a transhorizon path, distance from the transmit antenna to
                its horizon (km). For a LoS path, each is set to the distance
                from the terminal to the profile point identified as the Bullington
                point in the diffraction method for 50% time
    dlr     -   For a transhorizon path, distance from the receive antenna to
                its horizon (km). The same note as for dlt applies here.

    Output parameters:
    Lbfs    -   Basic transmission loss due to free-space propagation
    Lb0p    -   Basic transmission loss not exceeded for time percentage, p%, due to LoS propagation
    Lb0b    -   Basic transmission loss not exceedd for time percentage, b0%, due to LoS propagation

    Example:
    Lbfs, Lb0p, Lb0b = pl_los(d, f, p, b0, dlt, dlr)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22     Ivica Stevanovic, OFCOM         First implementation

    """

    # Basic transmission loss due to free-space propagation
    dfs2 = d**2 + ((hts - hrs) / 1000.0) ** 2  # (8a)

    Lbfs = 92.4 + 20.0 * np.log10(f) + 10.0 * np.log10(dfs2)  # (8)

    # Corrections for multipath and focusing effects at p and b0
    Esp = 2.6 * (1 - np.exp(-0.1 * (dlt + dlr))) * np.log10(p / 50)  # (9a)
    Esb = 2.6 * (1 - np.exp(-0.1 * (dlt + dlr))) * np.log10(b0 / 50)  # (9b)

    # Basic transmission loss not exceeded for time percentage p# due to
    # LoS propagation
    Lb0p = Lbfs + Esp  # (10)

    # Basic transmission loss not exceeded for time percentage b0% due to
    # LoS propagation
    Lb0b = Lbfs + Esb  # (11)

    return Lbfs, Lb0p, Lb0b


def path_fraction(d, zone, zone_r):
    """
    path_fraction Path fraction belonging to a given zone_r
    omega = path_fraction(d, zone, zone_r)
    This function computes the path fraction belonging to a given zone_r
    of the great-circle path (km)

    Input arguments:
    d       -   vector of distances in the path profile
    zone    -   vector of zones in the path profile
    zone_r  -   reference zone for which the fraction is computed

    Output arguments:
    omega   -   path fraction belonging to the given zone_r

    Example:
    omega = path_fraction(d, zone, zone_r)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    17MAR22     Ivica Stevanovic, OFCOM         First implementation in python
    v1    11NOV22     Ivica Stevanovic, OFCOM         Corrected a bug in the second if clause (suggested by Martin-Pierre Lussier @mplussier)
    """
    dm = 0

    start, stop = find_intervals((zone == zone_r))

    n = len(start)

    for i in range(0, n):
        delta = 0
        if d[stop[i]] < d[-1]:
            delta = delta + (d[stop[i] + 1] - d[stop[i]]) / 2.0

        if d[start[i]] > 0:
            delta = delta + (d[start[i]] - d[start[i] - 1]) / 2.0

        dm = dm + d[stop[i]] - d[start[i]] + delta

    omega = dm / (d[-1] - d[0])

    return omega


def longest_cont_dist(d, zone, zone_r):
    """
    longest_cont_dist Longest continuous path belonging to the zone_r
    dm = longest_cont_dist(d, zone, zone_r)
    This function computes the longest continuous section of the
    great-circle path (km) for a given zone_r

    Input arguments:
    d       -   vector of distances in the path profile
    zone    -   vector of zones in the path profile
    zone_r  -   reference zone for which the longest continuous section
                is computed

    Output arguments:
    dm      -   the longest continuous section of the great-circle path (km) for a given zone_r

    Example:
    dm = longest_cont_dist(d, zone, zone_r)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22     Ivica Stevanovic, OFCOM         Initial version
    v1    11NOV22     Ivica Stevanovic, OFCOM         Corrected a bug in the second if clause for delta (suggested by Martin-Pierre Lussier @mplussier)

    """
    dm = 0

    if zone_r == 34:  # inland + coastal land
        start, stop = find_intervals((zone == 3) + (zone == 4))
    else:
        start, stop = find_intervals((zone == zone_r))

    n = len(start)

    for i in range(0, n):
        delta = 0
        if d[stop[i]] < d[-1]:
            delta = delta + (d[stop[i] + 1] - d[stop[i]]) / 2.0

        if d[start[i]] > 0:
            delta = delta + (d[start[i]] - d[start[i] - 1]) / 2.0

        dm = max(d[stop[i]] - d[start[i]] + delta, dm)

    return dm


def inv_cum_norm(x):
    """
    inv_cum_norm approximation to the inverse cummulative normal distribution
    I = inv_cum_norm( x )
    This function implements an approximation to the inverse cummulative
    normal distribution function for 0< x < 1 as defined in Attachment 2 to
    Annex 1 of the ITU-R P.1812-6

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22     Ivica Stevanovic, OFCOM         Initial version
    """
    if x < 0.000001:
        x = 0.000001

    if x > 0.999999:
        x = 0.999999

    if x <= 0.5:
        I = T(x) - C(x)  # (96a)
    else:
        I = -(T(1 - x) - C(1 - x))  # (96b)

    return I


def T(y):
    return np.sqrt(-2.0 * np.log(y))  # eq (97a)


def C(z):  # eq (97)
    C0 = 2.515516698
    C1 = 0.802853
    C2 = 0.010328
    D1 = 1.432788
    D2 = 0.189269
    D3 = 0.001308

    return (((C2 * T(z) + C1) * T(z)) + C0) / (((D3 * T(z) + D2) * T(z) + D1) * T(z) + 1)  # eq (97b)


def great_circle_path(Phire, Phite, Phirn, Phitn, Re, dpnt):
    """
    great_circle_path Great-circle path calculations according to Attachment H
    This function computes the great-circle intermediate points on the
    radio path as defined in ITU-R P.2001-4 Attachment H

        Input parameters:
        Phire   -   Receiver longitude, positive to east (deg)
        Phite   -   Transmitter longitude, positive to east (deg)
        Phirn   -   Receiver latitude, positive to north (deg)
        Phitn   -   Transmitter latitude, positive to north (deg)
        Re      -   Average Earth radius (km)
        dpnt    -   Distance from the transmitter to the intermediate point (km)

        Output parameters:
        Phipnte -   Longitude of the intermediate point (deg)
        Phipntn -   Latitude of the intermediate point (deg)
        Bt2r    -   Bearing of the great-circle path from Tx towards the Rx (deg)
        dgc     -   Great-circle path length (km)

        Example:
        [Bt2r, Phipnte, Phipntn, dgc] = great_circle_path(Phire, Phite, Phirn, Phitn, Re, dpnt)

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    05SEP22     Ivica Stevanovic, OFCOM         Initial version
    """
    ## H.2 Path length and bearing

    # Difference (deg) in longitude between the terminals (H.2.1)

    Dlon = Phire - Phite

    # Calculate quantity r (H.2.2)

    r = sind(Phitn) * sind(Phirn) + cosd(Phitn) * cosd(Phirn) * cosd(Dlon)

    # Calculate the path length as the angle subtended at the center of
    # average-radius Earth (H.2.3)

    Phid = np.arccos(r)  # radians

    # Calculate the great-circle path length (H.2.4)

    dgc = Phid * Re  # km

    # Calculate the quantity x1 (H.2.5a)

    x1 = sind(Phirn) - r * sind(Phitn)

    # Calculate the quantity y1 (H.2.5b)

    y1 = cosd(Phitn) * cosd(Phirn) * sind(Dlon)

    # Calculate the bearing of the great-circle path for Tx to Rx (H.2.6)

    if abs(x1) < 1e-9 and abs(y1) < 1e-9:
        Bt2r = Phire
    else:
        Bt2r = atan2d(y1, x1)

    ## H.3 Calculation of intermediate path point

    # Calculate the distance to the point as the angle subtended at the center
    # of average-radius Earth (H.3.1)

    Phipnt = dpnt / Re  # radians

    # Calculate quantity s (H.3.2)

    s = sind(Phitn) * np.cos(Phipnt) + cosd(Phitn) * np.sin(Phipnt) * cosd(Bt2r)

    # The latitude of the intermediate point is now given by (H.3.3)

    Phipntn = asind(s)  # degs

    # Calculate the quantity x2 (H.3.4a)

    x2 = np.cos(Phipnt) - s * sind(Phitn)

    # Calculate the quantity y2 (H.3.4b)

    y2 = cosd(Phitn) * np.sin(Phipnt) * sind(Bt2r)

    # Calculate the longitude of the intermediate point Phipnte (H.3.5)

    if x2 < 1e-9 and y2 < 1e-9:
        Phipnte = Bt2r
    else:
        Phipnte = Phite + atan2d(y2, x2)

    return Phipnte, Phipntn, Bt2r, dgc


def isempty(x):
    if np.size(x) == 0:
        return True
    else:
        return False


def issorted(a):
    if np.all(np.diff(a) >= 0):
        return True
    else:
        return False


def sind(x):
    return np.sin(x * np.pi / 180.0)


def cosd(x):
    return np.cos(x * np.pi / 180.0)


def asind(x):
    return np.arcsin(x) * 180.0 / np.pi


def atan2d(y, x):
    return np.arctan2(y, x) * 180.0 / np.pi


def find_intervals(series):
    """
    find_intervals Find all intervals with consecutive 1's
    [k1, k2] = find_intervals(series)
    This function finds all 1's intervals, namely, the indices when the
    intervals start and where they end

    For example, for the input indices
            0 0 1 1 1 1 0 0 0 1 1 0 0
    this function will give back
        k1 = 3, 10
        k2 = 6, 11

    Input arguments:
    indices -   vector containing zeros and ones

    Output arguments:
    k1      -   vector of start-indices of the found intervals
    k2      -   vector of end-indices of the found intervals

    Example:
    [k1, k2] = find_intervals(indices)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    18MAR22     Ivica Stevanovic, OFCOM         First implementation in python

    """
    k1 = []
    k2 = []
    series_int = 1 * series

    if max(series_int) == 1:
        (k1,) = np.where(np.diff(np.append(0, series_int)) == 1)
        (k2,) = np.where(np.diff(np.append(series_int, 0)) == -1)

    return k1, k2


def earth_rad_eff(DN):
    """
    earth_rad_eff Median value of the effective Earth radius
    [ae, ab] = earth_rad_eff(DN)
    This function computes the median value of the effective earth
    radius, and the effective Earth radius exceeded for beta0% of time
    as defined in ITU-R P.1812-6.

    Input arguments:
    DN      -   the average radio refractivity lapse-rate through the
                lowest 1 km of the atmosphere (N-units/km)

    Output arguments:
    ae      -   the median effective Earth radius (km)
    ab      -   the effective Earth radius exceeded for beta0 % of time

    Example:
    ae, ab = earth_rad_eff(DN)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22   Ivica Stevanovic, OFCOM         Initial implementation
    """

    k50 = 157 / (157 - DN)  # (6)

    ae = 6371 * k50  # (7a)

    kbeta = 3

    ab = 6371 * kbeta  # (7b)

    return ae, ab


def dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f):
    """
    dl_se_ft_inner The inner routine of the first-term spherical diffraction loss
    This function computes the first-term part of Spherical-Earth diffraction
    loss exceeded for p% time for antenna heights
    as defined in Sec. 4.3.3 of the ITU-R P.1812-4, equations (29-36)

        Input parameters:
        epsr    -   Relative permittivity
        sigma   -   Conductivity (S/m)
        d       -   Great-circle path distance (km)
        hte     -   Effective height of interfering antenna (m)
        hre     -   Effective height of interfered-with antenna (m)
        adft    -   effective Earth radius (km)
        f       -   frequency (GHz)

        Output parameters:
        Ldft   -   The first-term spherical-Earth diffraction loss not exceeded for p% time
                    implementing equations (29-36), Ldft(1) is for horizontal
                    and Ldft(2) for the vertical polarization

        Example:
        Ldft = dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f)

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    29SEP22     Ivica Stevanovic, OFCOM         Initial implementation
    """

    # Normalized factor for surface admittance for horizontal (1) and vertical
    # (2) polarizations

    K = np.zeros(2)

    K[0] = 0.036 * (adft * f) ** (-1.0 / 3.0) * ((epsr - 1) ** 2 + (18 * sigma / f) ** 2.0) ** (-1.0 / 4.0)  # Eq (29a)

    K[1] = K[0] * (epsr**2 + (18 * sigma / f) ** 2) ** (1.0 / 2.0)  # Eq (29b)

    # Earth ground/polarization parameter

    beta_dft = (1 + 1.6 * K**2 + 0.67 * K**4) / (1 + 4.5 * K**2 + 1.53 * K**4)  # Eq (30)

    # Normalized distance

    X = 21.88 * beta_dft * (f / adft**2) ** (1.0 / 3.0) * d  # Eq (31)

    # Normalized transmitter and receiver heights

    Yt = 0.9575 * beta_dft * (f**2 / adft) ** (1 / 3) * hte  # Eq (32a)

    Yr = 0.9575 * beta_dft * (f**2 / adft) ** (1 / 3) * hre  # Eq (32b)

    # Calculate the distance term given by:

    Fx = np.zeros(2)

    for ii in range(0, 2):
        if X[ii] >= 1.6:
            Fx[ii] = 11 + 10 * np.log10(X[ii]) - 17.6 * X[ii]
        else:
            Fx[ii] = -20 * np.log10(X[ii]) - 5.6488 * (X[ii]) ** 1.425  # Eq (33)

    Bt = beta_dft * Yt  # Eq (35)

    Br = beta_dft * Yr  # Eq (35)

    GYt = np.zeros(2)
    GYr = np.zeros(2)

    for ii in range(0, 2):
        if Bt[ii] > 2:
            GYt[ii] = 17.6 * (Bt[ii] - 1.1) ** 0.5 - 5 * np.log10(Bt[ii] - 1.1) - 8
        else:
            GYt[ii] = 20 * np.log10(Bt[ii] + 0.1 * Bt[ii] ** 3)

        if Br[ii] > 2:
            GYr[ii] = 17.6 * (Br[ii] - 1.1) ** 0.5 - 5 * np.log10(Br[ii] - 1.1) - 8
        else:
            GYr[ii] = 20 * np.log10(Br[ii] + 0.1 * Br[ii] ** 3)

        if GYr[ii] < 2 + 20 * np.log10(K[ii]):
            GYr[ii] = 2 + 20 * np.log10(K[ii])

        if GYt[ii] < 2 + 20 * np.log10(K[ii]):
            GYt[ii] = 2 + 20 * np.log10(K[ii])

    Ldft = -Fx - GYt - GYr  # Eq (36)

    return Ldft


def dl_se_ft(d, hte, hre, adft, f, omega):
    """
    dl_se_ft First-term part of spherical-Earth diffraction according to ITU-R P.1812-6
    This function computes the first-term part of Spherical-Earth diffraction
    loss exceeded for p% time for antenna heights
    as defined in Sec. 4.3.3 of the ITU-R P.1812-6

    Input parameters:
    d       -   Great-circle path distance (km)
    hte     -   Effective height of interfering antenna (m)
    hre     -   Effective height of interfered-with antenna (m)
    adft    -   effective Earth radius (km)
    f       -   frequency (GHz)
    omega   -   fraction of the path over sea

    Output parameters:
    Ldft   -   The first-term spherical-Earth diffraction loss not exceeded for p% time
                Ldft(1) is for the horizontal polarization
                Ldft(2) is for the vertical polarization

    Example:
    Ldft = dl_se_ft(d, hte, hre, adft, f, omega)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22     Ivica Stevanovic, OFCOM         Initial implementation
    """

    # First-term part of the spherical-Earth diffraction loss over land


    Ldft_land = dl_se_ft_inner(22, 0.003, d, hte, hre, adft, f)

    # First-term part of the spherical-Earth diffraction loss over sea

    Ldft_sea = dl_se_ft_inner(80, 5, d, hte, hre, adft, f)

    # First-term spherical diffraction loss

    Ldft = omega * Ldft_sea + (1 - omega) * Ldft_land  # Eq (28)

    return Ldft


def dl_se(d, hte, hre, ap, f, omega):
    """
    dl_se spherical-Earth diffraction loss exceeded for p% time according to ITU-R P.1812-6
    This function computes the Spherical-Earth diffraction loss exceeded
    for p% time for antenna heights hte and hre (m)
    as defined in Sec. 4.3.2 of the ITU-R P.1812-6

    Input parameters:
    d       -   Great-circle path distance (km)
    hte     -   Effective height of interfering antenna (m)
    hre     -   Effective height of interfered-with antenna (m)
    ap      -   the effective Earth radius in kilometers
    f       -   frequency expressed in GHz
    omega   -   the fraction of the path over sea

    Output parameters:
    Ldsph   -   The spherical-Earth diffraction loss not exceeded for p% time
                Ldsph(1) is for the horizontal polarization
                Ldsph(2) is for the vertical polarization

    Example:
    Ldsph = dl_se(d, hte, hre, ap, f, omega)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22     Ivica Stevanovic, OFCOM         Initial version
    """

    # Wavelength in meters
    # speed of light as per ITU.R P.2001
    lam = 0.2998 / f

    # Calculate the marginal LoS distance for a smooth path

    dlos = np.sqrt(2.0 * ap) * (np.sqrt(0.001 * hte) + np.sqrt(0.001 * hre))  # Eq (22)

    if d >= dlos:
        # calculate diffraction loss Ldft using the method in Sec. Sec. 4.3.3 for
        # adft = ap and set Ldsph to Ldft

        Ldsph = dl_se_ft(d, hte, hre, ap, f, omega)

        return Ldsph
    else:
        # calculate the smallest clearance between the curved-Earth path and
        # the ray between the antennas, hse

        c = (hte - hre) / (hte + hre)  # Eq (24d)
        m = 250 * d * d / (ap * (hte + hre))  # Eq (24e)

        b = 2 * np.sqrt((m + 1.0) / (3.0 * m)) * np.cos(np.pi / 3 + 1.0 / 3.0 * np.arccos(3 * c / 2.0 * np.sqrt(3.0 * m / (m + 1.0) ** 3)))  # Eq (24c)

        dse1 = d / 2.0 * (1.0 + b)  # Eq (24a)
        dse2 = d - dse1  # Eq (24b)

        hse = (hte - 500 * dse1 * dse1 / ap) * dse2 + (hre - 500 * dse2 * dse2 / ap) * dse1
        hse = hse / d  # Eq (23)

        # Calculate the required clearance for zero diffraction loss

        hreq = 17.456 * np.sqrt(dse1 * dse2 * lam / d)  # Eq (25)

        if hse > hreq:
            Ldsph = np.zeros(2)
            return Ldsph
        else:
            # calculate the modified effective Earth radius aem, which gives
            # marginal LoS at distance d

            aem = 500 * (d / (np.sqrt(hte) + np.sqrt(hre))) ** 2  # Eq (26)

            # Use the method in Sec. 4.2.2.1 for adft ) aem to obtain Ldft

            Ldft = dl_se_ft(d, hte, hre, aem, f, omega)
            Ldft[Ldft < 0.0] = 0.0
            Ldsph = (1 - hse / hreq) * Ldft  # Eq (27)

    return Ldsph


def dl_p(d, g, hts, hrs, hstd, hsrd, f, omega, p, b0, DN, flag4):
    """
    dl_p Diffraction loss model not exceeded for p% of time according to P.1812-6

    This function computes the diffraction loss not exceeded for p% of time
    as defined in ITU-R P.1812-6 (Section 4.3-5) and Attachment 4 to Annex 1

        Input parameters:
        d       -   vector of distances di of the i-th profile point (km)
        g       -   vector gi of heights of the i-th profile point (meters
                    above mean sea level) + Representative clutter height.
                    Both vectors g and d contain n+1 profile points
        hts     -   transmitter antenna height in meters above sea level (i=0)
        hrs     -   receiver antenna height in meters above sea level (i=n)
        hstd    -   Effective height of interfering antenna (m amsl) c.f. 5.1.6.3
        hsrd    -   Effective height of interfered-with antenna (m amsl) c.f. 5.1.6.3
        f       -   frequency expressed in GHz
        omega   -   the fraction of the path over sea
        p       -   percentage of time
        b0      -   the time percentage that the refractivity gradient (DELTA-N) exceeds 100 N-units/km in the first 100m of the lower atmosphere
        DN      -   the average radio-refractivity lapse-rate through the
                    lowest 1 km of the atmosphere. Note that DN is positive
                    quantity in this procedure
        flag4   -   Set to 1 if the alternative method is used to calculate Lbulls
                    without using terrain profile analysis (Attachment 4 to Annex 1)

        Output parameters:
        Ldp      -   diffraction loss for the general path not exceeded for p % of the time
                     according to Section 4.3.4 of ITU-R P.1812-6.
                     Ldp(1) is for the horizontal polarization
                     Ldp(2) is for the vertical polarization
        Ldb      -   diffraction loss for p = beta_0%
        Ld50     -   diffraction loss for p = 50%
        Lbulla50 -   Bullington diffraction (4.3.1) for actual terrain profile g and antenna heights
        Lbulls50 -   Bullington diffraction (4.3.1) with all profile heights g set to zero and modified antenna heights
        Ldshp50  -   Spherical diffraction (4.3.2) for the actual path d and modified antenna heights

        Example:
        Ldp, Ldb, Ld50, Lbulla50, Lbulls50, Ldsph50 = dl_p( d, h, hts, hrs, hstd, hsrd, ap, f, omega, p, b0, DN, flag4 )


        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    29SEP22     Ivica Stevanovic, OFCOM         Initial version

    """

    # Use the method in 4.3.4 to calculate diffraction loss Ld for median effective
    # Earth radius ap = ae as given by equation (7a). Set median diffraction
    # loss to Ldp50

    ae, ab = earth_rad_eff(DN)

    ap = ae

    Ld50, Lbulla50, Lbulls50, Ldsph50 = dl_delta_bull(d, g, hts, hrs, hstd, hsrd, ap, f, omega, flag4)

    if p == 50:
        Ldp = Ld50
        ap = ab
        Ldb, Lbulla50, Lbulls50, Ldsph50 = dl_delta_bull(d, g, hts, hrs, hstd, hsrd, ap, f, omega, flag4)
        return Ldp, Ldb, Ld50, Lbulla50, Lbulls50, Ldsph50

    if p < 50:
        # Use the method in 4.3.4  to calculate diffraction loss Ld for effective
        # Earth radius ap = abeta, as given in equation (7b). Set diffraction loss
        # not exceeded for beta0# time Ldb = Ld

        ap = ab

        Ldb, Lbulla50, Lbulls50, Ldsph50 = dl_delta_bull(d, g, hts, hrs, hstd, hsrd, ap, f, omega, flag4)

        # Compute the interpolation factor Fi

        if p > b0:
            Fi = inv_cum_norm(p / 100) / inv_cum_norm(b0 / 100)  # eq (40a)

        else:
            Fi = 1

        # The diffraction loss Ldp not exceeded for p% of time is now given by

        Ldp = Ld50 + Fi * (Ldb - Ld50)  # eq (41)

    return Ldp, Ldb, Ld50, Lbulla50, Lbulls50, Ldsph50


def dl_delta_bull(d, g, hts, hrs, hstd, hsrd, ap, f, omega, flag4):
    """
    dl_delta_bull Complete 'delta-Bullington' diffraction loss model P.1812-6

    This function computes the complete 'delta-Bullington' diffraction loss
    as defined in ITU-R P.1812.6 (Section 4.3.4)

        Input parameters:
        d       -   vector of distances di of the i-th profile point (km)
        g       -   vector of heights hi of the i-th profile point (meters
                    above mean sea level) + Representative clutter height.
                    Both vectors d, g contain n+1 profile points
        hts     -   transmitter antenna height in meters above sea level (i=0)
        hrs     -   receiver antenna height in meters above sea level (i=n)
        hstd    -   Effective height of interfering antenna (m amsl) c.f. 5.1.6.3
        hsrd    -   Effective height of interfered-with antenna (m amsl) c.f. 5.1.6.3
        ap      -   the effective Earth radius in kilometers
        f       -   frequency expressed in GHz
        omega   -   the fraction of the path over sea
        flag4   -   Set to 1 if the alternative method is used to calculate Lbulls
                    without using terrain profile analysis (Attachment 4 to Annex 1)

        Output parameters:
        Ld     -   diffraction loss for the general path according to
                   Section 4.3.3 of ITU-R P.1812-4.
                   Ld(1) is for the horizontal polarization
                   Ld(2) is for the vertical polarization
        Lbulla -   Bullington diffraction (4.3.1) for actual terrain profile g and antenna heights
        Lbulls -   Bullington diffraction (4.3.1) with all profile heights g set to zero and modified antenna heights
        Ldshp  -   Spherical diffraction (4.3.2) for the actual path d and modified antenna heights

        Example:
        Ld, Lbulla, Lbulls, Ldsph = dl_delta_bull( d, g, hts, hrs, hstd, hsrd, ap, f, omega, flag4)


        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    29SEP22     Ivica Stevanovic, OFCOM         Initial version
    """

    # Use the method in 4.3.1 for the actual terrain profile and antenna
    # heights. Set the resulting Bullington diffraction loss for the actual
    # path to Lbulla

    Lbulla = dl_bull(d, g, hts, hrs, ap, f)

    # Use the method in 4.3.1 for a second time, with all profile heights hi
    # set to zero and modified antenna heights given by

    hts1 = hts - hstd  # eq (37a)
    hrs1 = hrs - hsrd  # eq (7b)
    h1 = np.zeros(g.shape)
    dtot = d[-1] - d[0]

    # where hstd and hsrd are given in 5.6.2 of Attachment 1. Set the
    # resulting Bullington diffraction loss for this smooth path to Lbulls

    if flag4 == 1:
        # compute the spherical earth diffraction Lbuls using an
        # alternative method w/o terrain profile analysis
        # as defined in Attachment 4 to Annex 1 of ITU-R P.1812-6

        Lbulls = dl_bull_att4(dtot, hts1, hrs1, ap, f)
    else:
        # Compute Lbuls using §4.3.1

        Lbulls = dl_bull(d, h1, hts1, hrs1, ap, f)

    # Use the method in 4.3.2 to calculate the spherical-Earth diffraction loss
    # for the actual path length (dtot) with

    hte = hts1  # eq (38a)
    hre = hrs1  # eq (38b)

    Ldsph = dl_se(dtot, hte, hre, ap, f, omega)

    # Diffraction loss for the general paht is now given by

    Ld = np.zeros(2)

    Ld[0] = Lbulla + max(Ldsph[0] - Lbulls, 0)  # eq (39)
    Ld[1] = Lbulla + max(Ldsph[1] - Lbulls, 0)  # eq (39)

    return Ld, Lbulla, Lbulls, Ldsph


def dl_bull(d, g, hts, hrs, ap, f):
    """
    dl_bull Bullington part of the diffraction loss according to P.1812-6
    This function computes the Bullington part of the diffraction loss
    as defined in ITU-R P.1812-6 in 4.3.1

        Input parameters:
        d       -   vector of distances di of the i-th profile point (km)
        g       -   vector of heights hi of the i-th profile point (meters
                    above mean sea level) + representative clutter height
                    Both vectors d and g contain n+1 profile points
        hts     -   transmitter antenna height in meters above sea level (i=0)
        hrs     -   receiver antenna height in meters above sea level (i=n)
        ap      -   the effective earth radius in kilometers
        f       -   frequency expressed in GHz

        Output parameters:
        Lbull   -   Bullington diffraction loss for a given path

        Example:
        Lbull = dl_bull(d, g, hts, hrs, ap, f)

        Rev   Date        Author                          Description
        -------------------------------------------------------------------------------
        v0    29SEP22     Ivica Stevanovic, OFCOM         First implementation in matlab
    """

    # Effective Earth curvature Ce (km^-1)

    Ce = 1.0 / ap

    # Wavelength in meters
    # speed of light as per ITU.R P.2001
    lam = 0.2998 / f

    # Complete path length

    dtot = d[-1] - d[0]

    # Find the intermediate profile point with the highest slope of the line
    # from the transmitter to the point

    di = d[1:-1]
    gi = g[1:-1]

    Stim = max((gi + 500 * Ce * di * (dtot - di) - hts) / di)  # Eq (13)

    # Calculate the slope of the line from transmitter to receiver assuming a
    # LoS path

    Str = (hrs - hts) / dtot  # Eq (14)

    if Stim < Str:  # Case 1, Path is LoS
        # Find the intermediate profile point with the highest diffraction
        # parameter nu:

        numax = max((gi + 500 * Ce * di * (dtot - di) - (hts * (dtot - di) + hrs * di) / dtot) * np.sqrt(0.002 * dtot / (lam * di * (dtot - di))))  # Eq (15)

        Luc = 0
        if numax > -0.78:
            Luc = 6.9 + 20 * np.log10(np.sqrt((numax - 0.1) ** 2 + 1) + numax - 0.1)  # Eq (12), (16)

    else:
        # Path is transhorizon

        # Find the intermediate profile point with the highest slope of the
        # line from the receiver to the point

        Srim = max((gi + 500 * Ce * di * (dtot - di) - hrs) / (dtot - di))  # Eq (17)

        # Calculate the distance of the Bullington point from the transmitter:

        dbp = (hrs - hts + Srim * dtot) / (Stim + Srim)  # Eq (18)

        # Calculate the diffraction parameter, nub, for the Bullington point

        nub = (hts + Stim * dbp - (hts * (dtot - dbp) + hrs * dbp) / dtot) * np.sqrt(0.002 * dtot / (lam * dbp * (dtot - dbp)))  # Eq (20)

        # The knife-edge loss for the Bullington point is given by

        Luc = 0
        if nub > -0.78:
            Luc = 6.9 + 20 * np.log10(np.sqrt((nub - 0.1) ** 2 + 1) + nub - 0.1)  # Eq (12), (20)

    # For Luc calculated using either (16) or (20), Bullington diffraction loss
    # for the path is given by

    Lbull = Luc + (1 - np.exp(-Luc / 6.0)) * (10 + 0.02 * dtot)  # Eq (21)
    return Lbull


def dl_bull_att4(dtot, hte, hre, ap, f):
    """dl_bull_att4 Bullington part of the diffraction loss for smooth path according to Attachment 4 to Annex 1 of P.1812-6
    %   This function computes the spherical earth diffraction Lbuls using an
    %   alternative method w/o terrain profile analysis
    %   as defined in Attachment 4 to Annex 1 of ITU-R P.1812-5
    %
    %     Input parameters:
    %     dtot    -   Great-circle path distance (km)
    %     hte     -   Effective height of interfering antenna (m)
    %     hre     -   Effective height of interfered-with antenna (m)
    %     ap      -   the effective earth radius in kilometers
    %     f       -   frequency expressed in GHz
    %
    %     Output parameters:
    %     Lbulls   -   Bullington diffraction loss for a given path
    %
    %     Example:
    %     Lbulls = dl_bull_att4(d, hte, hre, ap, f)
    %
    %     Rev   Date        Author                          Description
    %     -------------------------------------------------------------------------------
    %     v1    28JUL20     Ivica Stevanovic, OFCOM         First implementation for P.1812-5
    """

    # Effective Earth curvature Ce (km^-1)

    Ce = 1.0 / ap

    # Wavelength in meters
    # speed of light as per ITU.R P.2001

    lam = 0.2998 / f

    # Calculate the marginal LoS distance for a smooth path

    dlos = np.sqrt(2.0 * ap) * (np.sqrt(0.001 * hte) + np.sqrt(0.001 * hre))  # Eq (22)

    if dtot < dlos:  # LoS
        # calculate the smallest clearance between the curved-Earth path and
        # the ray between the antennas, hse

        c = (hte - hre) / (hte + hre)  # Eq (24d)
        m = 250 * dtot * dtot / (ap * (hte + hre))  # Eq (24e)

        b = 2 * np.sqrt((m + 1.0) / (3.0 * m)) * np.cos(np.pi / 3.0 + 1.0 / 3.0 * np.arccos(3.0 * c / 2.0 * np.sqrt(3.0 * m / ((m + 1.0) ** 3))))  # Eq (24c)

        dse1 = dtot / 2.0 * (1.0 + b)  # Eq (24a)
        dse2 = dtot - dse1  # Eq (24b)

        hse = (hte - 500 * dse1 * dse1 / ap) * dse2 + (hre - 500 * dse2 * dse2 / ap) * dse1
        hse = hse / dtot  # Eq (23)

        # calculate the difraction parameter for the smallest clearance height hse
        # between the curved-Earth path and the ray between the antennas with the
        # distance dse1

        numax = -hse * np.sqrt(0.002 * dtot / (lam * dse1 * (dtot - dse1)))  # Eq (105)

        Lus = 0
        if numax > -0.78:
            Lus = 6.9 + 20 * np.log10(np.sqrt((numax - 0.1) ** 2 + 1) + numax - 0.1)  # Eq (12), (106)

    else:  # d>=dlos, NLoS
        # Find the highest slope of the line from the transmitter antenna to the curved-Earth path.

        Stm = 500 * Ce * dtot - 2 * np.sqrt(500.0 * Ce * hte)  # Eq (107)

        # find the highest slope of the line from the receiver antenna to the curved-Earth path

        Srm = 500 * Ce * dtot - 2 * np.sqrt(500.0 * Ce * hre)  # Eq (108)

        # Use these two slopes to calculate the Bullington point as:

        ds = (hre - hte + Srm * dtot) / (Stm + Srm)
        # Eq (109)

        # Calculate the diffraction parameter nus for the Bullington point:

        nus = hte + Stm * ds - (hte * (dtot - ds) + hre * ds) / dtot
        nus = nus * np.sqrt(0.002 * dtot / (lam * ds * (dtot - ds)))  # Eq (110)

        Lus = 0
        if nus > -0.78:
            Lus = 6.9 + 20 * np.log10(np.sqrt((nus - 0.1) ** 2 + 1) + nus - 0.1)  # Eq (12), (111)

    # For Luc calculated using either (106) or (111), Bullington diffraction loss
    # for the path is given by

    Lbulls = Lus + (1 - np.exp(-Lus / 6.0)) * (10 + 0.02 * dtot)  # Eq (112)
    return Lbulls


def beta0(phi, dtm, dlm):
    """
    This function computes the time percentage for which refractivity
    lapse-rates exceeding 100 N-units/km can be expected in the first 100
    m of the lower atmosphere
    as defined in ITU-R P.1812-6.

    Input arguments:
    phi     -   path centre latitude (deg)
    dtm     -   the longest continuous land (inland + coastal) section of the great-circle path (km)
    dlm     -   the longest continuous inland section of the great-circle path (km)

    Output arguments:
    b0      -   the time percentage that the refractivity gradient (DELTA-N) exceeds 100 N-units/km in the first 100 m of the lower atmosphere

    Example:
    b0 = beta0(phi, dtm, dlm)

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22     Ivica Stevanovic, OFCOM         Initial implementation
    """

    tau = 1 - np.exp(-(4.12 * 1e-4 * dlm**2.41))  # (3)

    mu1 = (10 ** (-dtm / (16 - 6.6 * tau)) + 10 ** (-5 * (0.496 + 0.354 * tau))) ** 0.2  # (2)

    if mu1 > 1:
        mu1 = 1

    if np.abs(phi) <= 70:
        mu4 = mu1 ** (-0.935 + 0.0176 * abs(phi))
        # (4)
        b0 = 10 ** (-0.015 * abs(phi) + 1.67) * mu1 * mu4  # (5)
    else:
        mu4 = mu1 ^ (0.3)  # (4)
        b0 = 4.17 * mu1 * mu4  # (5)

    return b0


def stdDev(f, h, R, wa):
    """stdDev computes standard deviation of location variability
    sigmaLoc = stdDev(f, h, R, wa)
    This function computes the standard deviation according to ITU-R
    P.1812-6, Annex 1, §4.8 and §4.10

    Input parameters:
    f       -   Frequency (GHz)
    h       -   receiver/mobile antenna height above the ground (m)
    R       -   height of representative clutter at the receiver/mobile location (m)
    wa      -   prediction resolution, i.e., the width of the square area over which the variability applies

    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v0    29SEP22    Ivica Stevanovic, OFCOM         Initial version
    """
    sigmaLoc = (0.52 + 0.024 * f) * wa**0.28

    if h < R:
        uh = 1
    else:
        if h >= R + 10:
            uh = 0
        else:
            uh = 1 - (h - R) / 10.0

    sigmaLoc = sigmaLoc * uh

    return sigmaLoc

def interp2(matrix_map, lon, lat, lon_spacing, lat_spacing):
    """
    Bi-linear interpolation of data contained in 2D matrix map at point (lon,lat)
    It assumes that the grid is rectangular with spacing of 1.5 deg in both lon and lat
    It assumes that lon goes from 0 to 360 deg and lat goes from 90 to -90 deg
    """

    
    latitudeOffset = 90.0 - lat
    longitudeOffset = lon
    
    if (lon < 0.0):
        longitudeOffset = lon + 360.0

    
    sizeY, sizeX = matrix_map.shape

    latitudeIndex = int(latitudeOffset / lat_spacing)
    longitudeIndex = int(longitudeOffset / lon_spacing)

    latitudeFraction = (latitudeOffset / lat_spacing) - latitudeIndex
    longitudeFraction = (longitudeOffset / lon_spacing) - longitudeIndex

    value_ul = matrix_map[latitudeIndex][longitudeIndex]
    value_ur = matrix_map[latitudeIndex][(longitudeIndex + 1) % sizeX]
    value_ll = matrix_map[(latitudeIndex + 1) % sizeY][longitudeIndex]
    value_lr = matrix_map[(latitudeIndex + 1) % sizeY][(longitudeIndex + 1) % sizeX]

    interpolatedHeight1 = (longitudeFraction * (value_ur - value_ul)) + value_ul
    interpolatedHeight2 = (longitudeFraction * (value_lr - value_ll)) + value_ll
    interpolatedHeight3 = latitudeFraction * (interpolatedHeight2 - interpolatedHeight1) + interpolatedHeight1

    return interpolatedHeight3


def isempty(x):
    if np.size(x) == 0:
        return True
    else:
        return False


def issorted(a):
    if np.all(np.diff(a) >= 0):
        return True
    else:
        return False


def sind(x):
    return np.sin(x * np.pi / 180.0)


def cosd(x):
    return np.cos(x * np.pi / 180.0)


def asind(x):
    return np.arcsin(x) * 180.0 / np.pi


def atan2d(y, x):
    return np.arctan2(y, x) * 180.0 / np.pi


def read_sg3_measurements2(filename, fileformat):
    """
    sg3db=read_sg3_measurements2(filename,fileformat)

    This function reads the file <filename> from the ITU-R SG3 databank
    written using the format <fileformat> and returns output variables in the
    cell structure vargout.
    <filename> is a string defining the file in which the data is stored
    <fileformat> is the format with which the data is written in the file:
                    = 'fryderyk_cvs' (implemented)
                    = 'cvs'          (tbi)
                    = 'xml'          (tbi)

    Output variable is a struct sg3db containing the following fields
    d               - distance between Tx and Rx
    Ef              - measured field strength at distance d
    f_GHz           - frequency in GHz
    Tx_AHaG_m       - Tx antenna height above ground in m
    RX_AHaG_m       - Rx antenna height above ground in m
    etc

    Author: Ivica Stevanovic, Federal Office of Communications, Switzerland
    Revision History:
    Date            Revision
    23MAY2016       Initial python version
    06SEP2013       Introduced corrections in order to read different
                    versions of .csv file (netherlands data, RCRCU databank and kholod data)
    22JUL2013       Initial version (IS)"""
    filename1 = filename

    sg3db = SG3DB()

    #  read the file
    try:
        fid = open(filename1, "r")
    except:
        raise IOError("P1812: File cannot be opened: " + filename1)
    if fid == -1:
        return sg3db

    #    [measurementFolder, measurementFileName, ext] = fileparts(filename1)
    measurementFolder, measurementFileName = os.path.split(filename1)
    sg3db.MeasurementFolder = measurementFolder
    sg3db.MeasurementFileName = measurementFileName

    #
    if fileformat.find("Fryderyk_csv") != -1:
        # case 'Fryderyk_csv'

        sg3db.first_point_transmitter = 1
        sg3db.coveragecode = np.array([])
        sg3db.h_ground_cover = np.array([])
        sg3db.radio_met_code = np.array([])

        # read all the lines of the file

        lines = fid.readlines()

        fid.close()

        # strip all new line characters
        lines = [line.rstrip("\n") for line in lines]

        count = 0

        while True:
            if count >= len(lines):
                break

            line = lines[count]

            dummy = line.split(",")

            if strcmp(dummy[0], "Tx LAT"):
                Tx_LAT_deg = float(dummy[1])
                sg3db.TxLAT = Tx_LAT_deg

            if strcmp(dummy[0], "Tx LON"):
                Tx_LON_deg = float(dummy[1])
                sg3db.TxLON = Tx_LON_deg

            if strcmp(dummy[0], "Rx LAT"):
                Rx_LAT_deg = float(dummy[1])
                sg3db.RxLAT = Rx_LAT_deg

            if strcmp(dummy[0], "Rx LON"):
                Rx_LON_deg = float(dummy[1])
                sg3db.RxLON = Rx_LON_deg

            if strcmp(dummy[0], "First Point Tx or Rx"):
                # if dummy[1].find('T') != -1:
                if strcmp(dummy[1], "T"):
                    sg3db.first_point_transmitter = 1
                else:
                    sg3db.first_point_transmitter = 0

            if dummy[0].find("Tot. Path Length(km):") != -1:
                TxRxDistance_km = float(dummy[1])
                sg3db.TxRxDistance = TxRxDistance_km

            if strcmp(dummy[0], "Tx site name:"):
                TxSiteName = dummy[1]
                sg3db.TxSiteName = TxSiteName

            if strcmp(dummy[0], "Rx site name:"):
                RxSiteName = dummy[1]
                sg3db.RxSiteName = RxSiteName

            if strcmp(dummy[0], "Tx Country:"):
                TxCountry = dummy[1]
                sg3db.TxCountry = TxCountry

            if strcmp(dummy[0], "Average annual values DN (N-units/km):"):
                DN = float(dummy[1])
                sg3db.DN = DN

            if strcmp(dummy[0], "Average annual sea-level surface refractivity No (N-units):"):
                N0 = float(dummy[1])
                sg3db.N0 = N0

            ## read the height profile
            if strcmp(dummy[0], "Number of Points:"):
                Npoints = int(dummy[1])

                sg3db.x = np.zeros((Npoints))
                sg3db.h_gamsl = np.zeros((Npoints))
                for i in range(0, Npoints):
                    count = count + 1

                    readLine = lines[count]

                    dummy = readLine.split(",")
                    sg3db.x[i] = float(dummy[0])
                    sg3db.h_gamsl[i] = float(dummy[1])
                    if len(dummy) > 2:
                        value = np.nan
                        if dummy[2] != "":
                            value = float(dummy[2])
                        sg3db.coveragecode = np.append(sg3db.coveragecode, value)
                        if (len(dummy)) > 3:
                            value = np.nan
                            if dummy[3] != "":
                                value = float(dummy[3])
                            sg3db.h_ground_cover = np.append(sg3db.h_ground_cover, value)
                            if (len(dummy)) > 4:
                                # Land=4, Coast=3, Sea=1
                                value = np.nan
                                if dummy[4] != "":
                                    value = float(dummy[4])
                                sg3db.radio_met_code = np.append(sg3db.radio_met_code, value)

            ## read the field strength
            if strcmp(dummy[0], "Frequency"):
                # read the next line that defines the units
                count = count + 1
                readLine = lines[count]

                # the next line should be {Begin Measurements} and the one
                # after that the number of measurement records. However, in
                # the Dutch implementation, those two lines are missing.
                # and in the implementations of csv files from RCRU, {Begin
                # Mof Measurements} is there, but the number of
                # measurements (line after) may be missing
                # This is the reason we are checking for these two lines in
                # the following code
                f = np.array([])

                dutchflag = True
                count = count + 1
                readLine = lines[count]
                if strcmp(readLine, "{Begin of Measurements"):
                    # check if the line after that contains only one number
                    # or the data
                    count = count + 1
                    readLine = lines[count]  # the line with the number of records or not
                    dummy = readLine.split(",")
                    if len(dummy) > 2:
                        # if isempty([dummy{2:end}])

                        if dummy[1:-1] == "":
                            # this is the number of data - the info we do
                            # not use, read another line
                            count = count + 1
                            readLine = lines[count]
                            dutchflag = False
                        else:
                            dutchflag = True

                    else:
                        dutchflag = False
                        count = count + 1
                        readLine = lines[count]

                # read all the lines until the {End of Measurements} tag

                kindex = 0
                while True:
                    if kindex == 0:
                        # do not read the new line, but use the one read in
                        # the previous step
                        kindex = 0
                    else:
                        count = count + 1
                        readLine = lines[count]

                    if count >= len(lines):
                        break

                    if strcmp(readLine, "{End of Measurements}"):
                        break

                    dummy = readLine.split(",")

                    f = np.append(f, float(dummy[0]))
                    sg3db.frequency = np.append(sg3db.frequency, float(dummy[0]))

                    col = 1
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])

                    sg3db.hTx = np.append(sg3db.hTx, value)

                    col = 2
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])

                    sg3db.hTxeff = np.append(sg3db.hTxeff, value)

                    col = 3
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.hRx = np.append(sg3db.hRx, value)

                    col = 4
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.polHVC = np.append(sg3db.polHVC, value)

                    col = 5
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.TxdBm = np.append(sg3db.TxdBm, value)

                    col = 6
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.MaxLb = np.append(sg3db.MaxLb, value)

                    col = 7
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.Txgn = np.append(sg3db.Txgn, value)

                    col = 8
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.Rxgn = np.append(sg3db.Rxgn, value)

                    col = 9
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.RxAntDO = np.append(sg3db.RxAntDO, value)

                    col = 10
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.ERPMaxHoriz = np.append(sg3db.ERPMaxHoriz, value)

                    col = 11
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.ERPMaxVertical = np.append(sg3db.ERPMaxVertical, value)

                    col = 12
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.ERPMaxTotal = np.append(sg3db.ERPMaxTotal, value)

                    col = 13
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.HRPred = np.append(sg3db.HRPred, value)

                    col = 14
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])

                    if np.isnan(value):
                        print("Time percentage not defined. Default value 50% assumed.")
                        value = 50

                    sg3db.TimePercent = np.append(sg3db.TimePercent, value)

                    col = 15
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.LwrFS = np.append(sg3db.LwrFS, value)

                    col = 16
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.MeasuredFieldStrength = np.append(sg3db.MeasuredFieldStrength, value)

                    #
                    col = 17
                    value = np.nan
                    if dummy[col] != "":
                        value = float(dummy[col])
                    sg3db.BasicTransmissionLoss = np.append(sg3db.BasicTransmissionLoss, value)

                    #
                    if len(dummy) > 18:
                        col = 18

                        value = np.nan
                        if dummy[col] != "":
                            value = float(dummy[col])
                        sg3db.RxHeightGainGroup = np.append(sg3db.RxHeightGainGroup, value)

                        col = 19
                        value = np.nan
                        if dummy[col] != "":
                            value = float(dummy[col])
                        sg3db.IsTopHeightInGroup = np.append(sg3db.IsTopHeightInGroup, value)

                    else:
                        sg3db.RxHeightGainGroup = np.append(sg3db.RxHeightGainGroup, np.nan)
                        sg3db.IsTopHeightInGroup = np.append(sg3db.IsTopHeightInGroup, np.nan)

                    kindex = kindex + 1

                # Number of different measured data sets
                Ndata = kindex
                sg3db.Ndata = Ndata

            count = count + 1

    elif fileformat.find("csv") != -1:
        print("csv format not yet implemented.")

    elif fileformat.find("xml") != -1:
        print("xml format not yet implemented.")

    return sg3db


class SG3DB:
    def __init__(self):
        # sg3db : structure containing
        #  - data from ITU-R SG3 databank file
        #  - data used for P1546Compute

        self.MeasurementFolder = ""
        self.MeasurementFileName = ""
        self.first_point_transmitter = 1
        self.coveragecode = np.array([])
        self.h_ground_cover = np.array([])
        self.radio_met_code = np.array([])
        self.TxRxDistance = []
        self.TxSiteName = ""
        self.TxLAT = []
        self.TxLON = []
        self.RxLAT = []
        self.RxLON = []
        self.N0 = []
        self.DN = []
        self.RxSiteName = ""
        self.TxCountry = ""
        self.x = np.array([])
        self.h_gamsl = np.array([])
        self.frequency = np.array([])
        self.hTx = np.array([])
        self.hTxeff = np.array([])
        self.hRx = np.array([])
        self.polHVC = np.array([])
        self.TxdBm = np.array([])
        self.MaxLb = np.array([])
        self.Txgn = np.array([])
        self.Rxgn = np.array([])
        self.RxAntDO = np.array([])
        self.ERPMaxHoriz = np.array([])
        self.ERPMaxVertical = np.array([])
        self.ERPMaxTotal = np.array([])
        self.HRPred = np.array([])
        self.TimePercent = np.array([])
        self.LwrFS = np.array([])
        self.MeasuredFieldStrength = np.array([])
        self.BasicTransmissionLoss = np.array([])
        self.RxHeightGainGroup = np.array([])
        self.IsTopHeightInGroup = np.array([])
        self.RxHeightGainGroup = np.array([])
        self.IsTopHeightInGroup = np.array([])

        self.debug = 0
        self.pathinfo = 0
        self.fid_log = 0
        self.TransmittedPower = np.array([])
        self.LandPath = 0
        self.SeaPath = 0
        self.ClutterCode = []
        self.userChoiceInt = []
        self.RxClutterCodeP1546 = ""
        self.RxClutterHeight = []
        self.TxClutterHeight = []
        self.PredictedFieldStrength = []
        self.q = 50
        self.heff = []
        self.Ndata = []
        self.eff1 = []
        self.tca = []

    def __str__(self):
        userChoiceInt = self.userChoiceInt
        out = "The following input data is defined:" + "\n"
        out = out + " PTx            = " + str(self.TransmittedPower[userChoiceInt]) + "\n"
        out = out + " f              = " + str(self.frequency[userChoiceInt]) + "\n"
        out = out + " t              = " + str(self.TimePercent[userChoiceInt]) + "\n"
        out = out + " q              = " + str(self.q) + "\n"
        out = out + " heff           = " + str(self.heff) + "\n"
        out = out + " area           = " + str(self.RxClutterCodeP1546) + "\n"
        out = out + " pathinfo       = " + str(self.pathinfo) + "\n"
        out = out + " h2             = " + str(self.hRx[userChoiceInt]) + "\n"
        out = out + " ha             = " + str(self.hTx[userChoiceInt]) + "\n"
        out = out + " htter          = " + str(self.h_gamsl[0]) + "\n"
        out = out + " hrter          = " + str(self.h_gamsl[-1]) + "\n"
        out = out + " R1             = " + str(self.TxClutterHeight) + "\n"
        out = out + " R2             = " + str(self.RxClutterHeight) + "\n"
        out = out + " eff1           = " + str(self.eff1) + "\n"
        out = out + " eff2           = " + str(self.tca) + "\n"
        out = out + " debug          = " + str(self.debug) + "\n"
        # out = out + " fid_log        = " + str(self.fid_log           ) + "\n"
        out = out + " Predicted Field Strength        = " + str(self.PredictedFieldStrength) + "\n"

        return out

    def update(self, other):
        self.MeasurementFolder = other.MeasurementFolder
        self.MeasurementFileName = other.MeasurementFileName
        self.first_point_transmitter = other.first_point_transmitter
        self.coveragecode = other.coveragecode
        self.h_ground_cover = other.h_ground_cover
        self.radio_met_code = other.radio_met_code
        self.TxRxDistance = other.TxRxDistance
        self.TxLAT = other.TxLAT
        self.TxLON = other.TxLON
        self.RxLAT = other.RxLAT
        self.RxLON = other.RxLON
        self.N0 = other.N0
        self.DN = other.DN
        self.TxSiteName = other.TxSiteName
        self.RxSiteName = other.RxSiteName
        self.TxCountry = other.TxCountry
        self.coveragecode = other.coveragecode
        self.h_ground_cover = other.h_ground_cover
        self.radio_met_code = other.radio_met_code
        self.x = other.x
        self.h_gamsl = other.h_gamsl
        self.frequency = other.frequency
        self.hTx = other.hTx
        self.hTxeff = other.hTxeff
        self.hRx = other.hRx
        self.polHVC = other.polHVC
        self.TxdBm = other.TxdBm
        self.MaxLb = other.MaxLb
        self.Txgn = other.Txgn
        self.Rxgn = other.Rxgn
        self.RxAntDO = other.RxAntDO
        self.ERPMaxHoriz = other.ERPMaxHoriz
        self.ERPMaxVertical = other.ERPMaxVertical
        self.ERPMaxTotal = other.ERPMaxTotal
        self.HRPred = other.HRPred
        self.TimePercent = other.TimePercent
        self.LwrFS = other.LwrFS
        self.MeasuredFieldStrength = other.MeasuredFieldStrength
        self.BasicTransmissionLoss = other.BasicTransmissionLoss
        self.RxHeightGainGroup = other.RxHeightGainGroup
        self.IsTopHeightInGroup = other.IsTopHeightInGroup
        self.RxHeightGainGroup = other.RxHeightGainGroup
        self.IsTopHeightInGroup = other.IsTopHeightInGroup

        self.debug = other.debug
        self.pathinfo = other.pathinfo
        self.fid_log = other.fid_log
        self.TransmittedPower = other.TransmittedPower
        self.LandPath = other.LandPath
        self.SeaPath = other.SeaPath
        self.ClutterCode = other.ClutterCode
        self.userChoiceInt = other.userChoiceInt
        self.RxClutterCodeP1546 = other.RxClutterCodeP1546
        self.RxClutterHeight = other.RxClutterHeight
        self.TxClutterHeight = other.TxClutterHeight
        self.PredictedFieldStrength = other.PredictedFieldStrength
        self.q = other.q
        self.heff = other.heff
        self.Ndata = other.Ndata


def strcmp(str1, str2):
    """
    This function compares two strings (by previously removing any white
    spaces and open/close brackets from the strings, and changing them to
    lower case).
    Author: Ivica Stevanovic, Federal Office of Communications, Switzerland
    Revision History:
    Date            Revision
    23MAY2016       Initial python version (IS)
    22JUL2013       Initial version (IS)
    """

    str1 = str1.replace(" ", "")
    str2 = str2.replace(" ", "")
    str1 = str1.replace("(", "")
    str2 = str2.replace("(", "")
    str1 = str1.replace(")", "")
    str2 = str2.replace(")", "")
    str1 = str1.lower()
    str2 = str2.lower()

    if str1.find(str2) == -1:
        return False
    else:
        return True


def clutter(i, ClutterCodeType):
    """
    ClutterClass, P1546ClutterClass, R = clutter(i, ClutterCode)
    This function maps the value i of a given clutter code type into
    the corresponding clutter class description, P1546 clutter class description
    and clutter height R.
    The implemented ClutterCodeTypes are:
    'OFCOM' (as defined in the SG3DB database on SUI data from 2012
    'TDB'   (as defined in the RCRU database and UK data) http://www.rcru.rl.ac.uk/njt/linkdatabase/linkdatabase.php
    'NLCD'  (as defined in the National Land Cover Dataset) http://www.mrlc.gov/nlcd06_leg.php
    'LULC'  (as defined in Land Use and Land Clutter database) http://transition.fcc.gov/Bureaus/Engineering_Technology/Documents/bulletins/oet72/oet72.pdf
    'GlobCover' (as defined in ESA's GlobCover land cover maps) http://due.esrin.esa.int/globcover/
    'DNR1812' (as defined in the implementation tests for DNR P.1812)
    'default' (land paths, rural area, R = 10 m)


    Rev   Date        Author                          Description
    -------------------------------------------------------------------------------
    v3    24MAY16     Ivica St3vanovic, OFCOM         Initial python version
    v2    29Apr15     Ivica Stevanovic, OFCOM         Introduced 'default' option for ClutterCodeTypes
    v1    26SEP13     Ivica Stevanovic, OFCOM         Introduced it as a function
    """

    if strcmp(ClutterCodeType, "OFCOM"):
        if i == 0:
            RxClutterCode = "Unknown"
        elif i == 1:
            RxClutterCode = "Water (salt)"
        elif i == 2:
            RxClutterCode = "Water (fresh)"
        elif i == 3:
            RxClutterCode = "Road/Freeway"
        elif i == 4:
            RxClutterCode = "Bare land"
        elif i == 5:
            RxClutterCode = "Bare land/rock"
        elif i == 6:
            RxClutterCode = "Cultivated land"
        elif i == 7:
            RxClutterCode = "Scrub"
        elif i == 8:
            RxClutterCode = "Forest"
        elif i == 9:
            RxClutterCode = "Low dens. suburban"
        elif i == 10:
            RxClutterCode = "Suburban"
        elif i == 11:
            RxClutterCode = "Low dens. urban"
        elif i == 12:
            RxClutterCode = "Urban"
        elif i == 13:
            RxClutterCode = "Dens. urban"
        elif i == 14:
            RxClutterCode = "High dens. urban"
        elif i == 15:
            RxClutterCode = "High rise industry"
        elif i == 16:
            RxClutterCode = "Skyscraper"
        else:
            RxClutterCode = "Unknown data"

        if i == 8 or i == 12:
            RxP1546Clutter = "Urban"
            R2external = 15
        elif i == 1 or i == 2:
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i < 8 and i > 2:
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i <= 11 and i > 8:
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i <= 16 and i > 11:
            RxP1546Clutter = "Dense Urban"
            R2external = 20
        else:
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "TDB"):
        if i == 0:
            RxClutterCode = "No data"
            RxP1546Clutter = ""
            R2external = []
        elif i == 1:
            RxClutterCode = "Fields"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Road"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 3:
            RxClutterCode = "BUILDINGS"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 4:
            RxClutterCode = "URBAN"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 5:
            RxClutterCode = "SUBURBAN"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 6:
            RxClutterCode = "VILLAGE"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 7:
            RxClutterCode = "SEA"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 8:
            RxClutterCode = "LAKE"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 9:
            RxClutterCode = "RIVER"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 10:
            RxClutterCode = "CONIFER"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 11:
            RxClutterCode = "NON_CONIFER"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 12:
            RxClutterCode = "MUD"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 13:
            RxClutterCode = "ORCHARD"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 14:
            RxClutterCode = "MIXED_TREES"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 15:
            RxClutterCode = "DENSE_URBAN"
            RxP1546Clutter = "Dense Urban"
            R2external = 30
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "NLCD"):
        if i == 11:
            RxClutterCode = "Open Water"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 12:
            RxClutterCode = "Perennial Ice/Snow"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 21:
            RxClutterCode = "Developed, Open Space"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 22:
            RxClutterCode = "Developed, Low Intensity"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 23:
            RxClutterCode = "Developed, Medium Intensity"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 24:
            RxClutterCode = "Developed High Intensity"
            RxP1546Clutter = "Urban"
            R2external = 20

        elif i == 31:
            RxClutterCode = "Barren Land (Rock/Sand/Clay)"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 41:
            RxClutterCode = "Deciduous Forest"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 42:
            RxClutterCode = "Evergreen Forest"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 43:
            RxClutterCode = "Mixed Forest"
            RxP1546Clutter = "Urban"
            R2external = 20

        elif i == 51:
            RxClutterCode = "Dwarf Scrub"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 52:
            RxClutterCode = "Shrub/Scrub"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 71:
            RxClutterCode = "Grassland/Herbaceous"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 72:
            RxClutterCode = "Sedge/Herbaceous"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 73:
            RxClutterCode = "Lichens - Alaska only"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 74:
            RxClutterCode = "Moss - Alaska only"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 81:
            RxClutterCode = "Pasture/Hay"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 82:
            RxClutterCode = "Cultivated Crops"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 90:
            RxClutterCode = "Woody Wetlands"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 95:
            RxClutterCode = "Emergent Herbaceous Wetlands"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "LULC"):
        if i == 11:
            RxClutterCode = "Residential"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 12:
            RxClutterCode = "Commercial services"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 13:
            RxClutterCode = "Industrial"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 14:
            RxClutterCode = "Transportation, communications, utilities"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 15:
            RxClutterCode = "Industrial and commercial complexes"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 16:
            RxClutterCode = "Mixed urban and built-up lands"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 17:
            RxClutterCode = "Other urban and built-up land"
            RxP1546Clutter = "Suburban"
            R2external = 10

        elif i == 21:
            RxClutterCode = "Cropland and pasture"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 22:
            RxClutterCode = "Orchards, groves, vineyards, nurseries, and horticultural"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 23:
            RxClutterCode = "Confined feeding operations"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 24:
            RxClutterCode = "Other agricultural land"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 31:
            RxClutterCode = "Herbaceous rangeland"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 32:
            RxClutterCode = "Shrub and brush rangeland"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 33:
            RxClutterCode = "Mixed rangeland"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 41:
            RxClutterCode = "Deciduous forest land"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 42:
            RxClutterCode = "Evergreen forest land"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 43:
            RxClutterCode = "Mixed forest land"
            RxP1546Clutter = "Urban"
            R2external = 20

        elif i == 51:
            RxClutterCode = "Streams and canals"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 52:
            RxClutterCode = "Lakes"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 53:
            RxClutterCode = "Reservoirs"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 54:
            RxClutterCode = "Bays and estuaries"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10

        elif i == 61:
            RxClutterCode = "Forested wetland"
            RxP1546Clutter = "Urban"
            R2external = 20
        elif i == 62:
            RxClutterCode = "Non-forest wetland"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10

        elif i == 71:
            RxClutterCode = "Dry salt flats"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 72:
            RxClutterCode = "Beaches"
            RxP1546Clutter = "Adjacent to Sea"
            R2external = 10
        elif i == 73:
            RxClutterCode = "Sandy areas other than beaches"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 74:
            RxClutterCode = "Bare exposed rock"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 75:
            RxClutterCode = "Strip mines, quarries, and gravel pits"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 76:
            RxClutterCode = "Transitional areas"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 77:
            RxClutterCode = "Mixed barren land"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 81:
            RxClutterCode = "Shrub and brush tundra"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 82:
            RxClutterCode = "Herbaceous tundra"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 83:
            RxClutterCode = "Bare ground"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 84:
            RxClutterCode = "Wet tundra"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 85:
            RxClutterCode = "Mixed tundra"
            RxP1546Clutter = "Rural"
            R2external = 10

        elif i == 91:
            RxClutterCode = "Perennial snowfields"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 92:
            RxClutterCode = "Glaciers"
            RxP1546Clutter = "Rural"
            R2external = 10
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "GlobCover"):
        if i == 1:
            RxClutterCode = "Water/Sea"
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Open/Rural"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 3:
            RxClutterCode = "Suburban"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 4:
            RxClutterCode = "Urban/trees/forest"
            RxP1546Clutter = "Urban"
            R2external = 15
        elif i == 5:
            RxClutterCode = "Dense Urban"
            RxP1546Clutter = "Dense Urban"
            R2external = 20
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "P1546"):
        if i == 1:
            RxClutterCode = "Water/Sea"
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Open/Rural"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 3:
            RxClutterCode = "Suburban"
            RxP1546Clutter = "Suburban"
            R2external = 10
        elif i == 4:
            RxClutterCode = "Urban/trees/forest"
            RxP1546Clutter = "Urban"
            R2external = 15
        elif i == 5:
            RxClutterCode = "Dense Urban"
            RxP1546Clutter = "Dense Urban"
            R2external = 20
        else:
            RxClutterCode = "Unknown"
            RxP1546Clutter = "Suburban"
            R2external = 0

    elif strcmp(ClutterCodeType, "DNR1812"):
        if i == 0:
            RxClutterCode = "Inland"
            RxP1546Clutter = "Rural"
            R2external = 10
        elif i == 1:
            RxClutterCode = "Coastal"
            RxP1546Clutter = "Sea"
            R2external = 10
        elif i == 2:
            RxClutterCode = "Sea"
            RxP1546Clutter = "Sea"
            R2external = 10
        else:
            RxClutterCode = "Unknown data"
            RxP1546Clutter = ""
            R2external = []

    elif strcmp(ClutterCodeType, "default"):
        print("Clutter code type set to default:")
        print("Rural, R = 10 m")
        RxClutterCode = "default"
        RxP1546Clutter = "Rural"
        R2external = 10

    else:
        RxClutterCode = ""
        RxP1546Clutter = ""
        R2external = []

    return RxClutterCode, RxP1546Clutter, R2external
