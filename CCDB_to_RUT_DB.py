'''
Created on Apr 06, 2016

@author: shunt
'''

'''Python Modules'''
from numpy import array, zeros, asarray
from datetime import date
from os import listdir, makedirs
from os.path import join, split
from collections import OrderedDict
from netCDF4 import Dataset
import glob
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import exp


'''----------------------------- Constants --------------------------------'''

#Sceince
Q = 1.6E-19

#CCD
DELTA_L = 1.25
PXL_SIZE = 22.5E-6
PXL_AREA = PXL_SIZE**2*10**4
N_PXL_TOT = 576
DARK_NOISE = [0.731, 0.779, 0.852, 0.929, 0.971, 1.102, 1.284, 1.465]

#SPECTRAL
LS =  array([[400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 
              665.0, 673.75, 681.25, 708.75, 753.75, 761.25, 
              764.375, 767.50, 778.75, 865.0, 885.0, 900.0, 
              940.0, 1020.0],
             [400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 
              665.0, 673.75, 681.25, 708.75, 753.75, 761.25, 
              764.375, 767.50, 778.75, 865.0, 885.0, 900.0, 
              940.0, 1020.0],
             [400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 
              665.0, 673.75, 681.25, 708.75, 753.75, 761.25, 
              764.375, 767.50, 778.75, 865.0, 885.0, 900.0, 
              940.0, 1020.0],
             [400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 
              665.0, 673.75, 681.25, 708.75, 753.75, 761.25, 
              764.375, 767.50, 778.75, 865.0, 885.0, 900.0, 
              940.0, 1020.0],
             [400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 
              665.0, 673.75, 681.25, 708.75, 753.75, 761.25, 
              764.375, 767.50, 778.75, 865.0, 885.0, 900.0, 
              940.0, 1020.0]])
LS_SM= array([1040., 1040., 1040., 1040., 1040.])
L_MAX = 1040.
L_MIN = 390.

#Error contributors
U_DIFFCHAR = 0.0025 #Diffuser Characterisation Error
U_DIFFMOD = 0.0056 # Diffuser Model Error
U_DIFFALIGN = 0.0010 # Diffuser Alignment Error
U_CALISTR = 0.00016 # Calibration Straylight Error
U_POSTCALISTR = 0.002 # Post-Calibration Strayligh Error
U_CALISPECK = 0.002 # Calibration Speckle Error

U_INSTAGE = array([0.000800000, 0.000783871, 0.000745161, 0.000683871,
                   0.000658065, 0.000593548, 0.000516129, 0.000458065, 
                   0.000446774, 0.000437097, 0.000401613, 0.000343548, 
                   0.000333871, 0.000329839, 0.000325806, 0.000311290, 
                   0.000200000, 0.000174194, 0.000154839, 0.000103226, 
                   0.000000000])

#Instrument parameters
E_INL = 0.0015
E_DNL = 0.406
I = 1.59 # Dark Current Density

class CCDB_Data():
    
    def __init__(self, directory=None, **kwargs):
        """Open required OLCI programming, characterisation and calibration
        data from CCDB to be save to XML read_read_cfile for use in OLCI_RUT"""
        
        '''Initialise Data Groups'''
        # Data group dictionary first entry defines the data group:
        # {'grp_def': [description, NetCDF_group_name]}
        # Following are data entries of format:
        # {'variable name': [data, description, units, NetCDF_dims]}
  
        #OLCI band settings group
        self.band_grp = OrderedDict()
        self.band_grp["grp_def"] = ["OLCI band settings", "olci_band_settings"]
        self.band_grp["ls"] = [None, "Band centre wavelength [SCCDB]", "nm", ("mod", "band")]
        self.band_grp["n_pxls"] = [None, "Number of pixels per band [SCCDB]", "-", ("mod", "band")]
        self.band_grp["n_ub_pxls"] = [None, "Number of pixels per uband [SCCDB]", "-", ("mod", "band")]
        self.band_grp["n_ubands"] = [None, "Number of ubands per band [SCCDB]", "-", ("band",)]
        self.band_grp["gains"] = [None, "Gain value per band [SCCDB]", "-", ("mod", "band")]
        self.band_grp["gain_nums"] = [None, "Gain number per band [SCCDB]", "-", ("mod", "band")]
            
        #OLCI smear band settings group
        self.smear_band_grp = OrderedDict()
        self.smear_band_grp["grp_def"] = ["OLCI smear band settings", "olci_smear_band_settings"]
        self.smear_band_grp["ls_sm"] = [None, "Smear Band Centre Wavelength [SCCDB]", "nm", ("mod",)]
        self.smear_band_grp["n_pxls_sm"] = [None, "Number of pixels in smear band per module [SCCDB]", "-", ("mod",)]
        self.smear_band_grp["n_ub_pxls_sm"] = [None, "Number of pixels per smear uband per module [SCCDB]", "-", ("mod",)]
        self.smear_band_grp["n_ubands_sm"] = [None, "Number of ubands in smear band [SCCDB]", "-", None]
        self.smear_band_grp["gains_sm"] = [None, "Gain value in smear band per module [SCCDB]", "-", ("mod",)]
        self.smear_band_grp["gain_nums_sm"] = [None, "Gain number in smear band per module [SCCDB]", "-", ("mod",)]
        
        #OLCI CCD parameters group
        self.CCD_grp = OrderedDict()
        self.CCD_grp["grp_def"] = ["OLCI CCD parameters", "olci_ccd_parameters"]
        self.CCD_grp["n_pxl_tot"] = [None, "Total number of pxls across CCD [SCCDB]", "-", None]
        self.CCD_grp["delta_l"] = [DELTA_L, "Spectral Sampling Interval of CCDB [SCCDB]", "-", None]
        
        #OLCI time parameters group
        self.time_grp = OrderedDict()
        self.time_grp["grp_def"] = ["OLCI time parameters", "olci_ccd_parameters"]
        self.time_grp["t_frame"] = [None, "Integration time of olci image [SCCDB]", "s", None]
        self.time_grp["t_int"] = [None, "Integration time of olci image [SCCDB]", "s", None]
        self.time_grp["t_trans"] = [None, "Integration time of olci image [SCCDB]", "s", None]
        
        #OLCI-RUT parameters group
        self.rut_grp = OrderedDict()
        self.rut_grp["grp_def"] = ["OLCI-RUT parameters", "olci_rut_parameters"]
        self.rut_grp["C_RSs"] = [None, "Radiance to CCD signal coefficient [calculated from SCCDB]", "", ("mod", "band")]
        self.rut_grp["digi_steps"] =[None, "Coefficient to propagate signal through the VAM [calculated from SCCDB]", "", ("mod", "band")]
        self.rut_grp["digi_steps_sm"] = [None, "Coefficient to propagate smear signal through the VAM [calculated from SCCDB]", "", ("mod",)]
        self.rut_grp["DSs"] = [None, "Dark signal per band [calculated from SCCDB]", "", ("mod", "band")]
        self.rut_grp["DSs_sm"] = [None, "Dark signal of smear band [calculated from SCCDB]", "", ("mod",)]
        self.rut_grp["OCL_steps"] = [None, "OCL coefficient to propagate signal through the VAM [calculated from SCCDB]", "", ("mod", "band")]
        self.rut_grp["OCL_steps_sm"] = [None, "OCL coefficient to propagate smear signal through the VAM [calculated from SCCDB]", "", ("mod",)]

        #OLCI unc parameters group
        self.unc_grp = OrderedDict()
        self.unc_grp["grp_def"] = ["OLCI unc parameters", "olci_unc_parameters"]
        
        #Constants
        self.unc_grp["u_calispeck_2"] = [U_CALISPECK**2, "u_calispeck^2", "%", None]
        self.unc_grp["u_calistr_2"] = [U_CALISTR**2, "Uncertainty caused by straylight within calibration mechanism affecting the BRDF modelisation squared", "%", None]
        self.unc_grp["u_diffalign_2"] = [U_DIFFALIGN**2, "Uncertainty caused by variation in sun incidence angle on to the diffuser squared", "%", None]
        self.unc_grp["u_diffchar_2"] = [U_DIFFCHAR**2, "Calibration diffuser characterisation measurement uncertainty squared", "%", None]
        self.unc_grp["u_diffmod_2"] = [U_DIFFMOD**2, "Calibration diffuser BRDF modelisation uncertainty squared", "%", None]
        self.unc_grp["u_postcalistr_2"] = [U_POSTCALISTR**2, "Camera stralight during calibration uncertainty squared", "%", None]
        
        #Band-dependent
        self.unc_grp["u_diff1age"] = [None, "Uncertainty due to aging of the calibration diffuser", "%", None]
        self.unc_grp["u_diff2age"] = [None, "Uncertainty due to aging of the reference diffuser", "%", None]
        self.unc_grp["u_instage"] = [U_INSTAGE, "Uncertainty caused by instrument aging between calibrations", "%", ("band",)]
        self.unc_grp["u_ccdstab_2"] = [None, "CCD response stability uncertainty squared", "%", ("band","mod")]
        
        #Scene-dependent parameters
        self.unc_grp["a_DNL"] = [None, "Parameter used to determine uncertainty from non-linaerity in the ADC qunatisation process", "%", ("band", "mod")]
        self.unc_grp["b_DNL"] = [None, "Parameter used to determine uncertainty from non-linaerity in the ADC qunatisation process", "%", ("band", "mod")]
        self.unc_grp["a_off"] = [None, "Parameter used to determine uncertainty from offset compensation loop clamping stability", "%", ("band", "mod")]
        self.unc_grp["b_off"] = [None, "Parameter used to determine uncertainty from offset compensation loop clamping stability", "%", ("band", "mod")]
        self.unc_grp["a_darkstab"] = [None, "Parameter used to determine dark signal stability uncertainty", "%", ("band", "mod")]
        self.unc_grp["a_PS"] = [None, "Parameter used to determine error caused by uncorrected Period Signal", "%", ("band", "mod")]
        self.unc_grp["b_PS"] = [None, "Parameter used to determine error caused by uncorrected Period Signal", "%", ("band", "mod")]
        self.unc_grp["a_SGR"] = [None, "Parameter used to determine uncertainty due to lack of smear gain knowledge", "%", ("band", "mod")]
        self.unc_grp["b_SGR"] = [None, "Parameter used to determine uncertainty due to lack of smear gain knowledge", "%", ("band", "mod")]
        self.unc_grp["c_SGR"] = [None, "Parameter used to determine uncertainty due to lack of smear gain knowledge", "%", ("band", "mod")]
        self.unc_grp["a_INL"] = [None, "Parameter to determine video chain non-linearity correction uncertainty", "%", ("band", "mod")]
        self.unc_grp["b_INL"] = [None, "Parameter to determine video chain non-linearity correction uncertainty", "%", ("band", "mod")]
        self.unc_grp["c_INL"] = [None, "Parameter to determine video chain non-linearity correction uncertainty", "%", ("band", "mod")]
        self.unc_grp["d_INL"] = [None, "Parameter to determine video chain non-linearity correction uncertainty", "%", ("band", "mod")]
        self.unc_grp["a_noise"] = [None, "Parameter to determine Instrument noise", "%", ("band", "mod")]
        
        '''Open Variables'''
        self.SCCDB = split(directory)[1]
        self.open_SCCDB(directory, **kwargs)
        self.open_u_contributors()

    def open_SCCDB(self, dir):

        # Path to CCDB NetCDF files
        path_prog = glob.glob(join(dir, 'NetCDF', "S3A_OL_CCDB_PROG*"))[0]
        path_cali = glob.glob(join(dir, 'NetCDF', "S3A_OL_CCDB_CALI*"))[0]
        path_char = glob.glob(join(dir, 'NetCDF', "S3A_OL_CCDB_CHAR*"))[0]
        
        ''' -------------------- programmed data ------------------------- '''
        #Open OLCI band setting programming data from the CCDB
        prog_grp = Dataset(path_prog, "r", format='NETCDF4')
        
        #  n_pxl_tot - total number of pixels
        self.CCD_grp['n_pxl_tot'][0] = N_PXL_TOT
        
        # n_ubands(_sm) - number of microbands per band
        self.band_grp['n_ubands'][0], self.smear_band_grp['n_ubands_sm'][0] = self.get_n_uband(prog_grp)

        # n_ub_pxls(_sm) - number of pixels per band microband
        self.band_grp['n_ub_pxls'][0], self.smear_band_grp['n_ub_pxls_sm'][0] = self.calc_n_ub_pxl(prog_grp,
                                                                                                   self.band_grp['n_ubands'][0])
        # n_pxls(_sm) - total number of pixels per band
        self.band_grp['n_pxls'][0], self.smear_band_grp['n_pxls_sm'][0] = self.calc_n_pxl(self.band_grp['n_ubands'][0], 
                                                                                          self.smear_band_grp['n_ubands_sm'][0],
                                                                                          self.band_grp['n_ub_pxls'][0],
                                                                                          self.smear_band_grp['n_ub_pxls_sm'][0])
        # gain_nums(_sm) - gain number of each band
        self.band_grp['gain_nums'][0], self.smear_band_grp['gain_nums_sm'][0] = self.calc_gain_num(prog_grp,
                                                                                                   self.band_grp['n_ubands'][0])
        # Close CCDB prog data
        prog_grp.close()

        ''' -------------------- calibration data ------------------------- '''
        #Open OLCI calibration data from the CCDB
        cali_grp = Dataset(path_cali, "r", format='NETCDF4')

        # Temporarily just hard code wavelength values until info becomes
        # available from the calibration
        self.band_grp['ls'][0] = LS
        self.smear_band_grp['ls_sm'][0] = LS_SM
        
        # Close CCDB char data
        cali_grp.close()

        ''' ----------------- characterisation data ---------------------- '''
        #Open OLCI characteristaion data from the CCDB
        char_grp = Dataset(path_char, "r", format='NETCDF4')        

        # t_int, t_trans & t_frame - integration, transfer and frame times
        self.time_grp['t_int'][0] = char_grp.groups['olci_time_parameters_Nominal']\
                                            .variables['olci_integration_time'][:]/1000000.0         
        self.time_grp['t_trans'][0] = char_grp.groups['olci_time_parameters_Nominal']\
                                              .variables['olci_transfer_time'][:]/1000000.0
        self.time_grp['t_frame'][0] = char_grp.groups['olci_time_parameters_Nominal']\
                                              .variables['olci_sampling_time'][:]/1000000.0
        # gains(_sm) - gain value of each band
        self.band_grp['gains'][0], self.smear_band_grp['gains_sm'][0] = self.calc_gains(char_grp, 
                                                                                        self.band_grp['gain_nums'][0], 
                                                                                        self.smear_band_grp['gain_nums_sm'][0])      
        ''' --------------------- OLCI-RUT Parameters -------------------- '''
        # digi_steps(_sm) - Coeff to propagate CCD values through VAM
        self.rut_grp['digi_steps'][0], self.rut_grp['digi_steps_sm'][0] = self.calc_digi_steps(char_grp,
                                                                                               self.band_grp['gains'][0],
                                                                                               self.smear_band_grp['gains_sm'][0])
        # OCL_digi_step(_sm) - Coeff with OCL
        self.rut_grp['OCL_steps'][0], self.rut_grp['OCL_steps_sm'][0] = self.calc_OCL_steps(self.rut_grp['digi_steps'][0], 
                                                                                            self.rut_grp['digi_steps_sm'][0],
                                                                                            self.band_grp['gain_nums'][0],
                                                                                            self.smear_band_grp['gain_nums_sm'][0])
        # DS - Dark signal
        self.rut_grp['DSs'][0] = self.calc_DS(char_grp, self.band_grp['ls'][0],
                                              self.time_grp['t_frame'][0])
        self.rut_grp['DSs_sm'][0] = self.calc_DS(char_grp, self.smear_band_grp['ls_sm'][0],
                                                 self.time_grp['t_frame'][0])
        
        # C_RS - Coefficient to convert radiance to CCD signal
        self.rut_grp['C_RSs'][0] = self.calc_C_RS(char_grp, self.time_grp['t_int'][0])
        
        # Close CCDB char data
        char_grp.close()
        
    def open_u_contributors(self):
        #Defined values           
        X_cal = [8346.74, 10275.87, 8490.68, 7040.31, 16019.41,
                 26274.02, 9586.41, 7028.65, 6713.09, 14257.85,
                 16630.85, 17849.13, 14590.06, 15959.11, 17091.92,
                 13097.76, 18697.28, 18248.93, 16978.38, 14756.30,
                 13260.50]
        X_cal_DS = [2.28, 2.55, 2.69, 2.74, 2.83, 3.09, 3.14, 3.15,
                    3.16, 3.19, 3.33, 3.43, 3.45, 3.48, 3.64, 3.86,
                    4.01, 4.08, 4.24, 4.34, 4.38]            
        X_cal_sm = 1451.86
        X_cal_tot = [X+X_cal_sm+DS for X, DS in zip(X_cal, X_cal_DS)]       
        X_cals = [X_cal, X_cal, X_cal, X_cal, X_cal]
        X_cals_sm = [X_cal_sm, X_cal_sm, X_cal_sm, X_cal_sm, X_cal_sm]
        X_cals_tot = [X_cal_tot, X_cal_tot, X_cal_tot, X_cal_tot, X_cal_tot]
        
        self.unc_grp["u_ccdstab_2"][0] = self.calc_u_ccdstabs_2(self.band_grp['ls'][0])
        self.unc_grp["u_diff1age"][0] = self.calc_u_diffage(self.band_grp['ls'][0], 0.25)
        self.unc_grp["u_diff2age"][0] = self.calc_u_diffage(self.band_grp['ls'][0], 7.5/10/6)
        
        self.unc_grp["a_DNL"][0] = self.calc_a_DNL(self.band_grp['n_ubands'][0],
                                                   self.band_grp['n_ub_pxls'][0],
                                                   self.smear_band_grp['n_ub_pxls_sm'][0],
                                                   self.band_grp['gains'][0],
                                                   self.smear_band_grp['gains_sm'][0])
        self.unc_grp["b_DNL"][0] = self.calc_b_DNL(self.unc_grp["a_DNL"][0],
                                                   X_cals)
        
        self.unc_grp["a_off"][0] = self.calc_a_off(self.rut_grp['digi_steps'][0],
                                                   self.rut_grp['OCL_steps'][0],
                                                   self.band_grp['n_ubands'][0],
                                                   self.band_grp['n_pxls'][0],
                                                   self.smear_band_grp['n_pxls_sm'][0],
                                                   self.band_grp['gains'][0],
                                                   self.smear_band_grp['gains_sm'][0])
        self.unc_grp["b_off"][0] = self.calc_b_off(self.unc_grp["a_off"][0],
                                                   X_cals)
        
        self.unc_grp["a_darkstab"][0] = self.calc_a_darkstab(self.band_grp['n_pxls'][0],
                                                             self.smear_band_grp['n_pxls_sm'][0],
                                                             self.band_grp['gains'][0],
                                                             self.smear_band_grp['gains_sm'][0])
        
        self.unc_grp["a_PS"][0] = self.calc_a_PS(self.band_grp['gains'][0],
                                                 self.band_grp['n_ub_pxls'][0],
                                                 self.smear_band_grp['gains_sm'][0],
                                                 self.smear_band_grp['n_ub_pxls_sm'][0],
                                                 self.band_grp['n_ubands'][0])
        self.unc_grp["b_PS"][0] = self.calc_b_PS(self.unc_grp["a_PS"][0],
                                                 X_cals)

        self.unc_grp["a_SGR"][0] = self.calc_a_SGR(self.band_grp['gain_nums'][0],
                                                   self.band_grp['n_pxls'][0],
                                                   self.smear_band_grp['n_pxls_sm'][0])
        self.unc_grp["b_SGR"][0] = self.calc_b_SGR(X_cals_sm,
                                                   X_cals)
        self.unc_grp["c_SGR"][0] = self.calc_c_SGR(self.band_grp['gain_nums'][0],
                                                   self.band_grp['n_pxls'][0],
                                                   self.smear_band_grp['n_pxls_sm'][0])

        self.unc_grp["a_INL"][0] = self.calc_a_INL()
        self.unc_grp["b_INL"][0] = self.calc_b_INL(self.band_grp['n_pxls'][0],
                                                   self.smear_band_grp['n_pxls_sm'][0],
                                                   self.band_grp['gains'][0],
                                                   self.smear_band_grp['gains_sm'][0])
        self.unc_grp["c_INL"][0] = self.calc_c_INL(X_cals_sm,
                                                   X_cals)
        self.unc_grp["d_INL"][0] = self.calc_d_INL(X_cals_tot,
                                                   X_cals)
        self.unc_grp["a_noise"][0] = self.calc_a_noise(self.band_grp['gain_nums'][0])

        return 0
    
    def save_to_python(self, directory):
        def write_grp(f, grp):
            '''write described data to file'''
            for i, vari in enumerate(grp.iterkeys()):
                if i==0:
                    f.write("'''"+grp[vari][0]+"'''\n")
                else:
                    f.write("#"+grp[vari][1]+" ("+grp[vari][2]+")"+"\n")
                    f.write(vari+" = "+repr(grp[vari][0]))
                    f.write("\n\n")
            f.write("\n")
            return 0

        #set filename to save to
        fname = 's3_olci_l1_rad_conf.py'
        path = join(directory, fname)

        #header to python document
        doc_string = '"""\nContains Sentinel-3 OLCI Level-1 radiometric configuration\n"""\n\n'
        imports = "'''Import Modules'''\nfrom numpy import array\n\n"
        refs = {'SCCDB': self.SCCDB}

        print 'Saving data to file: '+path
        with open(path, 'w') as f:
            #write document header
            f.write(doc_string)
            f.write(imports)
            f.write("'''References'''\n")
            for ref in refs:
                f.write("# > "+ref+': '+refs[ref]+"\n\n")

            #write variables
            write_grp(f, self.band_grp)
            write_grp(f, self.smear_band_grp)
            write_grp(f, self.CCD_grp)
            write_grp(f, self.time_grp)
            write_grp(f, self.rut_grp)
            write_grp(f, self.unc_grp)
            f.close()
        return 0
    
    def save_to_netcdf(self, directory):
        def write_grp(rootgrp, grp):
            '''write described data to file'''
            for i, vari in enumerate(grp.iterkeys()):
                if i==0:
                    new_grp = rootgrp.createGroup(grp[vari][1])
                else:
                    if grp[vari][3]==None:
                        new_vari = new_grp.createVariable(vari, "f4")
                    else:
                        new_vari = new_grp.createVariable(vari, "f4", grp[vari][3])
                    new_vari[:] = grp[vari][0]
                    new_vari.long_name = grp[vari][1]
                    new_vari.units = grp[vari][2]
            return 0

        #set filename to save to
        fname = "TEST_OLCI_RUT_DB_"+str(date.today())+".nc"
        path = join(directory, fname)
        
        print 'Saving data to file: '+path
        
        # Create file and band and module dimensions
        rootgrp = Dataset(path, "w", format="NETCDF4")
        band = rootgrp.createDimension("band", 21)
        mod = rootgrp.createDimension("mod", 5)

        write_grp(rootgrp, self.band_grp)
        write_grp(rootgrp, self.smear_band_grp)
        write_grp(rootgrp, self.CCD_grp)
        write_grp(rootgrp, self.time_grp)
        write_grp(rootgrp, self.rut_grp)
        write_grp(rootgrp, self.unc_grp)

        return 0

    def get_n_uband(self, prog_grp):
        # Initialise list to contain number of ubands in each observational 
        # band and a variable to contain number of ubands in the smear band
        
        microband_band = prog_grp.groups['olci_microband_band_settings']\
                                 .variables['olci_microband_band'][:]
        
        n_uband = [0]*21
        n_uband_sm = 0
        
        # count number of ubands in smear and and observational bands
        for num in microband_band:
            if num == 1:
                n_uband_sm = n_uband_sm + 1
            elif num > 1:
                n_uband[22-num] = n_uband[22-num] + 1
            
        return n_uband, n_uband_sm
    
    def calc_n_ub_pxl(self, prog_grp, n_uband):
        # Initialise list to contain the number of pixels in each microband
        
        line_microband = prog_grp.groups['olci_microband_band_settings']\
                                 .variables['olci_CCD_line_microband'][:,:]
        microband_band = prog_grp.groups['olci_microband_band_settings']\
                                 .variables['olci_microband_band'][:]
        used_ub_mask = [0 if i==0 else 1 for i in microband_band]
        
        n_ub_pxl = zeros((5,21))
        n_ub_pxl_sm = zeros(5)
        
        for mod in range(5):
            
            n_ub_pxl_all = [0]*45
            
            # Count number of pixels per microband
            for num in line_microband[mod,:]:
                if num > 0:
                    n_ub_pxl_all[num-1] = n_ub_pxl_all[num-1] + 1
            
            # Calculate per microband per band from all the microbands
            n_ub_pxl[mod,:], n_ub_pxl_sm[mod] = self.microband_to_band(n_ub_pxl_all, 
                                                                         used_ub_mask,
                                                                         n_uband)
        return n_ub_pxl, n_ub_pxl_sm
    
    def calc_n_pxl(self, n_uband, n_uband_sm, n_ub_pxl, n_ub_pxl_sm):
        # Calc number of pixels per band by multiplying the number of pixels
        # in a band microband by the number of microbands in the band
        
        n_pxl = zeros((5,21))
        n_pxl_sm = zeros(5)
        
        for mod in range(5):
            for i, (n_u, n_u_p) in enumerate(zip(n_uband, n_ub_pxl[mod,:])):
                n_pxl[mod,i] = n_u*n_u_p
            n_pxl_sm[mod] = n_uband_sm * n_ub_pxl_sm[mod]

        return n_pxl, n_pxl_sm
    
    def calc_gain_num(self, prog_grp, n_uband):
        # Calculate per microband per band from the data for all microbands
        microband_band = prog_grp.groups['olci_microband_band_settings']\
                                 .variables['olci_microband_band'][:]
        gain_microband = prog_grp.groups['olci_microband_gain_settings']\
                                 .variables['olci_VAM_prog_gain_microband'][:,:]
        used_ub_mask = [0 if i==0 else 1 for i in microband_band]
        gain_num = zeros((5,21))
        gain_num_sm = zeros(5)
        
        for mod in range(5):
            gain_num[mod,:], gain_num_sm[mod] = self.microband_to_band(gain_microband[mod,:],
                                                                       used_ub_mask,
                                                                       n_uband)
        return gain_num.astype(int), gain_num_sm.astype(int)
        
    def microband_to_band(self, x_ub_all, used_ub_mask, n_uband):
        # Use microband use mask to remove ubands not used in bands
        x_ub_all = [x for mask, x in zip(used_ub_mask, x_ub_all) if mask == 1]
         
        # Save smear band separately and remove from total list
        x_ub_sm = x_ub_all[0]
        x_ub_all = list(reversed(x_ub_all[n_uband[0]-1:]))

        # Select one microband number per band
        x_ub = []
        idx = 0
        for num in n_uband:
            x_ub.append(x_ub_all[idx])
            idx = idx + num
        
        return x_ub, x_ub_sm    

    def calc_gains(self, char_grp, gain_nums, gain_nums_sm):
        FPA_gains = char_grp.groups['olci_gain_values']\
                            .variables['olci_FPA_preamplifier_gain'][:]
        VAM_gains = char_grp.groups['olci_gain_values']\
                            .variables['olci_VAM_prog_gain'][:, :]

        gains = zeros((len(FPA_gains),len(gain_nums[0,:])))
        gains_sm = zeros(len(FPA_gains))
        
        for i, (FPA_gain, VAM_gain, gain_num_m, gain_num_sm) in \
        enumerate(zip(FPA_gains, VAM_gains, gain_nums, gain_nums_sm)):
            gain_lookup_mod = FPA_gain*VAM_gain
            for j, gain_num in enumerate(gain_num_m):
                gains[i, j] = gain_lookup_mod[gain_num]
            gains_sm[i] = gain_lookup_mod[gain_num_sm]

        return gains, gains_sm
    
    def calc_digi_steps(self, char_grp, gains, gains_sm):
        ADC_inputs = char_grp.groups['olci_ADC']\
                             .variables['olci_ADC_saturation_limit'][:]                         
        bit_num = char_grp.groups['olci_ADC']\
                          .variables['olci_ADC_bits_number'][:]
    
        conv_factors = char_grp.groups['olci_gain_values']\
                                .variables['olci_CCD_conversion_factor'][:]\
                                /1000000.0
        
        digi_steps = zeros((len(ADC_inputs), len(gains[0,:])))
        digi_steps_sm = zeros(len(conv_factors))
        for mod, gain_mod, gain_sm in zip(range(len(ADC_inputs)), gains, gains_sm):
            for i, gain in enumerate(gain_mod):
                digi_steps[mod, i] = ADC_inputs[mod]/((2**bit_num-1)* \
                                     gain*conv_factors[mod])
            
            digi_steps_sm[mod] = ADC_inputs[mod]/((2**bit_num-1)*gain_sm \
                                 *conv_factors[mod])

        return digi_steps, digi_steps_sm
    
    def calc_OCL_steps(self, digi_steps, digi_steps_sm, gain_nums, gain_nums_sm):
        OCL_steps = zeros((5,21))
        OCL_steps_sm = zeros(5)
        
        OCL_step_lookup = [0.19, 0.23, 0.27, 0.33, 0.39, 0.47, 0.57, 0.68]  
        
        for mod in range(5):
            for i, gain_num in enumerate(gain_nums[mod,:]):
                OCL_steps[mod,i] = OCL_step_lookup[gain_num]*digi_steps[mod,i]
            OCL_steps_sm[mod] = OCL_step_lookup[gain_nums_sm[mod]]
            
        return OCL_steps, OCL_steps_sm
    
    def calc_DS(self, char_grp, ls, t_frame):
        """Return value of dark signal for given line."""
        dark_dens = char_grp.groups['olci_CCD_dark_current']\
                            .variables['olci_dark_current_mean_density'][:]        
        #bands case
        if ls.shape[-1]==21:
            DS = zeros((5, ls.shape[-1]))
            for mod in range(5):
                for i, l in enumerate(ls[mod,:]):
                    C_l = 1 + (l - L_MAX)/(L_MIN - L_MAX)
                    DS[mod, i] = (dark_dens[mod] * PXL_AREA * t_frame / Q) * C_l
        #smear case
        else:
            DS = zeros(5)
            for mod in range(5):
                C_l = 1 + (ls[mod] - L_MAX)/(L_MIN - L_MAX)
                DS[mod] = (dark_dens[mod] * PXL_AREA * t_frame / Q) * C_l
    
        return DS
    
    def calc_C_RS(self, char_grp, t_int):

        focal_lengths = char_grp.groups['olci_focal_plane_geometry']\
                                .variables['olci_focal_length'][:]
        pupil_areas = char_grp.groups['olci_focal_plane_geometry']\
                              .variables['olci_pupil_area'][:]
        Gs = self.calc_G(PXL_SIZE, pupil_areas, focal_lengths)
        
        Ts = self.calc_T(char_grp)
        
        C_RSs = zeros((5,21))
        for mod in range(5):
            C_RSs[mod, :] = Ts[mod]*Gs[mod]*10**-3*t_int*DELTA_L/Q

        return C_RSs
    
    def calc_T(self, char_grp):
        T = zeros((5, 21))
        
        for mod in range(5):
            T1_l_T1 = char_grp.groups['olci_transmission']\
                              .variables['olci_T1_transmission'][mod,:]
            l_T1 = char_grp.groups['olci_transmission']\
                           .variables['olci_T1_wavelengths'][:]
    
            T2_l_T2 = char_grp.groups['olci_transmission']\
                              .variables['olci_T2_transmission'][mod,:]
            l_T2 = char_grp.groups['olci_transmission']\
                           .variables['olci_T2_wavelengths'][:]
    
            Rccd_l_R = char_grp.groups['olci_CCD_spectral_responsivity']\
                               .variables['olci_CCD_spectral_responsivity'][mod,:]\
                              /1000.0
            l_R = char_grp.groups['olci_CCD_spectral_responsivity']\
                          .variables['olci_CCD_spectral_responsivity_wavelengths'][:]

            if l_T1 != self.band_grp['ls'][0][mod]:
                T1s = self.interp_for_bands(T1_l_T1, l_T1, asarray(self.band_grp['ls'][0][mod]))
            else:
                T1s = T1_l_T1
    
            if l_T2 != list(self.band_grp['ls'][0][mod]):
                T2s = self.interp_for_bands(T2_l_T2, l_T2, asarray(self.band_grp['ls'][0][mod]))
            else:
                T2s = T2_l_T2
    
            if l_R[1:] != list(self.band_grp['ls'][0][mod]):
                Rccds = self.interp_for_bands(Rccd_l_R, l_R, asarray(self.band_grp['ls'][0][mod]))
            else:
                Rccds = Rccd_l_R
    
            T[mod, :] = [T1*T2*Rccd for T1, T2, Rccd in zip(T1s, T2s, Rccds)]
            
        return T
    
    def calc_G(self, pxl_size, pupil_areas, focal_lengths):
        G = [p*pxl_size**2/f**2 for p, f in zip(pupil_areas, focal_lengths)]
        return G
    
    def calc_u_ccdstabs_2(self, ls):
        """Return abosulute radiometric uncertainty contribution caused by
           CCD response stability error        
        """

        # Constants for fourth order polynomial describing the effect       
        C4 = 1.096690E-13
        C3 = -2.285318E-10
        C2 = 1.838631E-07
        C1 = -6.536942E-05
        C0 = 7.324642E-03

        # Calculate polynomial at each band wavelength
        u_ccdstabs = zeros((5,21))
        for mod in range(5):
            for band, l in enumerate(ls[mod,:]):
                u_ccdstab = 0.15*(C4*l**4 + C3*l**3 + C2*l**2 + C1*l + C0)
                u_ccdstabs[mod, band] = u_ccdstab
        return (u_ccdstabs/(3**0.5))**2

    def calc_u_diffage(self, ls, a):
        """Return absolute radiometric uncertainty contribution caused by
           diffuser aging (constant a governs the extent of the aging)  
        """

        # > MERIS Diffuser aging (%/yr) to compute a_diff1 for custom band settings
        aging = [0.0020, 0.0015, 0.0009, 0.0007, 0.0004, 0.0002, 0.0001]
        l_diff = [410, 445, 490, 520, 560, 620, 700]
        
        # Calculate aging per band
        u_diffages = zeros((5,21))
        for mod in range(5):
            for band, l in enumerate(ls[mod,:]):
                if l < 700:         
                    u_diffage = self.interp_for_bands(aging, l_diff, l)*a
                else:
                    u_diffage = 0.0   
                u_diffages[mod, band] = u_diffage
    
        return u_diffages
    
    def calc_a_DNL(self, n_ubands, n_ub_pxls, n_ub_pxls_sm, gains, gains_sm):
        a_DNL = zeros((5,21))
        
        for mod, (gain_sm, n_ub_pxl_sm) in enumerate(zip(gains_sm, n_ub_pxls_sm)):
            for band, (n_ub, n_ub_pxl, gain) in enumerate(zip(n_ubands, n_ub_pxls[mod,:], gains[mod,:])):
                a_DNL[mod, band] = E_DNL**2 * n_ub * (1 + ((gain/gain_sm)*(n_ub_pxl/n_ub_pxl_sm))**2)

        return a_DNL
    
    def calc_b_DNL(self, a_DNLs, X_cals):
        b_DNL = zeros((5,21))
                
        for mod in range(5):
            for band, (a_DNL, X_cal) in enumerate(zip(a_DNLs[mod, :], X_cals[mod])):
                b_DNL[mod, band] = a_DNL * X_cal**-2

        return b_DNL
    
    def calc_a_off(self, DSs, OCL_steps, n_ubands, n_pxls, n_pxls_sm, gains, gains_sm):
        a_off = zeros((5,21))
        
        for mod, (gain_sm, n_pxl_sm) in enumerate(zip(gains_sm, n_pxls_sm)):
            for band, (n_ub, n_pxl, gain, DS, OCL_step) in enumerate(zip(n_ubands, n_pxls[mod,:], gains[mod,:], DSs[mod, :], OCL_steps[mod, :])):
                a_off[mod, band] = (OCL_step**2/4*DS**2) * n_ub * (1 + ((gain/gain_sm)*(n_pxl/n_pxl_sm))**2)

        return a_off
    
    def calc_b_off(self, a_offs, X_cals):
        b_off = zeros((5,21))
                
        for mod in range(5):
            for band, (a_off, X_cal) in enumerate(zip(a_offs[mod, :], X_cals[mod])):
                b_off[mod, band] = a_off * X_cal**-2

        return b_off
    
    def calc_a_darkstab(self, n_pxls, n_pxls_sm, gains, gains_sm):
        
        a_darkstab = zeros((5,21))
        
        #Constant
        c = ((I*0.000000000001*0.00225**2/Q*0.044)*(exp(0.053*0.115)-1))**2
        
            
        for mod, (gain_sm, n_pxl_sm) in enumerate(zip(gains_sm, n_pxls_sm)):
            for band, (n_pxl, gain) in enumerate(zip(n_pxls[mod,:], gains[mod,:])):
                a_darkstab[mod, band] = c * n_pxl**2 * (1 + ((gain/gain_sm)*(n_pxl/n_pxl_sm))**2)
        
        return a_darkstab
    
    def calc_a_PS(self, gains, n_ub_pxls, gains_sm , n_ub_pxls_sm, n_ubands):
        a_PS = zeros((5,21))
    
                #Peak to peak signal amplitude of PS by band (+smear band)
        Im_xs = [0.412233784, 0.566232432, 0.515390541, 0.449536486, 
                 0.482039189, 0.4157, 0.560993243, 0.378393243, 0.281418919,
                 0.420910811, 0.414306757, 0.33717027, 0.317059459, 
                 0.321048649, 0.48297027, 0.443473874, 0.379437838, 
                 0.399737838, 0.825648649, 0.948910811, 0.873464865]
        p2p_sm = 1.494413514
        
        for mod, (gain_sm, n_ub_pxl_sm) in enumerate(zip(gains_sm, n_ub_pxls_sm)):
            for band, (n_ub, Im_x, n_ub_pxl, gain) in enumerate(zip(n_ubands, Im_xs, n_ub_pxls[mod,:], gains[mod,:])):
                Im_sm = p2p_sm * (gain/gain_sm)/(n_ub_pxl/n_ub_pxl_sm)
                a_PS[mod, band] = ((n_ub**0.5 * Im_x)**2 * (n_ub * Im_sm)**2) / 3

        return a_PS
    
    def calc_b_PS(self, a_PSs, X_cals):
        b_PS = zeros((5,21))
                
        for mod in range(5):
            for band, (a_PS, X_cal) in enumerate(zip(a_PSs[mod, :], X_cals[mod])):
                b_PS[mod, band] = a_PS * X_cal**-2
        return b_PS
    
    def calc_a_SGR(self, gain_nums, n_pxls, n_pxls_sm):

        a_SGR = zeros((5,21))
        
        # Input data - by gain number
        # Gain ratio g/g(7)
        gR = [0.2796, 0.3356, 0.4026, 0.4829, 0.579, 0.6949, 0.8336, 1]
        # Internal gain ratio dispersion
        D = [0.0011, 0.0011, 0.001, 0.001, 0.0005, 0.0003, 0.0002, 0] 
        # Aging
        A = [0.0023, 0.0022, 0.0021, 0.002, 0.0009, 0.0006, 0.0004, 0]
        for mod, n_pxl_sm in enumerate(n_pxls_sm):
            for band, (n_pxl, gain_num) in enumerate(zip(n_pxls[mod,:], gain_nums[mod,:])):
                a_SGR[mod, band] = (gR[gain_num]*n_pxl/n_pxl_sm)**2*(D[gain_num]**2+A[gain_num]**2)        
        
        return a_SGR
    
    def calc_b_SGR(self, X_cals_sm, X_cals):
        b_SGR = zeros((5,21))
                
        for mod, X_cal_sm in enumerate(X_cals_sm):
            for band, X_cal in enumerate(X_cals[mod]):
                b_SGR[mod, band] = X_cal_sm/X_cal

        return b_SGR
    
    def calc_c_SGR(self, gain_nums, n_pxls, n_pxls_sm):
        
        c_SGR = zeros((5,21))
        
        # Input data - by gain number
        # Gain ratio g/g(7)
        gR = [0.2796, 0.3356, 0.4026, 0.4829, 0.579, 0.6949, 0.8336, 1]
        # Internal gain ratio dispersion
        T = [0.0007, 0.0007, 0.0007, 0.0006, 0.0003, 0.0002, 0.0001, 0]
        for mod, n_pxl_sm in enumerate(n_pxls_sm):
            for band, (n_pxl, gain_num) in enumerate(zip(n_pxls[mod,:], gain_nums[mod,:])):
                c_SGR[mod, band] = (gR[gain_num]*n_pxl/n_pxl_sm)**2*T[gain_num]**2      
        
        return c_SGR
    
    def calc_a_INL(self):
        return E_INL**2
    
    def calc_b_INL(self, n_pxls, n_pxls_sm, gains, gains_sm):
        
        b_INL= zeros((5,21))

        for mod, (gain_sm, n_pxl_sm) in enumerate(zip(n_pxls_sm, gains_sm)):
            for band, (n_pxl, gain) in enumerate(zip(n_pxls[mod,:], gains[mod,:])):
                b_INL[mod, band] = (gain/gain_sm)*(n_pxl/n_pxl_sm)
        return b_INL
    
    def calc_c_INL(self, X_cals_sm, X_cals):
        c_INL = zeros((5,21))
                
        for mod, X_cal_sm in enumerate(X_cals_sm):
            for band, X_cal in enumerate(X_cals[mod]):
                c_INL[mod, band] = X_cal_sm/X_cal

        return c_INL
    
    def calc_d_INL(self, X_cals_tot, X_cals):
        d_INL = zeros((5,21))
                
        for mod in range(5):
            for band, (X_cal_tot, X_cal) in enumerate(zip(X_cals_tot[mod], X_cals[mod])):
                d_INL[mod, band] = X_cal_tot/X_cal

        return d_INL
    
    def calc_a_noise(self, gain_nums):
        a_noise = zeros((5,21))
        
        for mod in range(5):
            for band, gain_num in enumerate(gain_nums[mod,:]):
                a_noise[mod, band] = DARK_NOISE[gain_num]

        return a_noise
        
    def interp_for_bands(self, F, l_f, l):
        """Take function of wavelength range and return interpolated
        values for given wavelength range.
        """
        func = InterpolatedUnivariateSpline(l_f, F, k=1)
        Z = func(l)
        return Z 
    
if __name__ == '__main__':

    def main():
        # directory of CCDBs
        directory_CCDB = '/home/shunt/data/CCDB/S3A_OL_1_CCDB___20160425T095210_20991231T235959_20160523T163900_PRODUCTION'
        directory_save = '/home/shunt/src/quality_indicator/OLCI_RUT_admin'

        # Open data
        print 'Opening CCDB data...'
        data = CCDB_Data(directory_CCDB)

        # Save data to NetCDF file to be used by OLCI_RUT
        #data.save_to_netcdf('/home/shunt/src/quality_indicator/OLCI_RUT_admin')
        data.save_to_python('/home/shunt/src/quality_indicator/snap-s3-olci-rut/src/main/python')
        
        print 'Done'

        return 0

    main()