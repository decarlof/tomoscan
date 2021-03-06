import epics
def scan_demo(tomo_prefix, exposure_time, scan_pv, start, step, points):
    """Demonstrates collecting a series of tomography datasets while scanning an EPICS PV.

       Parameters
       ----------
       tomo_prefix : str
          The EPICS PV prefix for the tomoScan database
       exposure_time : float
          The exposure time per projection for the tomography datasets
       scan_pv : str
          The name of the EPICS PV to scan
       start : float
          The starting value for the scanned PV
       step : float
          The step size for the scanned PV
       points : int
          The number of points in the scan
    """

    epics.caput(tomo_prefix + 'ExposureTime', exposure_time, wait=True)
    file_plugin_prefix = epics.caget(tomo_prefix + 'FilePluginPVPrefix')
    # Set the initial file number back to 1 and make sure AutoIncrement is enables
    epics.caput(file_plugin_prefix + 'FileNumber', 1)
    epics.caput(file_plugin_prefix + 'AutoIncrement', 'Yes')

    for i in range(1, points+1):
        epics.caput(scan_pv, start + step*i, wait=True)
        epics.caput(tomo_prefix + 'StartScan', 1, wait=True, timeout=100)
        print('Completed dataset %s' % epics.caget(file_plugin_prefix + 'FullFileName_RBV', as_string=True))
