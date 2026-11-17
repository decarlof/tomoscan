"""Software for tomography step scanning with EPICS at APS beamline 2-BM

   Classes
   -------
   TomoScan2BM
     Derived class for tomography step scanning with EPICS at APS beamline 2-BM
"""
import time
import os
import sys
from pathlib import Path
import h5py 
import traceback
import numpy as np

from tomoscan import data_management as dm
from tomoscan.tomoscan import TomoScan
from tomoscan.tomoscan_step import TomoScanSTEP
from tomoscan import log

EPSILON = .001

class TomoScan2BMSTEP(TomoScanSTEP):
    """Derived class used for tomography scanning with EPICS at APS beamline 2-BM

    Parameters
    ----------
    pv_files : list of str
        List of files containing EPICS pvNames to be used.
    macros : dict
        Dictionary of macro definitions to be substituted when
        reading the pv_files
    """

    def __init__(self, pv_files, macros):
        super().__init__(pv_files, macros)

        # Random sample shifts.  These are set in begin_scan()
        self.random_shift = False
        self.shifts = None
        self.sample_in_x = None
        self.sample_in_y = None

        # set TomoScan xml files
        self.epics_pvs['CamNDAttributesFile'].put('TomoScanDetectorAttributes.xml')
        self.epics_pvs['FPXMLFileName'].put('TomoScanLayout.xml')
        macro = 'DET=' + self.pv_prefixes['Camera'] + ',' + 'TS=' + self.epics_pvs['Testing'].__dict__['pvname'].replace('Testing', '', 1)
        self.control_pvs['CamNDAttributesMacros'].put(macro)

        # Enable auto-increment on file writer
        self.epics_pvs['FPAutoIncrement'].put('Yes')

        # # Set standard file template on file writer
        # self.epics_pvs['FPFileTemplate'].put("%s%s_%3.3d.h5", wait=True)

        # Disable over writing warning
        self.epics_pvs['OverwriteWarning'].put('Yes')

        # Set AD plugins            
        self.epics_pvs['PVANDArrayPort'].put('OVER1')
        self.epics_pvs['PVAEnableCallbacks'].put('Enable')
        self.epics_pvs['ROIEnableCallbacks'].put('Disable')
        self.epics_pvs['CBEnableCallbacks'].put('Disable')
        self.epics_pvs['FPEnableCallbacks'].put('Enable')  


        log.setup_custom_logger("./tomoscan.log")

    def open_frontend_shutter(self):
        """Opens the shutters to collect flat fields or projections.

        This does the following:

        - Checks if we are in testing mode. If we are, do nothing else opens the 2-BM front-end shutter.

        """
        if self.epics_pvs['Testing'].get():
            log.warning('In testing mode, so not opening shutters.')
        else:
            # Open 2-BM front-end shutter
            if not self.epics_pvs['OpenShutter'] is None:
                pv = self.epics_pvs['OpenShutter']
                value = self.epics_pvs['OpenShutterValue'].get(as_string=True)
                status = self.epics_pvs['ShutterStatus'].get(as_string=True)
                log.info('shutter status: %s', status)
                log.info('open shutter: %s, value: %s', pv, value)
                self.epics_pvs['OpenShutter'].put(value, wait=True)
                self.wait_frontend_shutter_open()
                # self.wait_pv(self.epics_pvs['ShutterStatus'], 1)
                status = self.epics_pvs['ShutterStatus'].get(as_string=True)
                log.info('shutter status: %s', status)

    def open_shutter(self):
        """Opens the shutters to collect flat fields or projections.

        This does the following:

        - Opens the 2-BM fast shutter.
        """

        # Open 2-BM fast shutter
        if not self.epics_pvs['OpenFastShutter'] is None:
            pv = self.epics_pvs['OpenFastShutter']
            value = self.epics_pvs['OpenFastShutterValue'].get(as_string=True)
            log.info('open fast shutter: %s, value: %s', pv, value)
            self.epics_pvs['OpenFastShutter'].put(value, wait=True)
            log.warning("Wait 2s  - Temporarily while there is no fast shutter at 2bmb ")
            time.sleep(2)

    def close_frontend_shutter(self):
        """Closes the shutters to collect dark fields.
        This does the following:

        - Closes the 2-BM front-end shutter.

        """
        if self.epics_pvs['Testing'].get():
            log.warning('In testing mode, so not opening shutters.')
        else:
            # Close 2-BM front-end shutter
            if not self.epics_pvs['CloseShutter'] is None:
                pv = self.epics_pvs['CloseShutter']
                value = self.epics_pvs['CloseShutterValue'].get(as_string=True)
                status = self.epics_pvs['ShutterStatus'].get(as_string=True)
                log.info('shutter status: %s', status)
                log.info('close shutter: %s, value: %s', pv, value)
                self.epics_pvs['CloseShutter'].put(value, wait=True)
                self.wait_pv(self.epics_pvs['ShutterStatus'], 1)  # ON = blocking = beam off
                status = self.epics_pvs['ShutterStatus'].get(as_string=True)
                log.info('shutter status: %s', status)

    def close_shutter(self):
        """Closes the shutters to collect dark fields.
        This does the following:

        - Closes the 2-BM fast shutter.
        """

        # Close 2-BM fast shutter
        if not self.epics_pvs['CloseFastShutter'] is None:
            pv = self.epics_pvs['CloseFastShutter']
            value = self.epics_pvs['CloseFastShutterValue'].get(as_string=True)
            log.info('close fast shutter: %s, value: %s', pv, value)
            self.epics_pvs['CloseFastShutter'].put(value, wait=True)
            log.warning("Wait 2s  - Temporarily while there is no fast shutter at 2bmb ")
            time.sleep(2)

    def set_trigger_mode(self, trigger_mode, num_images):
        """Sets the trigger mode SIS3820 and the camera.

        Parameters
        ----------
        trigger_mode : str
            Choices are: "FreeRun", "Internal", or "Software"

        num_images : int
            Number of images to collect.  Ignored if trigger_mode="FreeRun".
            This is used to set the ``NumImages`` PV of the camera.
        """
        self.epics_pvs['CamAcquire'].put('Done') ###
        self.wait_pv(self.epics_pvs['CamAcquire'], 0) ###
        log.info('set trigger mode: %s', trigger_mode)
        if trigger_mode == 'FreeRun':
            self.epics_pvs['CamImageMode'].put('Continuous', wait=True)
            self.epics_pvs['CamTriggerMode'].put('Off', wait=True)
            self.wait_pv(self.epics_pvs['CamTriggerMode'], 0)
            # self.epics_pvs['CamAcquire'].put('Acquire')
        elif trigger_mode == 'Internal':
            self.epics_pvs['CamTriggerMode'].put('Off', wait=True)
            self.wait_pv(self.epics_pvs['CamTriggerMode'], 0)
            self.epics_pvs['CamImageMode'].put('Multiple')            
            self.epics_pvs['CamNumImages'].put(num_images, wait=True)
        else: # set camera to software triggering
            # The step scan triggers each projection with CamTriggerSoftware, so the
            # trigger source must be 'Software' for every camera model.  Only the order
            # in which TriggerMode is toggled is model specific.
            # These are just in case the scan aborted with the camera in another state
            camera_model = self.epics_pvs['CamModel'].get(as_string=True)
            if(camera_model=='Oryx ORX-10G-51S5M' or camera_model=='Oryx ORX-10G-310S9M'):
                self.epics_pvs['CamTriggerMode'].put('Off', wait=True)   # VN: For FLIR we first switch to Off and then change overlap. any reason of that?
            elif(camera_model=='Grasshopper3 GS3-U3-23S6M'):# 2bmb
                self.epics_pvs['CamTriggerMode'].put('On', wait=True)     # VN: For PG we need to switch to On to be able to switch to readout overlap mode
            self.epics_pvs['CamTriggerSource'].put('Software', wait=True)
            self.epics_pvs['CamTriggerOverlap'].put('ReadOut', wait=True)
            self.epics_pvs['CamExposureMode'].put('Timed', wait=True)
            self.epics_pvs['CamImageMode'].put('Multiple')            
            self.epics_pvs['CamArrayCallbacks'].put('Enable')
            self.epics_pvs['CamFrameRateEnable'].put(0)
            self.epics_pvs['CamNumImages'].put(num_images, wait=True)
            self.epics_pvs['CamTriggerMode'].put('On', wait=True)
            self.wait_pv(self.epics_pvs['CamTriggerMode'], 1)

    def begin_scan(self):
        """Performs the operations needed at the very start of a scan.

        This does the following:

        - Set data directory.

        - Set the TomoScan xml files

        - Calls the base class method.

        - Computes the random sample shifts.

        - Opens the front-end shutter.
        """
        log.info('begin scan')

        # Set data directory
        file_path = Path(self.epics_pvs['DetectorTopDir'].get(as_string=True))
        file_path = file_path.joinpath(self.epics_pvs['ExperimentYearMonth'].get(as_string=True) + '-'
                                       + self.epics_pvs['UserLastName'].get(as_string=True) + '-'
                                       + self.epics_pvs['ProposalNumber'].get(as_string=True))
        self.epics_pvs['FilePath'].put(str(file_path), wait=True)

        # Call the base class method
        super().begin_scan()
        # The base class aborts the scan if the interlaced angle file is not valid
        if not self.scan_is_running:
            return
        # Compute the random sample shifts.  This must be done after the base class
        # method, which can change num_angles when reading an interlaced angle file.
        self.compute_random_shifts()
        # Opens the front-end shutter
        self.open_frontend_shutter()

    def compute_random_shifts(self):
        """Computes the random sample shift applied at each projection angle.

        The shifts are drawn from a uniform distribution over
        [-``RandomShiftAmplitude``, +``RandomShiftAmplitude``] in mm, independently for
        the SampleX and SampleY axes.  The shift of the first projection is always 0, so
        that projection is collected at the nominal ``SampleInX``/``SampleInY`` position.

        Sets ``self.shifts`` to a (num_angles, 2) array of offsets in mm, or to None if
        random shifts are disabled.
        """

        self.sample_in_x = self.epics_pvs['SampleInX'].value
        self.sample_in_y = self.epics_pvs['SampleInY'].value
        amplitude = self.epics_pvs['RandomShiftAmplitude'].get()
        self.random_shift = ((self.epics_pvs['RandomShift'].get(as_string=True) == 'Yes')
                             and (amplitude > 0))
        if self.random_shift:
            self.shifts = np.random.uniform(-amplitude, amplitude, (self.num_angles, 2))
            # Collect the first projection at the nominal position
            self.shifts[0] = 0.
            log.info('random sample shifts enabled, amplitude %f mm', amplitude)
        else:
            self.shifts = None

    def move_simultaneous(self, moves, timeout=600, stop_on_abort=True):
        """Starts all the moves at the same time and waits for all of them to finish.

        A blocking ``put(wait=True)`` on a motor does not return until that motion is
        complete, so moving several motors that way makes them move one after the other.
        This issues all the puts first with ``use_complete``, which uses the same
        ca_put_callback mechanism without blocking, and only then waits for every
        callback to arrive.  The motors therefore move at the same time and this returns
        when the slowest one has finished.

        Parameters
        ----------
        moves : list of (PV, float)
            The motor PVs to move and the positions to move them to.

        timeout : float
            Maximum number of seconds to wait for all the motions to complete.

        stop_on_abort : bool
            If True, stop waiting as soon as the scan is aborted.  Set this False for
            moves that must still complete during cleanup, when ``scan_is_running`` has
            already been cleared.

        Returns
        -------
        bool
            True if all the motions completed, False if the timeout was reached.
        """

        for epics_pv, position in moves:
            epics_pv.put(position, use_complete=True)
        start_time = time.time()
        while not all(epics_pv.put_complete for epics_pv, position in moves):
            if (time.time() - start_time) >= timeout:
                stalled = [epics_pv.pvname for epics_pv, position in moves
                           if not epics_pv.put_complete]
                log.error('  *** ERROR: motion timeout after %5.2f s on: %s',
                          timeout, ', '.join(stalled))
                return False
            if stop_on_abort and not self.scan_is_running:
                return False
            time.sleep(0.01)
        return True

    def collect_projections(self):
        """Collects projections in step scan mode, with random sample shifts.

        This is the base class method with the random sample shift applied to the
        SampleX and SampleY motors at each projection angle.  The shift is relative to
        the ``SampleInX``/``SampleInY`` position the sample is left at by
        ``move_sample_in()``.
        """

        log.info('collect projections')
        # Skip TomoScanSTEP.collect_projections(), this method replaces it
        TomoScan.collect_projections(self)

        self.set_trigger_mode('Software', self.num_angles)

        # Start the camera
        self.epics_pvs['CamAcquire'].put('Acquire')
        # Need to wait a short time for AcquireBusy to change to 1
        time.sleep(0.5)
        self.epics_pvs['HDF5Location'].put(self.epics_pvs['HDF5ProjectionLocation'].value)
        self.epics_pvs['FrameType'].put('Projection')

        start_time = time.time()
        stabilization_time = self.epics_pvs['StabilizationTime'].get()
        log.info("stabilization time %f s", stabilization_time)
        for k in range(self.num_angles):
            if(self.scan_is_running):
                log.info('angle %d: %f', k, self.theta[k])
                moves = [(self.epics_pvs['Rotation'], self.theta[k])]
                if self.random_shift:
                    log.info('shift %d: %f, %f', k, self.shifts[k, 0], self.shifts[k, 1])
                    moves.append((self.epics_pvs['SampleX'],
                                  self.sample_in_x + self.shifts[k, 0]))
                    moves.append((self.epics_pvs['SampleY'],
                                  self.sample_in_y + self.shifts[k, 1]))
                # Rotation, SampleX and SampleY all move at the same time
                self.move_simultaneous(moves)
                time.sleep(stabilization_time)
                self.epics_pvs['CamTriggerSoftware'].put(1)
                self.wait_pv(self.epics_pvs['CamNumImagesCounter'], k+1, 60)
                self.update_status(start_time)

        # wait until the last frame is saved (not needed)
        time.sleep(0.5)
        self.update_status(start_time)

    def end_scan(self):
        """Performs the operations needed at the very end of a scan.

        This does the following:

        - Reset rotation position by mod 360.

        - Undo the random sample shift.

        - Calls the base class method.

        - Stop the file plugin.

        - Closes shutter.  

        - Add theta to the raw data file. 

        - Copy raw data to data analysis computer.      
        """

        if self.return_rotation == 'Yes':
            # Reset rotation position by mod 360 , the actual return 
            # to start position is handled by super().end_scan()
            current_angle = self.epics_pvs['Rotation'].get() %360
            self.epics_pvs['RotationSet'].put('Set', wait=True)
            self.epics_pvs['Rotation'].put(current_angle, wait=True)
            self.epics_pvs['RotationSet'].put('Use', wait=True)
        # Undo the random sample shift.  The base class method calls move_sample_in(),
        # but that only moves the axes listed in FlatFieldAxis, so the other axis would
        # be left at the shift of the last projection.
        if self.random_shift and self.sample_in_x is not None:
            log.info('undo random sample shift')
            self.move_simultaneous([(self.epics_pvs['SampleX'], self.sample_in_x),
                                    (self.epics_pvs['SampleY'], self.sample_in_y)],
                                   stop_on_abort=False)
        # Call the base class method
        super().end_scan()
        # Close shutter
        self.close_shutter()
        # Stop the file plugin
        self.epics_pvs['FPCapture'].put('Done')
        self.wait_pv(self.epics_pvs['FPCaptureRBV'], 0)
        # Add theta in the hdf file
        self.add_theta()

        # Copy raw data to data analysis computer    
        if self.epics_pvs['CopyToAnalysisDir'].get():
            log.info('Automatic data trasfer to data analysis computer is enabled.')
            full_file_name = self.epics_pvs['FPFullFileName'].get(as_string=True)
            remote_analysis_dir = self.epics_pvs['RemoteAnalysisDir'].get(as_string=True)
            dm.scp(full_file_name, remote_analysis_dir)
        else:
            log.warning('Automatic data trasfer to data analysis computer is disabled.')

    def add_theta(self):
        """Add theta, and the random sample shifts if any, at the end of a scan.
        """
        log.info('add theta')

        full_file_name = self.epics_pvs['FPFullFileName'].get(as_string=True)
        if os.path.exists(full_file_name):
            try:
                f = h5py.File(full_file_name, "a")
                with f:
                    try:
                        if self.theta is not None:
                            # theta_ds = f.create_dataset('/exchange/theta', (len(self.theta),))
                            # theta_ds[:] = self.theta[:]

                            unique_ids = f['/defaults/NDArrayUniqueId']
                            shift_start = int(self.num_dark_fields > 0 and (self.dark_field_mode in ('Start', 'Both')))+ \
                                          int(self.num_flat_fields > 0 and (self.flat_field_mode in ('Start', 'Both')))                            

                            # find beginnings of sorted subarrays
                            # for [1,2,1,3,1,2,3,4,1,2] returns 0,2,4,8
                            ids_list = [0,*np.where(unique_ids[1:]-unique_ids[:-1]<0)[0]+1]                            
                            # extract projection ids
                            if(len(ids_list[shift_start:])==1):
                                proj_ids = unique_ids[ids_list[shift_start]:]
                            else:
                                proj_ids = unique_ids[ids_list[shift_start]:ids_list[shift_start+1]]
                            # subtract first id
                            proj_ids -= proj_ids[0]
                            # create theta dataset in hdf5 file
                            theta_ds = f.create_dataset('/exchange/theta', (len(proj_ids),))
                            theta_ds[:] = self.theta[proj_ids]

                            # Add the random sample shifts, indexed the same way as
                            # theta so a dropped frame drops its matching shift
                            if self.shifts is not None:
                                shifts_ds = f.create_dataset('/exchange/shifts', (len(proj_ids), 2))
                                shifts_ds[:] = self.shifts[proj_ids]
                                shifts_ds.attrs['units'] = 'mm'
                                shifts_ds.attrs['columns'] = 'sample_x, sample_y'

                            if(len(proj_ids) != len(self.theta)):
                                log.warning('There are %d missing frames',len(self.theta)-len(proj_ids))
                                missed_ids = [ele for ele in range(len(self.theta)) if ele not in proj_ids]
                                missed_theta = self.theta[missed_ids]
                                log.warning(f'Missed ids: {list(missed_ids)}')
                                log.warning(f'Missed theta: {list(missed_theta)}')
                                
                    except:
                        log.error('Add theta: Failed accessing: %s', full_file_name)
                        traceback.print_exc(file=sys.stdout)
            except OSError:
                log.error('Add theta aborted: %s not closed', full_file_name)
        else:
            log.error('Failed adding theta. %s file does not exist', full_file_name)

    def wait_pv(self, epics_pv, wait_val, timeout=-1):
        """Wait on a pv to be a value until max_timeout (default forever)
           delay for pv to change
        """

        time.sleep(.01)
        start_time = time.time()
        while True:
            pv_val = epics_pv.get()
            if isinstance(pv_val, float):
                if abs(pv_val - wait_val) < EPSILON:
                    return True
            if pv_val != wait_val:
                if timeout > -1:
                    current_time = time.time()
                    diff_time = current_time - start_time
                    if diff_time >= timeout:
                        log.error('  *** ERROR: DROPPED IMAGES ***')
                        log.error('  *** wait_pv(%s, %d, %5.2f reached max timeout. Return False',
                                      epics_pv.pvname, wait_val, timeout)
                        return False
                time.sleep(.01)
            else:
                return True

    def wait_frontend_shutter_open(self, timeout=-1):
        """Waits for the front end shutter to open, or for ``abort_scan()`` to be called.

        While waiting this method periodically tries to open the shutter..

        Parameters
        ----------
        timeout : float
            The maximum number of seconds to wait before raising a ShutterTimeoutError exception.

        Raises
        ------
        ScanAbortError
            If ``abort_scan()`` is called
        ShutterTimeoutError
            If the open shutter has not completed within timeout value.
        """

        start_time = time.time()
        pv = self.epics_pvs['OpenShutter']
        value = self.epics_pvs['OpenShutterValue'].get(as_string = True)
        log.info('open shutter: %s, value: %s', pv, value)
        elapsed_time = 0
        while True:
            if self.epics_pvs['ShutterStatus'].get() == 0:  # OFF = not blocking = beam on
                log.warning("Shutter is open in %f s", elapsed_time)
                return
            if not self.scan_is_running:
                exit()
            value = self.epics_pvs['OpenShutterValue'].get()
            time.sleep(1.0)
            current_time = time.time()
            elapsed_time = current_time - start_time
            log.warning("Waiting on shutter to open: %f s", elapsed_time)
            self.epics_pvs['OpenShutter'].put(value, wait=True)
            if timeout > 0:
                if elapsed_time >= timeout:
                   exit()
    
    def abort_scan(self):
        super().abort_scan()
        self.add_theta()
