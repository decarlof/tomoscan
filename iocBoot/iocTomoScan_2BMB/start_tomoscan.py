# This script creates an object of type TomoScan2BM for doing tomography scans at APS beamline 2-BM-A
# To run this script type the following:
#     python -i start_tomoscan_2bm.py
# The -i is needed to keep Python running, otherwise it will create the object and exit
from tomoscan.tomoscan_2bm import TomoScan2BM
import sys

req = {'TomoScanPSO':["../../db/tomoScan_settings.req",
                  "../../db/tomoScan_PSO_settings.req",                   
                  "../../db/tomoScan_2BM_settings.req"],
        'TomoScanHelical':["../../db/tomoScan_settings.req",
                  "../../db/tomoScan_PSO_settings.req", 
                  "../../db/tomoScan_Helical_settings.req", 
                  "../../db/tomoScan_2BM_settings.req"],
        'TomoScanStreamPSO':["../../db/tomoScan_settings.req",
                  "../../db/tomoScan_PSO_settings.req", 
                  "../../db/tomoScanStream_settings.req",
                  "../../db/tomoScan_2BM_settings.req"],
        
        'TomoScanSTEP':["../../db/tomoScan_settings.req",
                "../../db/tomoScan_step_settings.req",
                "../../db/tomoScan_2BM_settings.req"]
                  }

key = sys.argv[1]
ts = TomoScan2BM(key, req[key], {"$(P)":"2bmb:", "$(R)":"TomoScan:"})
