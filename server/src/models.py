import numpy as np
import os
import sys
import pandas as pd
from scripts_ML_FITTING_ALL.SCRIPTS.Eval_MLEmulator import EvalMLEmulator

class predictions:

    def __init__(self):
        self.emulator = EvalMLEmulator(library_basename="16par_RANDOM")
        self.x_axis = self.emulator.wavelength
    
    def load_models(self):
        models = list(self.emulator.load_best_model())
        return models

    def predict(self, inputs, models, graph_type):

        #StellarSEDmodel=StellarSEDmodel, DustSEDmodel=DustSEDmodel, SIZEmodel=SIZEmodel, SERSICmodel=SERSICmodel, Qmodel=Qmodel, Physicalmodel=Physicalmodel, Inboxmodel=Inboxmodel, inputs=None
        df_inputs = pd.DataFrame({'logMstar':[inputs[0]], 'logMdMs':[inputs[1]], 'theta':[inputs[2]], 'Rstar':[inputs[3]], 'CsRs':[inputs[4]], 'nstar':[inputs[5]], 'RdRs':[inputs[6]], 'CdRd':[inputs[7]], 'ndust':[inputs[8]], 'f_cov':[inputs[9]], 'Age':[inputs[10]], 't_peak':[inputs[11]], 'k_peak':[inputs[12]], 'fwhm':[inputs[13]], 'k_fwhm':[inputs[14]], 'metal':[inputs[15]]}) ## Vary logMstar
        output = list(self.emulator.eval_MLemulator_JZ(StellarSEDmodel=models[0], DustSEDmodel=models[1], SIZEmodel=models[2], SERSICmodel=models[3], Qmodel=models[4], Physicalmodel=models[5], Inboxmodel=models[6], inputs=df_inputs))[:4]
        
        # f-pred, r-pred, n-pred, q-pred, 

        if graph_type == "f":
            return(self.x_axis,output[0])
        elif graph_type == "r":
            return(self.x_axis,output[1])
        elif graph_type == "n":
            return(self.x_axis,output[2])
        elif graph_type == "q":
            return(self.x_axis,output[3])
        else:
            Exception("Invalid graph type")


        
        
        
