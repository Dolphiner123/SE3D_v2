# This script will be used to allow the Emulator to make predictions

# Imports
import numpy as np
import pandas as pd
import torch
import h5py
import pickle
import deepdish as dd # vvvvv special folder for this
import datetime
import sys
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import linalg
from scipy.ndimage import median_filter, gaussian_filter1d

# Torch Imports
#import torch
import torch.nn as nn
import torchbnn as bnn
import torch.nn.init as init
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.multiprocessing import Pool, Process, set_start_method
import torch.multiprocessing as tmp

############################################ CLASS WHICH INSTANTIATE OUR BAYESIAN NEURAL NETWORK ###################################################
# FOR JZ
class BayesLinear_JZ(bnn.BayesLinear):

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear_JZ, self).__init__(prior_mu, prior_sigma, in_features, out_features, bias)

    def forward(self, input):
        return F.linear(input, self.weight_mu, self.bias_mu)


# Our BNN class
class BayesianNN(nn.Module):
    # Define Constructor including BNN architecture
    def __init__(self, input_size, output_size, hidden_sizes, dropout_rate, mu, sig, activation, weight_init_name, bnn_active=False, emulator=None):
        super(BayesianNN, self).__init__()
        layers = []
        in_features = input_size  # Start with the input size
        #print(input_size, output_size)

        if bnn_active:
           BayesLinear = bnn.BayesLinear
        else:
           BayesLinear = BayesLinear_JZ

        #self.bnn_active = bnn_active

        #print(activation)
           
        # Create hidden Bayesian layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(BayesLinear(prior_mu=mu, prior_sigma=sig, in_features=in_features, out_features=hidden_size, bias=True))
            layers.append(nn.BatchNorm1d(hidden_size))
            #layers.append(bnn._BayesBatchNorm(hidden_size))
            layers.append(activation[i])
            layers.append(nn.Dropout(dropout_rate[i]))
            in_features = hidden_size  # Update in_features for the next layer to match current layer's out_features

        # Add the output Bayesian layer
        layers.append(BayesLinear(prior_mu=mu, prior_sigma=sig, in_features=in_features, out_features=output_size, bias=True))

        # Add sigmoid layer to the end if its a Physical ML we're training to squash outputs to 0 and 1
        if emulator == 'Physical':
            layers.append(nn.Sigmoid())

        # Add learnable weights to outputs for Inbox ML as some outputs are harder to predict than others!
        if emulator == 'Inbox' or emulator == 'SFH':
            self.param_weight = nn.Parameter(torch.zeros(output_size))

        
        # Build the model as a sequential container
        self.model = nn.Sequential(*layers)

        #print(self.model)

        # Apply weight initialization
        self.__initialize_weights__(weight_init_name)

    
    ###################################### FOWARD METHODS ###########################################
    def forward(self, x):
        return self.model(x)


    ###################################### HELPER METHODS ###########################################
    # Helper Method to initialise BNN weights
    def __initialize_weights__(self, weight_init_name):
        for m in self.model:
            if isinstance(m, bnn.BayesLinear):
                # Xavier initialization
                if weight_init_name == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight_mu)  # Initialize weight_mu
                    if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                        nn.init.constant_(m.bias_mu, 0)  # Initialize bias_mu to 0
                elif weight_init_name == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight_mu)  # Initialize weight_mu
                    if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                        nn.init.constant_(m.bias_mu, 0)  # Initialize bias_mu to 0
            
                # Kaiming initialization
                elif weight_init_name == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight_mu, nonlinearity='relu')
                    if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                        nn.init.constant_(m.bias_mu, 0)
                elif weight_init_name == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight_mu, nonlinearity='relu')
                    if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                        nn.init.constant_(m.bias_mu, 0)

                # Orthogonal initialization
                elif weight_init_name == 'orthogonal':
                    nn.init.orthogonal_(m.weight_mu, gain=1.0)  # Gain can be adjusted based on activations
                    if hasattr(m, 'bias_mu') and m.bias_mu is not None:
                        nn.init.constant_(m.bias_mu, 0)  # Initialize bias_mu to 0

            # If we're looking at batchnorm layer
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm1d initialization (same for all methods)
                nn.init.constant_(m.weight, 1)  # Initialize scale to 1
                nn.init.constant_(m.bias, 0)    # Initialize shift to 0

# Create BNN architecture for PhysicalGalaxiesEmulator
'''class PhysicalNN(nn.Module):
    # Constructor to initialise physical emulator model
    def __init__(self):
        super(PhysicalNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(15, 128),
                                   nn.GELU(),
                                   nn.BatchNorm1d(128),
                                   nn.Linear(128, 32),
                                   nn.GELU(),
                                   nn.BatchNorm1d(32),
                                   #nn.Linear(64, 32),
                                   #nn.ReLU(),
                                   #nn.BatchNorm1d(32),
                                   #nn.LeakyReLU(),
                                   nn.Linear(32, 1),
                                   nn.Sigmoid())

    # Forward Method
    def forward(self, x):
        return self.model(x) #torch.clip(self.model(x), 0, 1)'''


# Create BNN architecture for PhysicalGalaxiesEmulator
'''class InboxInputNN(nn.Module):
    # Constructor to initialise physical emulator model
    def __init__(self):
        super(InboxInputNN, self).__init__()
        self.model = nn.Sequential(nn.Linear(15, 128),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Linear(128, 32),
                                   nn.GELU(),
                                   nn.BatchNorm1d(32),
                                   #nn.Linear(64, 32),
                                   #nn.ReLU(),
                                   #nn.BatchNorm1d(32),
                                   #nn.LeakyReLU(),
                                   #nn.Linear(64, 32),
                                   #nn.GELU(),
                                   #nn.BatchNorm1d(32),
                                   nn.Linear(32, 15))#,
                                   #nn.Sigmoid())

    # Forward Method
    def forward(self, x):
        return self.model(x) #torch.clip(self.model(x), 0, 1)'''
    
######################## CLASS USED TO EVALUATE OUR EMULATOR ###############################
class EvalMLEmulator:
    # Define consturctor to initialise global variables and create an instance of our models
    def __init__(self, library_basename='16par', doing_pca=True, ref_norm=True, f_path_size=None):
        # Conditions
        self.library_basename = library_basename
        self.doing_pca = doing_pca # True for Junkai
        self.ref_norm = ref_norm # True for Junkai
        self.f_path_size = f_path_size

        # FELIX ADDED
        script_dir = os.path.dirname(__file__)
        self.skirt_file_path = os.path.join(script_dir, "../SKIRT_library")

        self.device = torch.device('cpu')
        '''try:
            set_start_method('spawn')

        except RuntimeError:
            pass'''

        # In features + Read flags
        columns_file_path = os.path.join(script_dir, 'ML_columns.h5') # FELIX ADDED
        
        self.table_ML_columns = pd.read_hdf(columns_file_path) #maybe changed
        flags = self.table_ML_columns['flags'].to_numpy()
        self.in_features = np.sum(flags)

        # F_path_size determiens if we are creating the Ntrain vs Nparam plot
        if f_path_size is not None:
            flags_df = pd.read_json(f_path_size + f'../{library_basename}_varying_flags.json')
            self.in_features = np.sum(flags_df.T.to_numpy()[0, :])

        # Load in Optuna best params
        with open(f'{self.skirt_file_path}/{self.library_basename}/out_ML/SE3D_Models/best_optuna/{self.library_basename}_optuna_best_params.txt', 'r') as f:
            loaded_data = json.load(f)

        # Extract Optuna best params + NMADs
        self.best_params_StellarSED, self.best_params_DustSED, self.best_params_SIZE, self.best_params_SERSIC, self.best_params_Q = loaded_data[0], loaded_data[1], loaded_data[2], loaded_data[3], loaded_data[4]

        # Load Optuna best params for Inbox ML
        with open(f'{self.skirt_file_path}/{self.library_basename}/out_ML/SE3D_Models/best_optuna/{self.library_basename}_Inbox_optuna_best_params.txt', 'r') as f:
            loaded_data = json.load(f)

        # Extract Optuna best params + NMADs
        self.best_params_Inbox = loaded_data[0]

        # Load Optuna best params for Physical ML
        with open(f'{self.skirt_file_path}/{self.library_basename}/out_ML/SE3D_Models/best_optuna/{self.library_basename}_Physical_optuna_best_params.txt', 'r') as f:
            loaded_data = json.load(f)

        # Extract Optuna best params + NMADs
        self.best_params_Physical = loaded_data[0]

        # Print
        print(f'Using Flux, Size, Sersic, and q models with NMADS: 0.049, 0.033, 0.058, 0.007')

        # Read in wvl
        self.wavelength = pd.read_hdf(f'{self.skirt_file_path}/{self.library_basename}/out_SKIRT/DATA/{self.library_basename}_wavelength.h5').to_numpy().flatten()[0]
        # x axis input vvvvv

        # Initialise all PCA data
        self.initialise()

    ####################################### PUBLIC METHODS ##########################################

    # Initialise files we need for ML predictions
    def initialise(self):
        if self.f_path_size is not None:
            print('Reading in params for Ntrain vs Nparam plot')
            # Extract components + mean for each model
            pca_inverse_flux_star = dd.io.load(self.f_path_size + f'../PCA_outputs/{self.library_basename}_PCA_inverse_flux_star.h5')
            pca_inverse_flux_dust = dd.io.load(self.f_path_size + f'../PCA_outputs/{self.library_basename}_PCA_inverse_flux_dust.h5')
            pca_inverse_size = dd.io.load(self.f_path_size + f'../PCA_outputs/{self.library_basename}_PCA_inverse_size.h5')
            pca_inverse_sersic = dd.io.load(self.f_path_size + f'../PCA_outputs/{self.library_basename}_PCA_inverse_sersic.h5')
            pca_inverse_q = dd.io.load(self.f_path_size + f'../PCA_outputs/{self.library_basename}_PCA_inverse_q.h5')
            
            # Read normalisation values for inputs and PCA basis coeffs using deepdish
            f = dd.io.load(self.f_path_size + f'../PCA_outputs/{self.library_basename}_PCA_normalisation_params.h5')
        else:
            #print('Reading actual values')
            #print(stop)
            # Extract components + mean for each model
            pca_inverse_flux_star = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/PCA_outputs/{self.library_basename}_PCA_inverse_flux_star.h5')
            pca_inverse_flux_dust = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/PCA_outputs/{self.library_basename}_PCA_inverse_flux_dust.h5')
            pca_inverse_size = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/PCA_outputs/{self.library_basename}_PCA_inverse_size.h5')
            pca_inverse_sersic = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/PCA_outputs/{self.library_basename}_PCA_inverse_sersic.h5')
            pca_inverse_q = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/PCA_outputs/{self.library_basename}_PCA_inverse_q.h5')
            
            # Read normalisation values for inputs and PCA basis coeffs using deepdish
            f = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/PCA_outputs/{self.library_basename}_PCA_normalisation_params.h5')

        # Get shape, components, and mean
        
        # StellarSED
        self.out_featuresStellarSED = pca_inverse_flux_star[f'pca_components'].shape[0] + 1
        self.pca_flux_star_components, self.pca_flux_star_mean = pca_inverse_flux_star[f'pca_components'], pca_inverse_flux_star[f'pca_mean']
        
        # DustSED
        self.out_featuresDustSED = pca_inverse_flux_dust[f'pca_components'].shape[0] + 1
        self.pca_flux_dust_components, self.pca_flux_dust_mean = pca_inverse_flux_dust[f'pca_components'], pca_inverse_flux_dust[f'pca_mean']
        
        # Size
        self.out_featuresSIZE = pca_inverse_size[f'pca_components'].shape[0] + 1
        self.pca_size_components, self.pca_size_mean = pca_inverse_size[f'pca_components'], pca_inverse_size[f'pca_mean']
        
        # Sersic
        self.out_featuresSERSIC = pca_inverse_sersic[f'pca_components'].shape[0] + 1
        self.pca_sersic_components, self.pca_sersic_mean = pca_inverse_sersic[f'pca_components'], pca_inverse_sersic[f'pca_mean']

        # Axial Ratio
        self.out_featuresQ = pca_inverse_q[f'pca_components'].shape[0] + 1
        self.pca_q_components, self.pca_q_mean = pca_inverse_q[f'pca_components'], pca_inverse_q[f'pca_mean']
 
        
        # Open the HDF5 file and access the datasets
        self.pca_fstar_mean = f['pca_weight_mean_flux_star']
        self.pca_fstar_std = f['pca_weight_std_flux_star']
        self.pca_fdust_mean = f['pca_weight_mean_flux_dust']
        self.pca_fdust_std = f['pca_weight_std_flux_dust']
        self.pca_r_mean = f['pca_weight_mean_size']
        self.pca_r_std = f['pca_weight_std_size']
        self.pca_n_mean = f['pca_weight_mean_sersic']
        self.pca_n_std = f['pca_weight_std_sersic']
        self.pca_axial_mean = f['pca_weight_mean_q']
        self.pca_axial_std = f['pca_weight_std_q']
        
        # Input normalisation params
        self.norm_input_values = f['normalisation_params']

        # Norm params for phys ml
        self.norm_input_physical_values = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/Physical_Unphysical_Emulator_Data/bool_inputs.h5')['input_normalisation_params']['normalisation_params']

        # Norm params for inbox ml INPUTS
        self.norm_input_inbox_values = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/Inbox_To_Input_Emulator/Inbox_inputs.h5')['input_normalisation_params']['normalisation_params']

        # Norm params for inbox ml OUTPUTS
        self.norm_output_inbox_values = dd.io.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/Inbox_To_Input_Emulator/Original_outputs.h5')['output_normalisation_params']['normalisation_params']

    # Public Method to Evaluate ML predictions - SR
    def eval_MLemulator_SR(self, StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, Inboxmodel, inputs=None, use_physical_emulator=True, need_all_iter=False, Ntrain_vs_Nparam=False):
        # Set the models to evaluation mode
        StellarSEDmodel.eval()
        DustSEDmodel.eval()
        SIZEmodel.eval()
        SERSICmodel.eval()
        Qmodel.eval()
        Physicalmodel.eval()
        Inboxmodel.eval()

        # Columns for Physical/Unphsycai ML
        #validity_columns = ['logMstar', 'logMdMs',  'nstar', 'Age', 't_peak', 'k_peak', 'fwhm', 'k_fwhm'] # Works worse with a smaller parameter set
        validity_columns = ['logMstar', 'logMdMs', 'Rstar', 'CsRs', 'nstar', 'RdRs', 'CdRd', 'ndust', 'f_cov', 'Age', 't_peak', 'k_peak', 'fwhm', 'k_fwhm', 'metal']

        ########### TESTS THE ML ###########
        # Load in testing data that was already preprocessed
        if inputs is None:
            # If called from within this script
            if __name__ == "__main__":
                # Read testing data
                inputs =  pd.read_hdf(f'{self.skirt_file_path}/{self.library_basename}/out_ML/ML_Inputs/{self.library_basename}_test_inputs_h5.h5')
                
                # Extract column labels
                inputs_columns = list(inputs.columns)

                # Make sure columns + column labels align with order ML wants
                table_inputs_columns = pd.DataFrame({'columns':inputs_columns})
                table_inputs_columns = pd.merge(self.table_ML_columns, table_inputs_columns, on='columns', how='inner')
                inputs_columns = table_inputs_columns[table_inputs_columns['flags']]['columns'].to_numpy().astype(str)
                all_inputs = inputs.loc[:, inputs_columns]#.to_numpy()

                # From all_input extract inputs that will be preprocessed into Physical/Unphysical Emulator -> Needs to be denormalized by input params and then normalised again as we use all inputs for Physical/Unphysical Emulator!

                # From all_inputs, extract columms that are used for Physical/Unphysical ML. Then denormalize inputs
                validity_inputs = self.denormalize_parameters(all_inputs).loc[:, validity_columns]#.to_numpy()

                # Normalize the valid inputs using mean std from Physical/Unphysical ML. Then torch
                validity_torch_inputs = torch.Tensor(self.normalize_physical_parameters(validity_inputs).to_numpy())

                # Convert all inputs to numpy too
                all_inputs = all_inputs.to_numpy()

        #######################################



        ############# CALLS TO ML ####################
        # If input are provided to this function
        elif inputs is not None:
            # If not called within this script i.e. called from Plotting Analysis scripts
            if not __name__ == "__main__":
                # Check if its a dataframe
                if isinstance(inputs, pd.DataFrame):
                    # Extract column labels
                    inputs_columns = list(inputs.columns)

                    # Convert inputs to numpy
                    all_inputs = inputs.to_numpy()
                    

                    # Make sure column labels align with ML order
                    table_inputs_columns = pd.DataFrame({'columns':inputs_columns})
                    table_inputs_columns = pd.merge(self.table_ML_columns, table_inputs_columns,  on='columns', how='inner')
                    inputs_columns = table_inputs_columns[table_inputs_columns['flags']]['columns'].to_numpy().astype(str)
                    all_inputs = inputs.loc[:, inputs_columns]

                    # Now we need to double check if the inputs have already been normalised or not!
                    
                    # If mean > 1 then logMstar has not been normalised and inputs should be normalised! This is because normalsied data would have mu ~ 0  ----> THIS WONT WORK PROPELY IF WE ONLY PROCESS ONE ROW OR DATA WITH HIGH logMstar VALUES!
                    if np.mean(all_inputs)[0] > 1:
                        # Print
                        print('Normalising input parameters')

                        # Normalize the data
                        all_inputs = self.normalize_parameters_for_SE3D(all_inputs)

                        
                ### USED WHEN CALLING FROM Analysis_Plots.py
                # If inputs is NOT a dataframe
                else:
                    # If not a dataframe, then convert into dataframe with these specified columns in order!
                    all_inputs = pd.DataFrame(inputs.numpy(), columns=['logMstar', 'logMdMs', 'theta', 'Rstar', 'CsRs', 'nstar', 'RdRs', 'CdRd', 'ndust', 'f_cov', 'Age', 't_peak', 'k_peak', 'fwhm', 'k_fwhm', 'metal'])

                # Extract input data which will be used to in preprocessed Physical/Unphysical Emulator!
                if use_physical_emulator:
                    ###### Remove theta row 
                    validity_inputs = all_inputs.drop('theta', axis=1)

                    # Denormalize inputs, then normalize using mean and std from Physical/Unphysical ML 
                    validity_torch_inputs = torch.Tensor(self.normalize_parameters_for_physical_ML(self.denormalize_parameters_for_SE3D(validity_inputs)).to_numpy())
                

            ########################################


            ########### If called in this script! ###############
            # If called within this script
            else:
                # Get list of input columns
                inputs_columns = list(inputs.columns)

                # Make sure column labels align with ML order
                table_inputs_columns = pd.DataFrame({'columns':inputs_columns})
                table_inputs_columns = pd.merge(self.table_ML_columns, table_inputs_columns, on='columns', how='inner')
                inputs_columns = table_inputs_columns[table_inputs_columns['flags']]['columns'].to_numpy().astype(str)
                all_inputs = inputs.loc[:, inputs_columns]

                # If mean > 1 then logMstar has no been normalised and inputs should be normalised! 
                if np.mean(all_inputs)[0] > 1:
                    # Print
                    print('Normalising input parameters')

                    # Normalize data
                    all_inputs = self.normalize_parameters_for_SE3D(all_inputs)
                    
                # Extract valid outputs using validity columns! Denroamlize it
                validity_inputs = self.denormalize_parameters_for_SE3D(all_inputs).loc[:, validity_columns]#.to_numpy()

                # Normalize the validity inputs required to be used in Physical/Unphysical ML
                validity_torch_inputs = torch.Tensor(self.normalize_parameters_for_physical_ML(validity_inputs).to_numpy())


        ######################### Here we determine if the inputs provided in this function are actually valid! Return a flag for invalid inputs!
        # If we want to use Physical/Unphysical ML. NEED TO CONVERT VALID INBOX INPUTS TO VALID ORIGINAL INPUTS
        if use_physical_emulator:
            # PRint
            print('Using Physical/Unphysical Emulator')

            # Print
            #print(validity_torch_inputs.shape)
            #print(inputs)


            # Denormalize physical validity outputs to pluf into Inbox ML
            validity_torch_inputs = torch.Tensor(self.normalize_inbox_inputs_for_inbox_ML(self.__make_df__(inputs=self.denormalize_parameters_for_physical_ML(self.__make_df__(inputs=validity_torch_inputs.numpy(), cols=validity_columns)), cols=validity_columns)).to_numpy())
            
            # Print
            print(validity_torch_inputs)
            
            # Dont use theta!
            validity_torch_inputs = Inboxmodel(validity_torch_inputs)

            # Denormalize the original outputs now and plug this into Physical/Unphysical ML
            validity_torch_inputs = torch.Tensor(self.normalize_parameters_for_physical_ML(self.__make_df__(inputs=self.denormalize_original_outputs_for_inbox_ML(self.__make_df__(inputs=validity_torch_inputs.detach().numpy(), cols=validity_columns)), cols=validity_columns)).to_numpy())

            # Print
            print(validity_torch_inputs)

            # Call Physical/Unphysical ML
            TrueInputs = Physicalmodel(validity_torch_inputs) ## Convert valid inbox inputs to valid original inputs then process through Physical/Unphysical ML
            TrueInputs = TrueInputs.detach().numpy()#.astype(bool)

            # Convert to bool
            TrueInputs = TrueInputs >= 0.5

            # Print
            print(f'Number of Unphysical Inputs: {len(np.where(np.squeeze(TrueInputs) == False)[0])}')
            
            # Slice out Unphysical Inputs from all inputs
            if not isinstance(all_inputs, np.ndarray):
                torch_inputs = torch.Tensor(all_inputs[np.squeeze(TrueInputs)].to_numpy())
            else:
                torch_inputs = torch.Tensor(all_inputs[np.squeeze(TrueInputs)])
                
        # If we dont want to use the Physical/Unphysical ML
        else:
            # Print
            print('NOT using Physical/Unphysical Emulator')

            # Torch all inputs
            torch_inputs = torch.Tensor(all_inputs.to_numpy())
            

        #########################
        # Stop gradient calculation + make predictions using post processed Physical Inputs
        #print(torch_inputs)
        with torch.no_grad():
            # Predictions for Flux, Size, Sersic
            #if self.bnn_active == False:
            # Set to only two runs as we get same repeat predictions
            SEDIter = 2
            StrucIter = 2
            #else:
            # Should be set to atleast 50 runs as we get different repeat predictions
            #SEDIter = 2
            #StrucIter = 2
            stellarf_pred = np.array([StellarSEDmodel(torch_inputs).detach().numpy() for _ in range(SEDIter)]).T
            dustf_pred = np.array([DustSEDmodel(torch_inputs).detach().numpy() for _ in range(SEDIter)]).T  
            r_pred = np.array([SIZEmodel(torch_inputs).detach().numpy() for _ in range(StrucIter)]).T
            n_pred = np.array([SERSICmodel(torch_inputs).detach().numpy() for _ in range(StrucIter)]).T
            q_pred = np.array([Qmodel(torch_inputs).detach().numpy() for _ in range(StrucIter)]).T

        # If we are doing PCA
        if self.doing_pca:
            print('Undoing PCA')
            # Flux: Denormalise the predicted PCA basis coefficients +/or ref norm 1micron
            stellarf_denorm = (stellarf_pred*self.pca_fstar_std[:, np.newaxis, np.newaxis]) + self.pca_fstar_mean[:, np.newaxis, np.newaxis]

            dustf_denorm = (dustf_pred*self.pca_fdust_std[:, np.newaxis, np.newaxis]) + self.pca_fdust_mean[:, np.newaxis, np.newaxis]

            # Size: Denormalise the predicted PCA basis coefficients +/or ref norm 1micron
            r_denorm = (r_pred*self.pca_r_std[:, np.newaxis, np.newaxis]) + self.pca_r_mean[:, np.newaxis, np.newaxis]

            # Sersic: Denormalise the predicted PCA basis coefficients +/or ref norm 1micron
            n_denorm = (n_pred*self.pca_n_std[:, np.newaxis, np.newaxis]) + self.pca_n_mean[:, np.newaxis, np.newaxis]

            # q: Denormalise the predicted PCA basis coefficients +/or ref norm 1micron
            q_denorm = (q_pred*self.pca_axial_std[:, np.newaxis, np.newaxis]) + self.pca_axial_mean[:, np.newaxis, np.newaxis]

            # If we initially normalised our outputs by taking the reference value at 1 micron
            if self.ref_norm:
                # PRint
                print('Undoing Reference Normalisation')
                # Flux: Extract PCA basis coefficients and ref norm 1 micron separately
                stellarf_weights = stellarf_denorm[1:, :, :] 
                stellarf_ref = stellarf_denorm[0, :, :]

                dustf_weights = dustf_denorm[1:, :, :] 
                dustf_ref = dustf_denorm[0, :, :]
                
                # Size: Extract PCA basis coefficients and ref norm 1 micron separately
                r_weights = r_denorm[1:, :, :] 
                r_ref = r_denorm[0, :, :]
                
                # Sersic: Extract PCA basis coefficients and ref norm 1 micron separately
                n_weights = n_denorm[1:, :, :] 
                n_ref = n_denorm[0, :, :]

                # q: Extract PCA basis coefficients and ref norm 1 micron separately
                q_weights = q_denorm[1:, :, :] 
                q_ref = q_denorm[0, :, :]

                # Essentially .inverse_transform() to convert PCA basis coefficients back into log space + ref norm 1 micron

                # StellarSED
                stellarf_pred_inverse = np.dot(np.transpose(stellarf_weights, (1, 2, 0)), self.pca_flux_star_components)
                stellarf_pred_inverse = stellarf_pred_inverse + self.pca_flux_star_mean[np.newaxis, np.newaxis, :] # Should return [#Test, #Runs, 512]

                # DustSED
                dustf_pred_inverse = np.dot(np.transpose(dustf_weights, (1, 2, 0)), self.pca_flux_dust_components)
                dustf_pred_inverse = dustf_pred_inverse + self.pca_flux_dust_mean[np.newaxis, np.newaxis, :] # Should return [#Test, #Runs, 512]

                # Size
                r_pred_inverse = np.dot(np.transpose(r_weights, (1, 2, 0)), self.pca_size_components)
                r_pred_inverse = r_pred_inverse + self.pca_size_mean[np.newaxis, np.newaxis, :]

                # Sersic
                n_pred_inverse = np.dot(np.transpose(n_weights, (1, 2, 0)), self.pca_sersic_components)
                n_pred_inverse = n_pred_inverse + self.pca_sersic_mean[np.newaxis, np.newaxis, :]

                # Q
                q_pred_inverse = np.dot(np.transpose(q_weights, (1, 2, 0)), self.pca_q_components)
                q_pred_inverse = q_pred_inverse + self.pca_q_mean[np.newaxis, np.newaxis, :]

                # Clip stellar SED like we do with Data
                stellarf_pred_inverse = np.clip(stellarf_pred_inverse, a_min=-5, a_max=None) 

                # Reshape and return to physical units
                stellarf_pred_out = 10**(stellarf_pred_inverse + stellarf_ref[:, :, np.newaxis])
                dustf_pred_out = 10**(dustf_pred_inverse + dustf_ref[:, :, np.newaxis])
                r_pred_out = 10**(r_pred_inverse + r_ref[:, :, np.newaxis])
                n_pred_out = 10**(n_pred_inverse + n_ref[:, :, np.newaxis]) 
                q_pred_out = (q_pred_inverse + q_ref[:, :, np.newaxis])

                # THIS WAS DONE TO PROVIDE ALL OUTPUTS IN DEX TO FIND CORRELATION COEFFICIENT BETWEEN BNN ERRORS AND NMADS -- I THINK?
                '''
                if need_all_iter:
                    stellarf_pred_out = (stellarf_pred_inverse + stellarf_ref[:, :, np.newaxis])
                    dustf_pred_out = (dustf_pred_inverse + dustf_ref[:, :, np.newaxis])
                    r_pred_out = (r_pred_inverse + r_ref[:, :, np.newaxis])
                    n_pred_out = (n_pred_inverse + n_ref[:, :, np.newaxis]) 
                '''
                

            # Return stellar and dust flux separately
            if Ntrain_vs_Nparam:
                # Return them all separately instead of combining StellarSED and DustSED into SED
                fs = np.percentile(stellarf_pred_out, 50, axis=1)
                fd = np.percentile(dustf_pred_out, 50, axis=1)
                r = np.percentile(r_pred_out, 50, axis=1)
                n = np.percentile(n_pred_out, 50, axis=1)
                q = np.percentile(q_pred_out, 50, axis=1)
                
                return fs, fd, r, n, q, TrueInputs
                

            # Add stellar and dust!
            f_pred_out = np.zeros((stellarf_pred_out.shape[0], stellarf_pred_out.shape[1], 512))
            f_pred_out[:, :, :387] += stellarf_pred_out
            f_pred_out[:, :, 512-383:] += dustf_pred_out

            # Convert back to dex to compute error
            #f_pred_out = np.log10(f_pred_out)
            #dustf_pred_out = np.log10(dustf_pred_out)

            # Find mean and std of results!
            print('Finding 50th Percentile and Std Of ML Predictions')

            # SED
            mean_f_results = np.percentile(f_pred_out, 50, axis=1)
            std_f_results = 0.5*(np.percentile(f_pred_out, 84, axis=1) - np.percentile(f_pred_out, 16, axis=1)) 

            # Size
            mean_r_results = np.percentile(r_pred_out, 50, axis=1)
            std_r_results = 0.5*(np.percentile(r_pred_out, 84, axis=1) - np.percentile(r_pred_out, 16, axis=1))  

            # Sersic
            mean_n_results = np.clip(np.percentile(n_pred_out, 50, axis=1), a_min=0.2, a_max=10)
            std_n_results = 0.5*(np.percentile(n_pred_out, 84, axis=1) - np.percentile(n_pred_out, 16, axis=1))

            # Axsial Ratio
            mean_q_results = np.clip(np.percentile(q_pred_out, 50, axis=1), a_min=0.000001, a_max=1)
            std_q_results = 0.5*(np.percentile(q_pred_out, 84, axis=1) - np.percentile(q_pred_out, 16, axis=1))   


            # Return -99 for any Unphysical inputs -> Only JZ needs this amendment?

            # Predictions are returned in linear units. Errors are returned in dex!
            
            # Return bool flags if we use Physical/Unphysical Emulator showing which inputs are unphysical!
            if use_physical_emulator:
                # Print
                print('Returning results using physical ML')
                return mean_f_results, std_f_results, mean_r_results, std_r_results, mean_n_results, std_n_results, mean_q_results, std_q_results, stellarf_ref, dustf_ref, r_ref, n_ref, q_ref, TrueInputs

            # IF we want to return results with all iterations -- used to make find correlation coefficient between BNN Erros and NMADs
            elif need_all_iter:
                # Print
                print('Returning results including all iteratations')
                return mean_f_results, std_f_results, mean_r_results, std_r_results, mean_n_results, std_n_results, mean_q_results, std_q_results, stellarf_ref, dustf_ref, r_ref, n_ref, q_ref, f_pred_out#, r_pred_out, n_pred_out, q_pred_out

            # Simply return without processing through Physical/Unphysical ML
            else:
                # Print
                print('Returning proper results')
                return mean_f_results, std_f_results, mean_r_results, std_r_results, mean_n_results, std_n_results, mean_q_results, std_q_results, stellarf_ref, dustf_ref, r_ref, n_ref, q_ref


    # Public Method to Evaluate ML predictions - JZ
    def eval_MLemulator_JZ(self, StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, Inboxmodel, inputs=None, use_physical_emulator=True):
        # Set models to evaluation mode
        StellarSEDmodel.eval()
        DustSEDmodel.eval()
        SIZEmodel.eval()
        SERSICmodel.eval()
        Qmodel.eval()
        Physicalmodel.eval()
        Inboxmodel.eval()

        # Validity columns - For Physical/Unphysical ML
        validity_columns = ['logMstar', 'logMdMs', 'Rstar', 'CsRs', 'nstar', 'RdRs', 'CdRd', 'ndust', 'f_cov', 'Age', 't_peak', 'k_peak', 'fwhm', 'k_fwhm', 'metal']

        ####### TEST SCRIPT ######
        # If ran from this script, read in testing inputs 
        if __name__ == "__main__":
            # Read testing inputs
            inputs =  pd.read_hdf(f'../SKIRT_library/{self.library_basename}/out_ML/ML_Inputs/{self.library_basename}_test_inputs_h5.h5')

            # Extract column labels
            inputs_columns = list(inputs.columns)

            # Merge columns so they align with Emulator order
            table_inputs_columns = pd.DataFrame({'columns':inputs_columns})
            #print(table_inputs_columns)
            table_inputs_columns = pd.merge(self.table_ML_columns, table_inputs_columns,  on='columns', how='inner')
            #print(table_inputs_columns)
            inputs_columns = table_inputs_columns[table_inputs_columns['flags']]['columns'].to_numpy().astype(str)
            all_inputs = inputs.loc[:, inputs_columns]

            #print(all_inputs)

            # CHECKING NORMALISATION AND DENORMALISATION ARE CONSISTENT
            
            # Inputs read in are normalised, so lets denormalsie and renormalise as a sanity check
            denorm_inputs = self.denormalize_parameters_for_SE3D(all_inputs)
            #all_inputs = self.normalize_parameters_for_SE3D(denorm_inputs)

            # Print
            #print(denorm_inputs)
            #print(all_inputs)
            #print(stop)
            
            # Extract required inputs to deteremine if the inputs are Physical/Unphysical through the preprocssing Physical/Unphysical Emulator
            validity_torch_inputs = torch.Tensor(self.normalize_parameters_for_physical_ML(denorm_inputs.loc[:, validity_columns]).to_numpy())

        # Input labels are now in this form for Emulator: ['logMstar', 'logMdMs', 'theta', 'Rstar', 'CsRs', 'nstar', 'ndust', 'RdRs', 'CdRd', 'f_cov', 'Age', 't_peak', 'k_peak', 'fwhm', 'k_fwhm', 'metal']
        
        # self.normalize_parameters(args) will log Rstar, CsRs, nstar, ndust, RdRs, CdRd, Age, t_peak, fwhm, and metal

        # Metal must be in mass fraction units not solar metal for Emulator

        # If NOT read in from this script i.e. if JZ reads it from fit_obs.py
        else:
            #inputs = inputs[inputs.columns[self.flags]]   ### THIS LINE SHOULD BE UNCOMMENTED IF PARAMETERS EXIST WITHIN INPUTS WHICH ARE NOT APART OF THE EMULATOR TRAINING i.e. BIRTHCLOUD PROPERTIES WHICH STAY CONSTANT

            # Inputs is a dataframe so extract column labels
            inputs_columns = list(inputs.columns)

            # Make sure columns align with ML required order
            table_inputs_columns = pd.DataFrame({'columns':inputs_columns})
            table_inputs_columns = pd.merge(self.table_ML_columns, table_inputs_columns,  on='columns', how='inner')
            inputs_columns = table_inputs_columns[table_inputs_columns['flags']]['columns'].to_numpy().astype(str)
            inputs = inputs.loc[:, inputs_columns]

            # Normalise the input params. These inputs are plugged into the ML
            all_inputs = self.normalize_parameters_for_SE3D(inputs)

            # Extract Input params required for Physical/Unphysical Emulator
            validity_torch_inputs = torch.Tensor(self.normalize_parameters_for_physical_ML(inputs.loc[:, validity_columns]).to_numpy())

        # Turn into torch array
        Ngal, N_features = np.shape(inputs)
        inputs = inputs.to_numpy()

        ######################### Here we determine if the inputs provided in this function are actually valid! Return a flag for invalid inputs!

        ############### CALL TWO ML'S HERE: 1) INBOX -> INPUT ML. 2) PHYSICAL/UNPHYSICAL ML ############
        # Denormalize physical validity outputs to pluf into Inbox ML
        validity_torch_inputs = torch.Tensor(self.normalize_inbox_inputs_for_inbox_ML(self.__make_df__(inputs=self.denormalize_parameters_for_physical_ML(self.__make_df__(inputs=validity_torch_inputs.numpy(), cols=validity_columns)), cols=validity_columns)).to_numpy())
        
        # Print
        #print(validity_torch_inputs)
        
        # Dont use theta!
        validity_torch_inputs = Inboxmodel(validity_torch_inputs)
        
        # Denormalize the original outputs now and plug this into Physical/Unphysical ML
        validity_torch_inputs = torch.Tensor(self.normalize_parameters_for_physical_ML(self.__make_df__(inputs=self.denormalize_original_outputs_for_inbox_ML(self.__make_df__(inputs=validity_torch_inputs.detach().numpy(), cols=validity_columns)), cols=validity_columns)).to_numpy())
        
        # Print
        #print(validity_torch_inputs)

        # Print
        #print(validity_torch_inputs.shape)
        
        # Run valid inputs through Physical/Unphysical ML
        TrueInputs = Physicalmodel(validity_torch_inputs) ## Convert valid inbox inputs to valid original inputs then process through Physical/Unphysical ML
        TrueInputs = TrueInputs.detach().numpy()#.astype(bool)

        # Convert to bool
        TrueInputs = TrueInputs >= 0.5

        # Print
        print(f'Number of Unphysical Inputs: {len(np.where(np.squeeze(TrueInputs) == False)[0])}')

        # Slice inputs so we only run Physical Inputs through Emulator
        torch_inputs = torch.Tensor(all_inputs.to_numpy())
        
        '''
        if not isinstance(all_inputs, np.ndarray):
            torch_inputs = torch.Tensor(all_inputs[np.squeeze(TrueInputs)].to_numpy())
        else:
            torch_inputs = torch.Tensor(all_inputs[np.squeeze(TrueInputs)])
        '''
        
        # Print
        #print(torch_inputs)

        # Relabel
        inputs = torch_inputs
        
        # Reshape Ngal
        Ngal = inputs.shape[0]
        
        ############################################################################################

        # Stop gradient computation
        with torch.no_grad():
            # Predictions for Flux, Size, Sersic
            fstellar_pred = StellarSEDmodel(inputs).detach().numpy() # (Ngal, N_features)
            fdust_pred = DustSEDmodel(inputs).detach().numpy()
            r_pred = SIZEmodel(inputs).detach().numpy() 
            n_pred = SERSICmodel(inputs).detach().numpy()
            q_pred = Qmodel(inputs).detach().numpy()

        # Stellar Flux: Denormalise the predicted PCA basis coefficients + ref norm 1micron
        fstellar_mean = self.pca_fstar_mean.reshape((1, -1)) # (1, N_features)
        fstellar_std = self.pca_fstar_std.reshape((1, -1))
        fstellar_denorm = fstellar_pred*fstellar_std + fstellar_mean # (Ngal, N_feautres)

        # Dust Flux: Denormalise the predicted PCA basis coefficients + ref norm 1micron
        fdust_mean = self.pca_fdust_mean.reshape((1, -1)) # (1, N_features)
        fdust_std = self.pca_fdust_std.reshape((1, -1))
        fdust_denorm = fdust_pred*fdust_std + fdust_mean # (Ngal, N_feautres)

        # Size: Denormalise the predicted PCA basis coefficients + ref norm 1micron
        r_mean = self.pca_r_mean.reshape((1, -1)) # (1, N_features)
        r_std = self.pca_r_std.reshape((1, -1))
        r_denorm = r_pred*r_std + r_mean # (Ngal, N_feautres)

        # Sersic: Denormalise the predicted PCA basis coefficients + ref norm 1micron
        n_mean = self.pca_n_mean.reshape((1, -1)) # (1, N_features)
        n_std = self.pca_n_std.reshape((1, -1))
        n_denorm = n_pred*n_std + n_mean # (Ngal, N_feautres)

        # Axial Ratio: Denormalise the predicted PCA basis coefficients + ref norm 1micron
        q_mean = self.pca_axial_mean.reshape((1, -1)) # (1, N_features)
        q_std = self.pca_axial_std.reshape((1, -1))
        q_denorm = q_pred*q_std + q_mean # (Ngal, N_feautres)

        # Stellar Flux: Extract PCA basis coefficients and ref norm 1 micron separately
        fstellar_weights = fstellar_denorm[:, 1:] # (Ngal, Npca)
        fstellar_ref = fstellar_denorm[:, 0] # (Ngal, 1micron_ref)

        # Dust Flux: Extract PCA basis coefficients and ref norm 1 micron separately
        fdust_weights = fdust_denorm[:, 1:] # (Ngal, Npca)
        fdust_ref = fdust_denorm[:, 0] # (Ngal, 1micron_ref)

        # Size: Extract PCA basis coefficients and ref norm 1 micron separately
        r_weights = r_denorm[:, 1:] # (Ngal, Npca)
        r_ref = r_denorm[:, 0] # (Ngal, 1micron_ref)

        # Sersic: Extract PCA basis coefficients and ref norm 1 micron separately
        n_weights = n_denorm[:, 1:] # (Ngal, Npca)
        n_ref = n_denorm[:, 0] # (Ngal, 1micron_ref)

        # Axial Ratio: Extract PCA basis coefficients and ref norm 1 micron separately
        q_weights = q_denorm[:, 1:] # (Ngal, Npca)
        q_ref = q_denorm[:, 0] # (Ngal, 1micron_ref)

        # Shape of Flux PCA component templates
        Npca, Nwvl = np.shape(self.pca_size_components)

        # Essentially .inverse_transform() to convert PCA basis coefficients back into log space + ref norm 1 micron
        fstellar_pred_inverse = np.matmul(fstellar_weights, self.pca_flux_star_components) # (Ngal, Nwvl)
        fstellar_pred_inverse = fstellar_pred_inverse + self.pca_flux_star_mean.reshape((1, 387))

        fdust_pred_inverse = np.matmul(fdust_weights, self.pca_flux_dust_components) # (Ngal, Nwvl)
        fdust_pred_inverse = fdust_pred_inverse + self.pca_flux_dust_mean.reshape((1, 383))
        
        r_pred_inverse = np.matmul(r_weights, self.pca_size_components) # (Ngal, Nwvl)
        r_pred_inverse = r_pred_inverse + self.pca_size_mean.reshape((1, Nwvl))
        
        n_pred_inverse = np.matmul(n_weights, self.pca_sersic_components) # (Ngal, Nwvl)
        n_pred_inverse = n_pred_inverse + self.pca_sersic_mean.reshape((1, Nwvl))
        
        q_pred_inverse = np.matmul(q_weights, self.pca_q_components) # (Ngal, Nwvl)
        q_pred_inverse = q_pred_inverse + self.pca_q_mean.reshape((1, Nwvl))

        # Reshape and return to physical units

        # StellarSED + DustSED
        stellarf_pred_out = 10**( fstellar_pred_inverse + fstellar_ref.reshape((Ngal, 1)) )
        dustf_pred_out = 10**( fdust_pred_inverse + fdust_ref.reshape((Ngal, 1)) )

        # Size
        r_pred_out = 10**( r_pred_inverse + r_ref.reshape((Ngal, 1)) )

        # Sersic
        n_pred_out = 10**( n_pred_inverse + n_ref.reshape((Ngal, 1)) )
        n_pred_out = np.clip(n_pred_out, a_min=0.2, a_max=10)
        
        # Axial Ratio
        q_pred_out = ( q_pred_inverse + q_ref.reshape((Ngal, 1)) )
        q_pred_out = np.clip(q_pred_out, a_min=0, a_max=1)

        # Total SED
        f_pred_out = np.zeros((stellarf_pred_out.shape[0], 512))
        f_pred_out[:, :387] += stellarf_pred_out
        f_pred_out[:, 512-383:] += dustf_pred_out

        ''' UNUSED
        # Return same shape as inputs but with -99 where inputs == Unphysical
        f = np.full((np.shape(TrueInputs)[0], 512), -99)
        f[TrueInputs.flatten()] = f_pred_out
        f_pred_out = f

        r = np.full((np.shape(TrueInputs)[0], 512), -99)
        r[TrueInputs.flatten()] = r_pred_out
        r_pred_out = r

        n = np.full((np.shape(TrueInputs)[0], 512), -99)
        n[TrueInputs.flatten()] = n_pred_out
        n_pred_out = n

        q = np.full((np.shape(TrueInputs)[0], 512), -99)
        q[TrueInputs.flatten()] = q_pred_out
        q_pred_out = q
        '''

        # Print
        #print(f_pred_out)
        #print(r_pred_out)
        #print(n_pred_out)
        #print(q_pred_out)
        #print(f_pred_out.shape)
        #print(f_pred_out[~TrueInputs.flatten()])
        
        return f_pred_out, r_pred_out, n_pred_out, q_pred_out, TrueInputs # Returns SDs w/ shape (Ngal, Nwvl) and Bool flags for Physical/Unphysical gals w/ shape (Ngal, 1)
        # FOUR OUTPUTS    vvvvv 

    # Public Method to initialise models
    def create_model_instances(self, bnn_active=False):
        # Find Optuna best Activation Function
        StellarSED_activation = [self.__get_activation__(self.best_params_StellarSED[f'activation_{i}']) for i in range(self.best_params_StellarSED['hidden_layers'])]
        DustSED_activation = [self.__get_activation__(self.best_params_DustSED[f'activation_{i}']) for i in range(self.best_params_DustSED['hidden_layers'])]
        SIZE_activation = [self.__get_activation__(self.best_params_SIZE[f'activation_{i}']) for i in range(self.best_params_SIZE['hidden_layers'])]
        SERSIC_activation = [self.__get_activation__(self.best_params_SERSIC[f'activation_{i}']) for i in range(self.best_params_SERSIC['hidden_layers'])]
        Q_activation = [self.__get_activation__(self.best_params_Q[f'activation_{i}']) for i in range(self.best_params_Q['hidden_layers'])]

        # Inbox ML
        Inbox_activation = [self.__get_activation__(self.best_params_Inbox[f'activation_{i}']) for i in range(self.best_params_Inbox['hidden_layers'])]

        # Physcal ML
        Physical_activation = [self.__get_activation__(self.best_params_Physical[f'activation_{i}']) for i in range(self.best_params_Physical['hidden_layers'])]
        
        # Searching for number of hidden layers and respsective values
        StellarSED_nodes = self.__get_nodes__(self.best_params_StellarSED)
        DustSED_nodes = self.__get_nodes__(self.best_params_DustSED)
        SIZE_nodes = self.__get_nodes__(self.best_params_SIZE)
        SERSIC_nodes = self.__get_nodes__(self.best_params_SERSIC)
        Q_nodes = self.__get_nodes__(self.best_params_Q)

        # Inbox ML
        Inbox_nodes = self.__get_nodes__(self.best_params_Inbox)

        # Physical ML
        Physical_nodes = self.__get_nodes__(self.best_params_Physical)

        # List of dropouts
        StellarSEDdrop = [self.best_params_StellarSED[f'dropout_rate_{i}'] for i in range(self.best_params_StellarSED['hidden_layers'])]
        DustSEDdrop = [self.best_params_DustSED[f'dropout_rate_{i}'] for i in range(self.best_params_DustSED['hidden_layers'])]
        SIZEdrop = [self.best_params_SIZE[f'dropout_rate_{i}'] for i in range(self.best_params_SIZE['hidden_layers'])]
        SERSICdrop = [self.best_params_SERSIC[f'dropout_rate_{i}'] for i in range(self.best_params_SERSIC['hidden_layers'])]
        Qdrop = [self.best_params_Q[f'dropout_rate_{i}'] for i in range(self.best_params_Q['hidden_layers'])]

        # Inbox ML
        Inboxdrop = [self.best_params_Inbox[f'dropout_rate_{i}'] for i in range(self.best_params_Inbox['hidden_layers'])]

        # Physical ML
        Physicaldrop = [self.best_params_Physical[f'dropout_rate_{i}'] for i in range(self.best_params_Physical['hidden_layers'])]
        
        # Model initialisation
        StellarSEDmodel = BayesianNN(self.in_features, self.out_featuresStellarSED, StellarSED_nodes, StellarSEDdrop, self.best_params_StellarSED['mu'], self.best_params_StellarSED['sig'], StellarSED_activation, self.best_params_StellarSED['weight_init_name'], bnn_active)
        DustSEDmodel = BayesianNN(self.in_features, self.out_featuresDustSED, DustSED_nodes, DustSEDdrop, self.best_params_DustSED['mu'], self.best_params_DustSED['sig'], DustSED_activation, self.best_params_DustSED['weight_init_name'], bnn_active)
        SIZEmodel = BayesianNN(self.in_features, self.out_featuresSIZE, SIZE_nodes, SIZEdrop, self.best_params_SIZE['mu'], self.best_params_SIZE['sig'], SIZE_activation, self.best_params_SIZE['weight_init_name'], bnn_active)
        SERSICmodel = BayesianNN(self.in_features, self.out_featuresSERSIC, SERSIC_nodes, SERSICdrop, self.best_params_SERSIC['mu'], self.best_params_SERSIC['sig'], SERSIC_activation, self.best_params_SERSIC['weight_init_name'], bnn_active)
        Qmodel = BayesianNN(self.in_features, self.out_featuresQ, Q_nodes, Qdrop, self.best_params_Q['mu'], self.best_params_Q['sig'], Q_activation, self.best_params_Q['weight_init_name'], bnn_active)

        # Inbox ML
        Inboxmodel = BayesianNN(15, 15, Inbox_nodes, Inboxdrop, self.best_params_Inbox['mu'], self.best_params_Inbox['sig'], Inbox_activation, self.best_params_Inbox['weight_init_name'], emulator='Inbox')

        # Physical ML
        Physicalmodel = BayesianNN(15, 1, Physical_nodes, Physicaldrop, self.best_params_Physical['mu'], self.best_params_Physical['sig'], Physical_activation, self.best_params_Physical['weight_init_name'], emulator='Physical')

        # Initialise PhysicalNN
        #Physicalmodel = PhysicalNN()

        # Initialise Inbox To Input ML
        #Inboxmodel = InboxInputNN()

        return StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, Inboxmodel

    # Public Method to use checkpoint to load in best models
    def load_best_model(self, bnn_active=False):
        # Call Public Method to create instances of each model
        StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, Inboxmodel = self.create_model_instances(bnn_active)

        # Print
        print(f'Loading Models for library {self.library_basename}')

        # If creating Ntrain vs Nparam plot
        if self.f_path_size is not None:
            # Use path size in argument
            path = self.f_path_size
            print('Using models for Ntrain vs Nparam')

        # If not making Ntrain vs Nparam plot
        else:
            # Print
            print('Using best models')
            path = f'{self.skirt_file_path}/{self.library_basename}/out_ML/SE3D_Models/'

        # Load best model parameters from checkpoints - Flux, Size, Sersic
        #checkpoint_StellarSED = torch.load(f'../SKIRT_library/{self.library_basename}/out_ML/SE3D_Models/best_models/{self.library_basename}_flux_star_best_model.pt', weights_only=True)
        checkpoint_StellarSED = torch.load(path + f'../SE3D_Models/best_models/{self.library_basename}_flux_star_best_model.pt', weights_only=True)
        StellarSEDmodel.load_state_dict(checkpoint_StellarSED['model_state_dict'])
        StellarSEDmodel.to(self.device)

        #checkpoint_DustSED = torch.load(f'../SKIRT_library/{self.library_basename}/out_ML/SE3D_Models/best_models/{self.library_basename}_flux_dust_best_model.pt', weights_only=True)
        checkpoint_DustSED = torch.load(path + f'../SE3D_Models/best_models/{self.library_basename}_flux_dust_best_model.pt', weights_only=True)
        DustSEDmodel.load_state_dict(checkpoint_DustSED['model_state_dict'])
        DustSEDmodel.to(self.device)
        
        #checkpoint_SIZE = torch.load(f'../SKIRT_library/{self.library_basename}/out_ML/SE3D_Models/best_models/{self.library_basename}_size_best_model.pt', weights_only=True)
        checkpoint_SIZE = torch.load(path + f'../SE3D_Models/best_models/{self.library_basename}_size_best_model.pt', weights_only=True)
        SIZEmodel.load_state_dict(checkpoint_SIZE['model_state_dict'])
        SIZEmodel.to(self.device)
        
        #checkpoint_SERSIC = torch.load(f'../SKIRT_library/{self.library_basename}/out_ML/SE3D_Models/best_models/{self.library_basename}_sersic_best_model.pt', weights_only=True)
        checkpoint_SERSIC = torch.load(path + f'../SE3D_Models/best_models/{self.library_basename}_sersic_best_model.pt', weights_only=True)
        SERSICmodel.load_state_dict(checkpoint_SERSIC['model_state_dict'])
        SERSICmodel.to(self.device)

        #checkpoint_Q = torch.load(f'../SKIRT_library/{self.library_basename}/out_ML/SE3D_Models/best_models/{self.library_basename}_q_best_model.pt', weights_only=True)
        checkpoint_Q = torch.load(path + f'../SE3D_Models/best_models/{self.library_basename}_q_best_model.pt', weights_only=True)
        Qmodel.load_state_dict(checkpoint_Q['model_state_dict'])
        Qmodel.to(self.device)
        
        # Load checkpoint for Physical/Unphysical Emulator
        #checkpoint_PhysicalNN = torch.load(f'../SKIRT_library/{self.library_basename}/out_ML/SE3D_Models/best_models/physical_NN.pt', weights_only=True)
        checkpoint_PhysicalNN = torch.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/SE3D_Models/best_models/Physical_Unphysical.pt', weights_only=True)
        Physicalmodel.load_state_dict(checkpoint_PhysicalNN['model_state_dict'])

        # Load checkpoint for Inbox To Input Emulator
        checkpoint_InboxNN = torch.load(f'{self.skirt_file_path}/{self.library_basename}/out_ML/SE3D_Models/best_models/Inbox_To_Original.pt', weights_only=True)
        Inboxmodel.load_state_dict(checkpoint_InboxNN['model_state_dict'])

        return StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, Inboxmodel 


    ########################  PUBLIC METHODS FOR NORMALIZATION AND DENORMALIZATION ##########################
    # Public Method to normalise parameters
    def normalize_parameters_for_SE3D(self, inputs):
        # Log inputs first!
        columns_to_log = ['Rstar', 'CsRs', 'nstar', 'ndust', 'RdRs', 'CdRd', 'Age', 't_peak', 'fwhm', 'metal']

        # Work on copied inputs -- SLOW!
        #inputs_copy = inputs.copy()

        # Ensure all columns to log exist in the DataFrame
        #missing_columns = [col for col in columns_to_log if col not in inputs.columns]
        #if missing_columns:
        #    raise ValueError(f"Missing columns for log transformation: {missing_columns}")

        # Confirm which columns are actually in df
        columns_to_log = [col for col in columns_to_log if col in inputs.columns]
        
        # Log-transform the specified columns
        #for column in columns_to_log:
            #inputs_copy[column] = np.log10(inputs_copy[column]) #.replace(0, np.nan))

        # Empty dict
        normalized_params = {}

        # Dict shape
        Ngal, Nparams = np.shape(inputs)
        normalized_params = pd.DataFrame(data=np.zeros([Ngal, Nparams], dtype=np.float32), columns=inputs.keys())

        # Iterate through and normalise
        for key in inputs.keys():
            if key in self.norm_input_values:
                # Extract mean, std
                mean = self.norm_input_values[key]['mean']
                std = self.norm_input_values[key]['std']

                # New arr to normaliez
                values = inputs[key].to_numpy(dtype=np.float32)

                if key in columns_to_log:
                    values = np.log10(values)

                # Now normalize!
                normalized_params[key] = (values - mean) / std
            else:
                raise KeyError(f"Normalization parameters for '{key}' are not provided.")
 
        return normalized_params

    # Public Method to normalise physical parameters ## FOR PHYSICAL/UNPHYSICAL EMULATOR
    def normalize_parameters_for_physical_ML(self, inputs):
        # List of params to log
        columns_to_log = ['Rstar', 'CsRs', 'nstar', 'RdRs', 'CdRd', 'ndust', 'Age', 't_peak', 'fwhm', 'metal']

        # Work on copied inputs -- SLOW!
        #inputs_copy = inputs.copy()
        
        # Empty dict
        normalized_params = {}
        #print(np.shape(inputs))

        # Dict shape
        #normalized_params = np.copy(params)
        Ngal, Nparams = np.shape(inputs)
        normalized_params = pd.DataFrame(data=np.zeros([Ngal, Nparams], dtype=np.float32), columns=inputs.keys())

        # Iterate through and normalise
        for key in inputs.keys():
            if key in self.norm_input_physical_values:
                # Extract mean, std
                mean = self.norm_input_physical_values[key]['mean']
                std = self.norm_input_physical_values[key]['std']
                #print('key', mean, std)

                # New arr to normaliez
                values = inputs[key].to_numpy(dtype=np.float32)

                # Log
                if key in columns_to_log:
                    values = np.log10(values)
                
                # Normalise
                normalized_params[key] = (values - mean) / std
            else:
                raise KeyError(f"Normalization parameters for '{key}' are not provided.")
 
        return normalized_params

    # Public Method to normalise physical parameters ## FOR INBOX to ORIGINAL INPUTS EMULATOR. THIS NORMALIZES INBOX INPUTS BEFORE INBOX NN
    def normalize_inbox_inputs_for_inbox_ML(self, inputs):
        # Log inputs first!
        columns_to_log = ['Rstar', 'CsRs', 'nstar', 'RdRs', 'CdRd', 'ndust', 'Age', 't_peak', 'fwhm', 'metal']

        # Work on copied inputs -- SLOW!
        #inputs_copy = inputs.copy()
        
        # Log columns!
        #inputs_copy.loc[:, columns_to_log] = np.log10(inputs_copy.loc[:, columns_to_log])
        
        # Empty dict
        normalized_params = {}
        #print(np.shape(inputs))

        # Dict shape
        #normalized_params = np.copy(params)
        Ngal, Nparams = np.shape(inputs)
        normalized_params = pd.DataFrame(data=np.zeros([Ngal, Nparams], dtype=np.float32), columns=inputs.keys())

        # Iterate through and normalise
        for key in inputs.keys():
            #print(key)
            if key in self.norm_input_inbox_values:
                # Extract mean, std
                mean = self.norm_input_inbox_values[key]['mean']
                std = self.norm_input_inbox_values[key]['std']

                # New arr to normaliez
                values = inputs[key].to_numpy(dtype=np.float32)

                # Log
                if key in columns_to_log:
                    values = np.log10(values)

                # Now normalize!
                normalized_params[key] = (values - mean) / std
            else:
                raise KeyError(f"Normalization parameters for '{key}' are not provided.")
 
        return normalized_params

    # Public Method to normalise parameters
    def denormalize_parameters_for_SE3D(self, inputs):
        # Empty dict
        denormalized_params = {}
        #print(np.shape(inputs))

        # Work on copied inputs -- SLOW!
        #inputs_copy = inputs.copy()

        # Dict shape
        #normalized_params = np.copy(params)
        Ngal, Nparams = np.shape(inputs)
        denormalized_params = pd.DataFrame(data=np.zeros([Ngal, Nparams], dtype=np.float32), columns=inputs.keys())

        #print(self.norm_input_values)

        # Iterate through and denormalise
        for key in inputs.keys():
            #print(key)
            if key in self.norm_input_values:
                # Extract mean, std
                mean = self.norm_input_values[key]['mean']
                std = self.norm_input_values[key]['std']

                # New arr
                values = inputs[key].to_numpy(dtype=np.float32)
                denormalized_params[key] = (values * std) + mean
            else:
                raise KeyError(f"Normalization parameters for '{key}' are not provided.")

        # Then 10** to put inputs back to physical form
        columns_to_unlog = ['Rstar', 'CsRs', 'nstar', 'ndust', 'RdRs', 'CdRd', 'Age', 't_peak', 'fwhm', 'metal']

        # Ensure all columns to log exist in the DataFrame
        missing_columns = [col for col in columns_to_unlog if col not in denormalized_params.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for log transformation: {missing_columns}")
        
        # Log-transform the specified columns
        for column in columns_to_unlog:
            denormalized_params[column] = 10**(denormalized_params[column]) #.replace(0, np.nan))
 
        return denormalized_params

    # Public Method to normalise Physical parameters ## FOR PHYSICAL/UNPHYSICAL EMULATOR
    def denormalize_parameters_for_physical_ML(self, inputs):
        # Empty dict
        denormalized_params = {}
        #print(np.shape(inputs))

        # Work on copied inputs -- SLOW!
        #inputs_copy = inputs.copy()

        # Dict shape
        #normalized_params = np.copy(params)
        Ngal, Nparams = np.shape(inputs)
        denormalized_params = pd.DataFrame(data=np.zeros([Ngal, Nparams], dtype=np.float32), columns=inputs.keys())

        #print(self.norm_input_values)

        # Iterate through and denormalise
        for key in inputs.keys():
            #print(key)
            if key in self.norm_input_physical_values:
                # Extract mean, std
                mean = self.norm_input_physical_values[key]['mean']
                std = self.norm_input_physical_values[key]['std']
                
                # New arr
                values = inputs[key].to_numpy(dtype=np.float32)
                denormalized_params[key] = (values * std) + mean
            else:
                raise KeyError(f"Normalization parameters for '{key}' are not provided.")

        # Then 10** to put inputs back to physical form
        columns_to_unlog = ['Rstar', 'CsRs', 'nstar', 'ndust', 'RdRs', 'CdRd', 'Age', 't_peak', 'fwhm', 'metal']

        '''
        # Ensure all columns to log exist in the DataFrame
        missing_columns = [col for col in columns_to_unlog if col not in denormalized_params.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for log transformation: {missing_columns}")
        '''
        
        # Log-transform the specified columns
        for column in columns_to_unlog:
            denormalized_params[column] = 10**(denormalized_params[column]) #.replace(0, np.nan))
 
        return denormalized_params

    # Public Method to normalise Physical parameters ## FOR INBOX To ORIGINAL INPUTS EMULATOR. THIS DENORMALIZES THE OUTPUTS OF INBOX NN
    def denormalize_original_outputs_for_inbox_ML(self, inputs):
        # Work on copied inputs -- SLOW!
        #inputs_copy = inputs.copy()
        
        # Init empty dict
        denormalized_params = {}
        #print(np.shape(inputs))

        # Dict shape
        #normalized_params = np.copy(params)
        Ngal, Nparams = np.shape(inputs)
        denormalized_params = pd.DataFrame(data=np.zeros([Ngal, Nparams], dtype=np.float32), columns=inputs.keys())

        #print(self.norm_input_values)

        # Iterate through and denormalise
        for key in inputs.keys():
            #print(key)
            if key in self.norm_output_inbox_values:
                # Extract mean, std
                mean = self.norm_output_inbox_values[key]['mean']
                std = self.norm_output_inbox_values[key]['std']
                
                # New arr
                values = inputs[key].to_numpy(dtype=np.float32)
                denormalized_params[key] = (values * std) + mean
            else:
                raise KeyError(f"Normalization parameters for '{key}' are not provided.")

        # Log inputs first!
        columns_to_unlog = ['Rstar', 'CsRs', 'nstar', 'RdRs', 'CdRd', 'ndust', 'Age', 't_peak', 'fwhm', 'metal']
        
        # Log-transform the specified columns
        for column in columns_to_unlog:
            denormalized_params[column] = 10**(denormalized_params[column]) #.replace(0, np.nan))
 
        return denormalized_params
    


    ####################################### FORWARD METHODS ##########################################

    # Forward Method
    def forward():
        pass


    ####################################### HELPER METHODS ##########################################

    # Helper Method to get Activation Function
    def __get_activation__(self, activation_name):
        # Dictionary of current activation functions
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "leaky_relu":nn.LeakyReLU(), "gelu": nn.GELU(), "gelu_tanh": nn.GELU(approximate='tanh'), "silu": nn.SiLU(), "celu": nn.CELU()}
        
        return activations[activation_name]

    # Helper Method to get Optimizer
    def __get_optimizer__(self, model, optimizer_name, weight_decay, lr, beta1, beta2, mom):
        # Adam
        if optimizer_name == 'Adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2))
            
        # AdamW
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

        # SGD
        if optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)

        # RMSprop
        if optimizer_name == 'RMSProp':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=mom)

        # NAdam
        if optimizer_name == 'NAdam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=lr, betas=(beta1, beta2))

        return optimizer

    # Helper Method to get number of nodes in each layer
    def __get_nodes__(self, data):
        hidden_layers = data.get('hidden_layers', 0)
        
        # Initialize a list to store the n_units values
        n_units_values = []
        
        # Extract n_units values for each layer
        for layer in range(hidden_layers):
            key = f'n_units_l{layer}'
            if key in data:
                n_units_values.append(data[key])

        # Display the extracted values
        #print(n_units_values)
        
        return n_units_values

    # Helper Method to make inputs go into a df with columns
    def __make_df__(self, inputs, cols):
        # Make a df
        df = pd.DataFrame(inputs, columns=cols)

        return df

# Execute
if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()
    ###############################################################################################

    ##################### for Felix ############################
    # Conditions
    library_basename = '16par_RANDOM'

    # Instantiate the Eval class
    Eval = EvalMLEmulator(library_basename, doing_pca=True, ref_norm=True)

    # Read Inputs 
    #inputs =  pd.read_hdf(f'../SKIRT_library/{library_basename}/out_ML/ML_Inputs/{library_basename}_test_inputs_h5.h5')
    
     ##### How physical galaxy inputs should look #####
    inputs = pd.DataFrame({'logMstar':[9.5, 10.0, 10.5, 11.0], 'logMdMs':[-2.0, -2.5, -3.0, -3.5], 'theta':[45, 45, 45, 45], 'Rstar':[1, 1, 1, 1], 'CsRs':[1, 1, 1, 1], 'nstar':[1, 1, 1, 1], 'RdRs':[1, 1, 1, 1], 'CdRd':[1, 1, 1, 1], 'ndust':[1, 1, 1, 1], 'f_cov':[0.1, 0.1, 0.1, 0.1], 'Age':[2, 2, 2, 2], 't_peak':[2, 2, 2, 2], 'k_peak':[0, 0, 0, 0], 'fwhm':[2, 2, 2, 2], 'k_fwhm':[0, 0, 0, 0], 'metal':[0.02, 0.02, 0.02, 0.02]}) ## Vary logMstar
                
    # Load in best models! ## bnn_active = True means we use BNN to make predictions. False means we set mean of the weights to be deterministic instead of probabilistic
    StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, Inboxmodel = Eval.load_best_model(bnn_active=True)

    # Eval function for SR 
    #Eval.eval_MLemulator_SR(StellarSEDmodel=StellarSEDmodel, DustSEDmodel=DustSEDmodel, SIZEmodel=SIZEmodel, SERSICmodel=SERSICmodel, Qmodel=Qmodel, Physicalmodel=Physicalmodel, Inboxmodel=Inboxmodel, inputs=inputs, use_physical_emulator=True)

    # Eval function for JZ
    Eval.eval_MLemulator_JZ(StellarSEDmodel=StellarSEDmodel, DustSEDmodel=DustSEDmodel, SIZEmodel=SIZEmodel, SERSICmodel=SERSICmodel, Qmodel=Qmodel, Physicalmodel=Physicalmodel, Inboxmodel=Inboxmodel, inputs=None, use_physical_emulator=False)

    ###################### for Felix ##############################
   
    '''
    ##### How physical galaxy inputs should look #####
    #inputs = pd.DataFrame({'logMstar':[9.5, 10.0, 10.5, 11.0], 'logMdMs':[-2.0, -2.5, -3.0, -3.5], 'theta':[45, 45, 45, 45], 'Rstar':[1, 1, 1, 1], 'CsRs':[1, 1, 1, 1], 'nstar':[1, 1, 1, 1], 'RdRs':[1, 1, 1, 1], 'CdRd':[1, 1, 1, 1], 'ndust':[1, 1, 1, 1], 'f_cov':[0.1, 0.1, 0.1, 0.1], 'Age':[2, 2, 2, 2], 't_peak':[2, 2, 2, 2], 'k_peak':[0, 0, 0, 0], 'fwhm':[2, 2, 2, 2], 'k_fwhm':[0, 0, 0, 0], 'metal':[0.02, 0.02, 0.02, 0.02]}) ## Vary logMstar
    
    #inputs = pd.DataFrame({'logMstar':[10.5, 10.5, 10.5, 10.5, 10.5], 'logMdMs':[-3.0, -3.0, -3.0, -3.0, -3.0], 'theta':[45, 45, 45, 45, 45], 'Rstar':[0.5, 1, 2, 3, 4], 'CsRs':[0.2, 0.2, 0.2, 0.2, 0.2], 'nstar':[1, 1, 1, 1, 1], 'RdRs':[1, 1, 1, 1, 1], 'CdRd':[0.2, 0.2, 0.2, 0.2, 0.2], 'ndust':[1, 1, 1, 1, 1], 'f_cov':[0.1, 0.1, 0.1, 0.1, 0.1], 'Age':[2, 2, 2, 2, 2], 't_peak':[2, 2, 2, 2, 2], 'k_peak':[0, 0, 0, 0, 0], 'fwhm':[2, 2, 2, 2, 2], 'k_fwhm':[0, 0, 0, 0, 0], 'metal':[0.02, 0.02, 0.02, 0.02, 0.02]}) ## Scale Galaxy Up/Down

    #inputs = pd.DataFrame({'logMstar':[10.5, 10.5, 10.5, 10.5], 'logMdMs':[-3.0, -3.0, -3.0, -3.0], 'theta':[45, 45, 45, 45], 'Rstar':[1, 2, 3, 4], 'CsRs':[0.2, 0.2, 0.2, 0.2], 'nstar':[2, 2, 2, 2], 'RdRs':[1, 1, 1, 1], 'CdRd':[0.2, 0.2, 0.2, 0.2], 'ndust':[1, 1, 1, 1], 'f_cov':[0.1, 0.1, 0.1, 0.1], 'Age':[2, 2, 2, 2], 't_peak':[2, 2, 2, 2], 'k_peak':[0, 0, 0, 0], 'fwhm':[2, 2, 2, 2], 'k_fwhm':[0, 0, 0, 0], 'metal':[0.02, 0.02, 0.02, 0.02]}) ## Scale Galaxy Up/Down
    
    inputs = pd.DataFrame({'logMstar':[10.5, 10.5, 10.5, 10.5, 10.5], 'logMdMs':[-2.0, -2.5, -3.0, -3.5, -4.0], 'theta':[45, 45, 45, 45, 45], 'Rstar':[3, 3, 3, 3, 3], 'CsRs':[0.2, 0.2, 0.2, 0.2, 0.2], 'nstar':[2, 2, 2, 2, 2], 'RdRs':[1, 1, 1, 1, 1], 'CdRd':[0.2, 0.2, 0.2, 0.2, 0.2], 'ndust':[1, 1, 1, 1, 1], 'f_cov':[0.1, 0.1, 0.1, 0.1, 0.1], 'Age':[2, 2, 2, 2, 2], 't_peak':[2, 2, 2, 2, 2], 'k_peak':[0, 0, 0, 0, 0], 'fwhm':[2, 2, 2, 2, 2], 'k_fwhm':[0, 0, 0, 0, 0], 'metal':[0.02, 0.02, 0.02, 0.02, 0.02]}) ##vary logMdust


    #inputs = pd.DataFrame({'logMstar':[10.5, 10.5, 10.5, 10.5], 'logMdMs':[-2.0, -2.0, -2.0, -2.0], 'theta':[45, 45, 45, 45], 'Rstar':[1, 1, 1, 1], 'CsRs':[1, 1, 1, 1], 'nstar':[1, 1, 1, 1], 'RdRs':[1, 1, 1, 1], 'CdRd':[1, 1, 1, 1], 'ndust':[1, 1, 1, 1], 'f_cov':[0.1, 0.1, 0.1, 0.1], 'Age':[2, 2, 2, 2], 't_peak':[2, 2, 2, 2], 'k_peak':[0, 0, 0, 0], 'fwhm':[2, 2, 2, 2], 'k_fwhm':[0, 0, 0, 0], 'metal':[0.01, 0.03, 0.04, 0.05]})  # Vary metal 

    print(f'Galaxy inputs:')
    print(inputs)

    # Make ML predictions from physical galaxy inputs
    #Eval.eval_MLemulator_JZ(StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel)
    f, ferr, r, rerr, n, nerr, q, qerr, fs_ref, fd_ref, r_ref, n_ref, q_ref = Eval.eval_MLemulator_SR(StellarSEDmodel, DustSEDmodel, SIZEmodel, SERSICmodel, Qmodel, Physicalmodel, inbox_model, inputs, use_physical_emulator=False)

    # Print
    #print(f'Number of galaxies to predict: {f.shape[0]}')
    #print(f'Physical Flags: {physical_flags}')

    # Wvl array
    wvl = pd.read_hdf(f'../SKIRT_library/{library_basename}/out_SKIRT/DATA/{library_basename}_wavelength.h5').to_numpy().flatten()[0]

    # Param to vary
    param_to_vary = 'Rstar'
    vals = inputs[param_to_vary]
    #print(vals)

    # List of SKIRT runs to plot
    #to_plot = [0, 1, 2, 3, 4]#, 4] ## Vary logMstar --> output_4 failed!

    #to_plot = [1, 2, 3, 4]#, 4] ## Scale gal
    to_plot = [5, 6, 7, 8]#, 9] ## Scale Galaxy Up/Down
    #to_plot = [10, 11, 12, 13]#, 14] ## Vary logMdust --> output_14 failed!

    # Quick plotting for visualisation
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    colours = plt.cm.jet(np.linspace(0.3, 0.9, np.shape(to_plot)[0]))
    colours = colours[::-1]
    #print(colours[0])

    # Plot legend flag
    show_legend = False
    # Loop through all galaxies to plot on the same figure
    for i, gal in enumerate(to_plot):  #enumerate(range(4)):
        #if i ==0:
            #continue
        
        # Read SKIRT data file
        d = dd.io.load(f'../SKIRT_library/{library_basename}/out_SKIRT/outputs/output_0/output_{gal}/data.h5')['data_6']
        #d = dd.io.load(f'../SKIRT_library/{library_basename}/out_SKIRT/outputs/varying_params_SKIRT_outputs/output_0/output_{gal}/data.h5')['data_6']
        #d1 = dd.io.load(f'../SKIRT_library/{library_basename}/out_SKIRT/outputs/varying_params_SKIRT_outputs/output_1/output_{gal}/data.h5')['data_6']
        #d2 = dd.io.load(f'../SKIRT_library/{library_basename}/out_SKIRT/outputs/output_0/output_{gal}/data.h5')['data_6']
        print(d)
        #2print(d1)
        #print(stop)
        
        # Plot ML + SKIRT predictions
        # Flux
        print(vals.iloc[i])
        #print(colours[i])
        
        
        # Show legend only once!
        if show_legend:
            axs[0].plot(wvl, f[i, :], color=colours[i],  linestyle='--', label=f'{vals.iloc[i]}')
            axs[0].fill_between(wvl, f[i, :] + ferr[i, :], f[i, :] - ferr[i, :], color=colours[i], alpha=0.3)
            axs[0].plot(wvl, d['flux'].to_numpy(), color=colours[i], label='SKIRT')
            axs[0].legend(fontsize=24, loc='lower left')
            # Change legend to false
            show_legend = False
        else:
            axs[0].plot(wvl, f[i, :], color=colours[i], linestyle='--')
            axs[0].fill_between(wvl, f[i, :] + ferr[i, :], f[i, :] - ferr[i, :], color=colours[i], alpha=0.3)
            axs[0].plot(wvl, d['flux'].to_numpy(), color=colours[i])
            #axs[0].plot(wvl, d1['flux'].to_numpy(), linestyle='-.', color=colours[i])
            #axs[0].plot(wvl, d2['flux'].to_numpy(), linestyle='dotted', color=colours[i])

        # Sixe
        axs[1].plot(wvl, r[i, :], color=colours[i], linestyle='--')
        axs[1].fill_between(wvl, r[i, :] + rerr[i, :], r[i, :] - rerr[i, :], color=colours[i], alpha=0.3)
        axs[1].plot(wvl, d['r'].to_numpy(), color=colours[i] )
        #axs[1].plot(wvl, d1['r'].to_numpy(), linestyle='-.', color=colours[i])
        #axs[1].plot(wvl, d2['r'].to_numpy(), linestyle='dotted', color=colours[i])

        # Sersic
        axs[2].plot(wvl, n[i, :], color=colours[i], linestyle='--')
        axs[2].fill_between(wvl, n[i, :] + nerr[i, :], n[i, :] - nerr[i, :], color=colours[i], alpha=0.3)
        axs[2].plot(wvl, gaussian_filter1d(d['n'].to_numpy(), 5), color=colours[i])
        #axs[2].plot(wvl, gaussian_filter1d(d1['n'].to_numpy(), 5), linestyle='-.', color=colours[i])
        #axs[2].plot(wvl, gaussian_filter1d(d2['n'].to_numpy(), 5), linestyle='dotted', color=colours[i])
        #axs[2].plot(wvl, d['n'].to_numpy(), color=colours[i])

        # q
        #axs[3].plot(wvl, q[i, :], color=colours[i])
        #axs[3].fill_between(wvl, q[i, :] + qerr[i, :], q[i, :] - qerr[i, :], color=colours[i], alpha=0.2)
        #axs[3].plot(wvl, d['q'].to_numpy(), color=colours[i], linestyle='--')

    #axs[0, 0].plot(wvl, f[2, :], color='black', label=f'{vals.iloc[2]}')
    #axs[0, 1].plot(wvl, r[2, :], color='black')
    #axs[1, 0].plot(wvl, n[2, :], color='black')
    #axs[1, 1].plot(wvl, q[2, :], color='black')
        
    # Set to log
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')
    #axs[2].set_xscale('log')
    #axs[3].set_xscale('log')
    axs[0].set_yscale('log')
    #plt.yscale('log')

    # Axis labels
    axs[0].set_ylabel('λF(λ)  [W/$m^{2}$]', fontsize=14)
    axs[1].set_ylabel('Half-light radius [kpc]', fontsize=14)
    axs[2].set_ylabel(r'${\rm{S\'{e}rsic~index}}~n$', fontsize=14)
    #axs[3].set_ylabel('Axial Ratio, q', fontsize=14)
    #axs[0, 0].set_xlabel('λ [$\\mu m$]')
    #axs[0, 1].set_xlabel('λ [$\\mu m$]')
    #axs[1, 0].set_xlabel('λ [$\\mu m$]')
    #axs[1, 1].set_xlabel('λ [$\\mu m$]')
    plt.xlabel('λ [$\\mu m$]', fontsize=14)
    

    # Legend
    axs[0].legend()
    #axs[3].set_ylim(0, 1)

    #print(q[0, 400])
    #print(q[1, 400])
    #print(wvl[400])

    # Plot
    plt.tight_layout()
    plt.savefig(f'../SKIRT_library/{library_basename}/plots/Varying_param.png')
    plt.close()
    '''
    

    ###############################################################################################
    # Record the end time
    end_time = datetime.now()
    
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Eval ML Emulator Script completed in: {elapsed_time}")
