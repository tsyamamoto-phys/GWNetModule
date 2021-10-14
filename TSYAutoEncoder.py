"""
TSYAutoEncoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
#import _utils as u
from . import _utils as u


class TSYAutoEncoder(nn.Module):

    def __init__(self, netstructure):
        super(TSYAutoEncoder, self).__init__()

        gl = u.GenerateLayer()

        if isinstance(netstructure, str):
            with open(netstructure, "r") as f:
                print(f"net structure is loaded from `{netstructure}`")
                netstructure = json.load(f)

        encoderlayers = []
        for l in netstructure["Encoder"]:
            layername = l["lname"]
            encoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder = nn.ModuleList(encoderlayers)

        decoderlayers = []
        for l in netstructure["Decoder"]:
            layername = l["lname"]
            decoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.decoder = nn.ModuleList(decoderlayers)


    def Encode(self, x):
        for i, l in enumerate(self.encoder):
            x = l(x)
        return x


    def Decode(self, x):
        for i, l in enumerate(self.decoder):
            x = l(x)
        return x


    def FreezeEncoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.Encode(x)
        x = self.Decode(x)
        return x



class TSYVariationalAutoEncoder(nn.Module):

    def __init__(self, netstructure, cudaflg=False):
        super(TSYVariationalAutoEncoder, self).__init__()

        self.cudaflg = cudaflg

        if isinstance(netstructure, str):
            with open(netstructure, "r") as f:
                print(f"net structure is loaded from `{netstructure}`")
                netstructure = json.load(f)


        gl = u.GenerateLayer()

        encoderlayers = []
        for l in netstructure["Encoder"]:
            layername = l["lname"]
            encoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder = nn.ModuleList(encoderlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        Nhid = netstructure["Dimension of hidden variable"]
        Nin = self.encoder[-2].out_features  # I need to sophisticate this part.
        self.encoder_mean = nn.Linear(Nin, Nhid)
        self.encoder_logvar = nn.Linear(Nin, Nhid)


        decoderlayers = []
        for l in netstructure["Decoder"]:
            layername = l["lname"]
            decoderlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.decoder = nn.ModuleList(decoderlayers)


    def encode(self, x):

        for l in self.encoder:
            x = l(x)
        mu = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar


    def decode(self, z):
        for l in self.decoder:
            z = l(z)
        return z
        

    def forward(self, x):
        # encode
        mu, logvar = self.encode(x)

        # sampling
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda()
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)

        # decode
        outputs = self.decode(z)

        return outputs, mu, logvar



class TSYConditionalVariationalAutoEncoder(nn.Module):

    def __init__(self, netstructure, cudaflg=False):
        super(TSYConditionalVariationalAutoEncoder, self).__init__()

        # Check cuda
        self.cudaflg = cudaflg
        # Check net structure
        if isinstance(netstructure, str):
            with open(netstructure, "r") as f:
                print(f"net structure is loaded from `{netstructure}`")
                netstructure = json.load(f)
        # Prepare layer generator
        gl = u.GenerateLayer()

        # parameters
        self.Nhid = netstructure["Dimension of hidden variable"]
        self.Nout = netstructure["Dimension of output"]

        ##### Encoder 1 layers ################################################################################
        encoder1layers = []
        for l in netstructure["Encoder1"]:
            layername = l["lname"]
            encoder1layers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder1 = nn.ModuleList(encoder1layers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        Nin = self.encoder1[-2].out_features  # I need to sophisticate this part.
        self.encoder1_mean = nn.Linear(Nin, self.Nhid)
        self.encoder1_logvar = nn.Linear(Nin, self.Nhid)
        ######################################################################################################

        ##### Encoder 2 convolutional layers #################################################################
        encoder2convlayers = []
        for l in netstructure["Encoder2 convolutional"]:
            layername = l["lname"]
            encoder2convlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder2conv = nn.ModuleList(encoder2convlayers)
        #######################################################################################################

        ##### Encoder 2 fully-connected layers ################################################################
        encoder2linearlayers = []
        for l in netstructure["Encoder2 fully connected"]:
            layername = l["lname"]
            encoder2linearlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.encoder2linear = nn.ModuleList(encoder2linearlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        Nin = self.encoder2linear[-2].out_features  # I need to sophisticate this part.
        self.encoder2_mean = nn.Linear(Nin, self.Nhid)
        self.encoder2_logvar = nn.Linear(Nin, self.Nhid)
        #######################################################################################################
        ##### Decoder convolutional layers ####################################################################
        decoderconvlayers = []
        for l in netstructure["Decoder convolutional"]:
            layername = l["lname"]
            decoderconvlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.decoderconv = nn.ModuleList(decoderconvlayers)
        #######################################################################################################
        ##### Decoder fully connected layers ##################################################################
        decoderlinearlayers = []
        for l in netstructure["Decoder fully connected"]:
            layername = l["lname"]
            decoderlinearlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.decoderlinear = nn.ModuleList(decoderlinearlayers)
        Nin = self.decoderlinear[-2].out_features  # I need to sophisticated this part.
        self.decoder_mean = nn.Linear(Nin, self.Nout)
        self.decoder_logvar = nn.Linear(Nin, self.Nout)
        #########################################################################################################


    # Encode 1
    def encode1(self, x):
        ##### Encode to mean and log variance ###################################################################
        for l in self.encoder1:
            x = l(x)
        mu = self.encoder1_mean(x)
        logvar = self.encoder1_logvar(x)
        return mu, logvar
        #########################################################################################################

    # Encode 2
    def encode2(self, x, label):
        ##### Convolutional layers ############################
        for l in self.encoder2conv:
            x = l(x)
        #######################################################
        ##### Stack x and label ###############################
        y = torch.cat([x, label], dim=-1)
        y.retain_grad()
        #######################################################
        ##### Fully connected layers ##########################
        for l in self.encoder2linear:
            y = l(y)
        #######################################################
        ##### Encoder to mean and log variance ################
        mu = self.encoder2_mean(y)
        logvar = self.encoder2_logvar(y)
        return mu, logvar
        #######################################################

    # Decode
    def decode(self, z, y):
        ##### Convolutional layers ############################
        for l in self.decoderconv:
            y = l(y)
        #######################################################
        ##### Concate z and y #################################
        y = torch.cat([y,z], dim=1)
        y.retain_grad()
        #######################################################
        ##### Get mean and log variance #######################
        for l in self.decoderlinear:
            y = l(y)
        mu = self.decoder_mean(y)
        logvar = self.decoder_logvar(y)
        return mu, logvar
        #######################################################

    # Forward calculation
    def forward_training(self, y, label):
        # Encode the input data into the posterior's parameters
        mu1, logvar1 = self.encode1(y)
        mu2, logvar2 = self.encode2(y, label)
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar2)
        if self.cudaflg: eps = eps.cuda()
        std2 = logvar2.mul(0.5).exp_()
        z = eps.mul(std2).add_(mu2)
        # Decode
        mu_x, logvar_x = self.decode(z, y)
        return mu_x, logvar_x, mu1, logvar1, mu2, logvar2

    # prediction
    def forward_prediction(self, y):
        # Encode the input into the posterior's parameters
        mu, logvar = self.encode1(y)        
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda()
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)
        # Decode
        mu_x, logvar_x = self.decode(z, y)
        return mu_x, logvar_x, mu, logvar

    # inference
    def forward_inference(self, y, Nloop=1000):
        """
        Parameters
        ------------------------------
        y: torch.tensor
            An input signal. The shape is (1, Ndata).
        
        Nloop: int
            The number of samples to be sampled.
            Default 1000

        Returns
        -------------------------------
        outlist: numpy.ndarray
            The samples. The shape is (Nloop, Npred), where Nphys is the number of dimensions of predicted parameters.
        """

        # Encode
        mu, logvar = self.encode1(y)
        std_enc = logvar.mul(0.5).exp_()
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.empty((Nloop, self.Nhid)).normal_(0.0, 1.0)
        if self.cudaflg: eps = eps.cuda()
        z = eps.mul(std_enc).add_(mu)
        # Decode
        mu_x, logvar_x = self.decode(z, torch.tile(y, dims=(Nloop, 1)))
        # Random sampling
        eps = torch.randn_like(mu_x)
        if self.cudaflg: eps = eps.cuda()
        std_dec = logvar_x.mul(0.5).exp_()
        pred = eps.mul(std_dec).add_(mu_x)
        if self.cudaflg:
            pred = pred.cpu()
        """
        if 'outlist' in locals():
            outlist = np.vstack((outlist, pred.detach().numpy()))
        else:
            outlist = pred.detach().numpy()
        """
        return pred

    def _get_output_of_encoder(self, y):
        # Encode the input into the posterior's parameters
        mu, logvar = self.encode1(y)        
        return mu, logvar



"""
CVAE class is implemented in an old manner.
Its structure is specified in a different manner from the current style.
But it's still useful.
For specification, please see `example/CVAEtutorial.ipynb`.
"""

class CVAE(nn.Module):

    def __init__(self, in_features, layer_params, hidden_features, phys_features, cudaflg=True):
        super(CVAE, self).__init__()

        '''
        Conditional Variational Auto Encoder Class
        All layers are fully-connected layers.

        Parameters
        ------------------------------------------------------
        hidden_features : int
            the number of the hidden variables
        
        layer_params : dict
            The keys are "encoder1", "encoder2" and "decoder".
            Each value is a list having elements [output dim 1, output dim 2, ,,, , output dim n].
            IF you set `hidden_feature = 2`,
            n-th layer is connected to the nn.Linear(output dim n, 2)


        phys_features : int
            the number of the parameters to be estimated.

        cudaflg : boolign
            Whether using GPU or CPU. If true, all calculation will be done with GPU.
            Default True.

        '''

        self.cudaflg = cudaflg

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.phys_features = phys_features

        self.enc1_params = layer_params['encoder1']
        self.enc2_params = layer_params['encoder2']
        self.dec_params = layer_params['decoder']

        self.N_enc1_layer = len(layer_params['encoder1'])
        self.N_enc2_layer = len(layer_params['encoder2'])
        self.N_dec_layer = len(layer_params['decoder'])

        encode1_layers = []
        encode2_layers = []
        decode_layers = []

        encode1_layers.append(nn.Linear(self.in_features, self.enc1_params[0]))
        for i in range(self.N_enc1_layer - 1):
            encode1_layers.append(nn.Linear(self.enc1_params[i], self.enc1_params[i+1]))
        self.encoder1 = nn.ModuleList(encode1_layers)
        self.encoder1_mu = nn.Linear(self.enc1_params[self.N_enc1_layer-1], self.hidden_features)
        self.encoder1_logvar = nn.Linear(self.enc1_params[self.N_enc1_layer-1], self.hidden_features)
 
        encode2_layers.append(nn.Linear(self.in_features + self.phys_features, self.enc2_params[0]))
        for i in range(self.N_enc2_layer - 1):
            encode2_layers.append(nn.Linear(self.enc2_params[i], self.enc2_params[i+1]))
        self.encoder2 = nn.ModuleList(encode2_layers)           
        self.encoder2_mu = nn.Linear(self.enc2_params[self.N_enc2_layer-1], self.hidden_features)
        self.encoder2_logvar = nn.Linear(self.enc2_params[self.N_enc2_layer-1], self.hidden_features)
 
        decode_layers.append(nn.Linear(self.hidden_features + self.in_features, self.dec_params[0]))
        for i in range(self.N_dec_layer - 1):
            decode_layers.append(nn.Linear(self.dec_params[i], self.dec_params[i+1]))
        self.decoder = nn.ModuleList(decode_layers)
        self.decoder_mu = nn.Linear(self.dec_params[self.N_dec_layer-1], self.phys_features)
        self.decoder_logvar = nn.Linear(self.dec_params[self.N_dec_layer-1], self.phys_features)

        print("TSYNet.py: Neural network is successfully initialized.")


    def encode_1(self, y):
        '''
        Encode module 1
        This is trained for inference.

        Parameters
        ---------------------------------------------------------
        y : torch.tensor

        Returns
        ---------------------------------------------------------
        mu : torch.tensor
            The average
        logvar : torch.tensor
            The logvar is the log(sigma).
        '''
        
        # Use the layers of encoder

        for i in range(len(self.encoder1)):
            y = F.relu(self.encoder1[i](y))
        mu = self.encoder1_mu(y)
        logvar = self.encoder1_logvar(y)

        return mu, logvar
 

    def encode_2(self, y, label):
        '''
        Encode module

        Parameters
        ---------------------------------------------------------
        y : torch.tensor
            The GW signal.

        label : torch.tensor
            The physical parameters corresponding to GW signal.

        Returns
        ---------------------------------------------------------
        mu : torch.tensor
            The average
        logvar : torch.tensor
            The logvar is the log(sigma).
        '''

        y = torch.cat([y, label], dim=-1)
        y.retain_grad()
        for i in range(len(self.encoder2)):
            y = F.relu(self.encoder2[i](y))
        mu = self.encoder2_mu(y)
        logvar = self.encoder2_logvar(y)

        return mu, logvar
        



    def decode(self, z, y):

        '''
        Decode module

        Parameters
        -----------------------------------
        z : torch.tensor
            The hidden variable sampled from Gaussian.

        y : torch.tensor
            The compressed GW signal using self.conv_compresser

        '''

        y = torch.cat([y,z], dim=1)
        y.retain_grad()
        for i in range(len(self.decoder)):
            y = F.relu(self.decoder[i](y))
        mu = self.decoder_mu(y)
        logvar = self.decoder_logvar(y)

        return mu, logvar
        
    
    def forward_training(self, y, label):

        # Encode the input into the posterior's parameters
        mu1, logvar1 = self.encode_1(y)
        mu2, logvar2 = self.encode_2(y, label)
                
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar2)
        if self.cudaflg: eps = eps.cuda()
        std2 = logvar2.mul(0.5).exp_()
        z = eps.mul(std2).add_(mu2)

        # Decode
        mu_x, logvar_x = self.decode(z, y)

        return mu_x, logvar_x, mu1, logvar1, mu2, logvar2


    def forward_prediction(self, y):

        # Encode the input into the posterior's parameters
        mu, logvar = self.encode_1(y)
        
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda()
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)

        # Decode
        mu_x, logvar_x = self.decode(z, y)

        return mu_x, logvar_x, mu, logvar



    def forward_inference(self, y, Nloop=1000):

        """
        Parameters
        ------------------------------
        y: torch.tensor
            An input signal. The shape is (1, Ndata).
        
        Nloop: int
            The number of samples to be sampled.
            Default 1000

        Returns
        -------------------------------
        outlist: numpy.ndarray
            The samples. The shape is (Nloop, Npred), where Nphys is the number of dimensions of predicted parameters.
        
        """

        # Encode
        mu, logvar = self.encode_1(y)
        std_enc = logvar.mul(0.5).exp_()
        
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.empty((Nloop, self.hidden_features)).normal_(0.0, 1.0)
        if self.cudaflg: eps = eps.cuda()
        z = eps.mul(std_enc).add_(mu)

        # Decode
        mu_x, logvar_x = self.decode(z, torch.tile(y, dims=(Nloop, 1)))

        # Random sampling
        eps = torch.randn_like(mu_x)
        if self.cudaflg: eps = eps.cuda()
        std_dec = logvar_x.mul(0.5).exp_()
        pred = eps.mul(std_dec).add_(mu_x)

        if self.cudaflg:
            pred = pred.cpu()

        """
        if 'outlist' in locals():
            outlist = np.vstack((outlist, pred.detach().numpy()))
        else:
            outlist = pred.detach().numpy()
        """

        return pred


    def _get_output_of_encoder(self, y):

        # Encode the input into the posterior's parameters
        mu, logvar = self.encode_1(y)
        
        return mu, logvar







if __name__ == "__main__":

    from collections import OrderedDict
    from torchinfo import summary
    import json

    params = "testconfig.json"
    net = TSYVariationalAutoEncoder(params)
    print("net is defined.")
    summary(net, inputsize=(1,2,1000))

    # trial
    inputs = torch.empty((10, 2, 1000)).normal_()
    print(f"Input size: {inputs.size()}")
    outputs = net(inputs)
    print(f"Output size: {outputs.size()}")
