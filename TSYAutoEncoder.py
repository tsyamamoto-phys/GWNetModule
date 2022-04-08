"""
TSYAutoEncoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from scipy.linalg import sqrtm
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

    def __init__(self, netstructure, cudaflg=False, device=None, kidx=-2, covarianceflg=False):
        super(TSYConditionalVariationalAutoEncoder, self).__init__()

        # Check cuda
        self.cudaflg = cudaflg
        self.covarianceflg = covarianceflg
        self.gpudevice = device
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
        Nin = self.encoder1[kidx].out_features  # I need to sophisticate this part.
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
        Nin = self.encoder2linear[kidx].out_features  # I need to sophisticate this part.
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
        Nin = self.decoderlinear[kidx].out_features  # I need to sophisticated this part.
        self.decoder_mean = nn.Linear(Nin, self.Nout)
        self.decoder_logvar = nn.Linear(Nin, self.Nout)
        if self.covarianceflg:
            self.decoder_cov = nn.Linear(Nin, 1)
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
        #outputlist = []
        ##### Convolutional layers ############################
        for l in self.encoder2conv:
            x = l(x)
            #outputlist.append(x)
        #######################################################
        ##### Stack x and label ###############################
        y = torch.cat([x, label], dim=-1)
        if y.requires_grad: y.retain_grad()
        #######################################################
        ##### Fully connected layers ##########################
        for l in self.encoder2linear:
            y = l(y)
            #outputlist.append(y)
        #######################################################
        ##### Encoder to mean and log variance ################
        mu = self.encoder2_mean(y)
        logvar = self.encoder2_logvar(y)
        return mu, logvar#, outputlist
        #######################################################

    # Decode
    def decode(self, z, y):
        ##### Convolutional layers ############################
        for l in self.decoderconv:
            y = l(y)
        #######################################################
        ##### Concate z and y #################################
        y = torch.cat([y,z], dim=1)
        if y.requires_grad: y.retain_grad()
        #######################################################
        ##### Get mean and log variance #######################
        for l in self.decoderlinear:
            y = l(y)
        mu = self.decoder_mean(y)
        logvar = self.decoder_logvar(y)
        if self.covarianceflg:
            r = self.decoder_cov(y)
            r = torch.tanh(r)
            return mu, logvar, r
        else:
            return mu, logvar
        #######################################################

    # Forward calculation
    def forward_training(self, y, label):
        # Encode the input data into the posterior's parameters
        mu1, logvar1 = self.encode1(y)
        mu2, logvar2 = self.encode2(y, label)
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar2)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)
        std2 = logvar2.mul(0.5).exp_()
        z = eps.mul(std2).add_(mu2)
        # Decode
        if self.covarianceflg:
            mu_x, logvar_x, r = self.decode(z, y)
            return mu_x, logvar_x, r, mu1, logvar1, mu2, logvar2
        else:
            mu_x, logvar_x = self.decode(z, y)
            return mu_x, logvar_x, mu1, logvar1, mu2, logvar2
        

    # prediction
    def forward_prediction(self, y):
        # Encode the input into the posterior's parameters
        mu, logvar = self.encode1(y)        
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)
        # Decode
        if self.covarianceflg:
            mu_x, logvar_x, r = self.decode(z, y)
            return mu_x, logvar_x, r, mu, logvar
        else:
            mu_x, logvar_x = self.decode(z, y)
            return mu_x, logvar_x, mu, logvar

    # inference
    def forward_inference(self, y, Nloop=1000, Nbatch=None):
        """
        Parameters
        ------------------------------
        y: torch.tensor
            An input signal. The shape is (1, Nchannel, Ndata).
        
        Nloop: int
            The number of samples to be sampled.
            Default 1000

        Nbatch: int
            The batch size. If it is None, Nbatch=Nloop.
            Default for None.

        Returns
        -------------------------------
        outlist: numpy.ndarray
            The samples. The shape is (Nloop, Npred), where Nphys is the number of dimensions of predicted parameters.
        """

        if Nbatch is None: Nbatch = Nloop
        kbatch = Nloop // Nbatch
        Nsize = kbatch * Nbatch
        predlist = torch.empty((Nsize, self.Nout))
        ydim = len(y.size()) - 1
        dims = [Nbatch]
        for _ in range(ydim): dims.append(1)
        # Encode
        mu, logvar = self.encode1(y)
        ytiled = torch.tile(y, dims=dims)
        std_enc = logvar.mul(0.5).exp_()
        for k in range(kbatch):
            # Sampling from the standard normal distribution and reparametrize.
            eps = torch.empty((Nbatch, self.Nhid)).normal_(0.0, 1.0)
            if self.cudaflg: eps = eps.cuda(self.gpudevice)
            z = eps.mul(std_enc).add_(mu)
            # Decode
            if self.covarianceflg:
                mu_x, logvar_x, r_x = self.decode(z, ytiled)
                pred = self._get_sample_from_multivariateGaussian_with_covariance(mu_x, logvar_x, r_x)
            else:
                mu_x, logvar_x = self.decode(z, ytiled)
                # Random sampling
                eps = torch.randn_like(mu_x)
                if self.cudaflg: eps = eps.cuda(self.gpudevice)
                std_dec = logvar_x.mul(0.5).exp_()
                pred = eps.mul(std_dec).add_(mu_x)

            if self.cudaflg:
                predlist[k * Nbatch : (k+1) * Nbatch] = pred.cpu()

            """
            if 'outlist' in locals():
                outlist = np.vstack((outlist, pred.detach().numpy()))
            else:
                outlist = pred.detach().numpy()
            """
        return predlist

    def _get_output_of_encoder(self, y):
        # Encode the input into the posterior's parameters
        mu, logvar = self.encode1(y)        
        return mu, logvar

    def _get_sample_from_multivariateGaussian_with_covariance(self, mu, logvar, r):
        var = torch.exp(logvar)
        std = torch.sqrt(var)
        eps = torch.randn_like(mu)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)

        trSigma = torch.sum(var, dim=1, keepdim=True)
        detSigma = (1.0 - r**2.0) * torch.prod(var, dim=1, keepdim=True)
        s_factor = torch.sqrt(detSigma)
        t_factor = torch.sqrt(trSigma + 2.0 * s_factor)
        pred = torch.empty_like(mu)

        pred[:,0] = mu[:,0] + ((var[:,0] + s_factor[:,0]) * eps[:,0] + r[:,0] * std[:,0] * std[:,1] * eps[:,1]) / t_factor[:,0
        ]
        pred[:,1] = mu[:,1] + (r[:,0] * std[:,0] * std[:,1] * eps[:,0] + (var[:,1] * s_factor[:,0]) * eps[:,1]) / t_factor[:,0]
        return pred

    
    
class TSYConditionalVariationalAutoEncoder_SharedConvolutionalLayers(nn.Module):

    def __init__(self, netstructure, cudaflg=False, device=None, kidx=-2):
        super(TSYConditionalVariationalAutoEncoder_SharedConvolutionalLayers, self).__init__()

        # Check cuda
        self.cudaflg = cudaflg
        self.gpudevice = device
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

        ##### Convolutional layers (shared)#####################################################################
        convlayers = []
        for l in netstructure["Convolutional"]:
            layername = l["lname"]
            convlayers.append(gl.LayersDict[layername](**(l["params"])))
        convlayers.append(nn.Flatten())
        self.convlayers = nn.ModuleList(convlayers)
        ########################################################################################################
        
        ##### Prior ############################################################################################
        priorlayers = []
        for l in netstructure["Prior"]:
            layername = l["lname"]
            priorlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.priorlayers = nn.ModuleList(priorlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        # prior output: mean
        priormu_layers = []
        for l in netstructure["Prior mean"]:
            layername = l["lname"]
            priormu_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.priormu_layers = nn.ModuleList(priormu_layers)
        # prior output: logvar
        priorlogvar_layers = []
        for l in netstructure["Prior logvar"]:
            layername = l["lname"]
            priorlogvar_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.priorlogvar_layers = nn.ModuleList(priorlogvar_layers)
        #######################################################################################################

        ##### Recognition ################################################################
        recognitionlayers = []
        for l in netstructure["Recognition"]:
            layername = l["lname"]
            recognitionlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.recognitionlayers = nn.ModuleList(recognitionlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        # recognition output: mean
        recognitionmu_layers = []
        for l in netstructure["Recognition mean"]:
            layername = l["lname"]
            recognitionmu_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.recognitionmu_layers = nn.ModuleList(recognitionmu_layers)
        # recognition output: logvar
        recognitionlogvar_layers = []
        for l in netstructure["Recognition logvar"]:
            layername = l["lname"]
            recognitionlogvar_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.recognitionlogvar_layers = nn.ModuleList(recognitionlogvar_layers)
        #######################################################################################################

        ##### Generator #######################################################################################
        generatorlayers = []
        for l in netstructure["Generator"]:
            layername = l["lname"]
            generatorlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.generatorlayers = nn.ModuleList(generatorlayers)
        # generator output: mean
        generatormu_layers = []
        for l in netstructure["Generator mean"]:
            layername = l["lname"]
            generatormu_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.generatormu_layers = nn.ModuleList(generatormu_layers)
        # generator output: logvar
        generatorlogvar_layers = []
        for l in netstructure["Generator logvar"]:
            layername = l["lname"]
            generatorlogvar_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.generatorlogvar_layers = nn.ModuleList(generatorlogvar_layers)
        ########################################################################################################

    def _prior_mean(self, x):
        for l in self.priormu_layers:
            x = l(x)
        return x

    def _prior_logvar(self, x):
        for l in self.priorlogvar_layers:
            x = l(x)
        return x

    def _recognition_mean(self, x):
        for l in self.recognitionmu_layers:
            x = l(x)
        return x

    def _recognition_logvar(self, x):
        for l in self.recognitionlogvar_layers:
            x = l(x)
        return x

    def _generator_mean(self, x):
        for l in self.generatormu_layers:
            x = l(x)
        return x

    def _generator_logvar(self, x):
        for l in self.generatorlogvar_layers:
            x = l(x)
        return x
            
    # Convolution and Flattening
    def convolution(self, x):
        for l in self.convlayers:
            x = l(x)
        return x 
    
    # prior
    def prior(self, x):
        # forward calculation
        for l in self.priorlayers:
            x = l(x)
        mu = self._prior_mean(x)
        logvar = self._prior_logvar(x)
        return mu, logvar

    # recognition
    def recognition(self, x, label):
        # Stack strain and label
        y = torch.cat([x, label], dim=-1)
        if y.requires_grad: y.retain_grad()
        # forward calculation
        for l in self.recognitionlayers:
            y = l(y)
        mu = self._recognition_mean(y)
        logvar = self._recognition_logvar(y)
        return mu, logvar

    # generator
    def generator(self, z, y):
        # Stack strain and latent variables
        y = torch.cat([y,z], dim=1)
        if y.requires_grad: y.retain_grad()
        # forward calculation
        for l in self.generatorlayers:
            y = l(y)
        mu = self._generator_mean(y)
        logvar = self._generator_logvar(y)
        return mu, logvar

    # Forward calculation
    def forward_training(self, y, label):
        # convolution and flattening
        y = self.convolution(y)
        # Encode the input data into the posterior's parameters
        mu1, logvar1 = self.prior(y)
        mu2, logvar2 = self.recognition(y, label)
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar2)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)
        std2 = logvar2.mul(0.5).exp_()
        z = eps.mul(std2).add_(mu2)
        # Decode
        mu_x, logvar_x = self.generator(z, y)
        return mu_x, logvar_x, mu1, logvar1, mu2, logvar2

    # prediction
    def forward_prediction(self, y):
        # convolution and flattening 
        y = self.convolution(y)
        # Encode the input into the posterior's parameters
        mu, logvar = self.prior(y)        
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)
        # Decode
        mu_x, logvar_x = self.generator(z, y)
        return mu_x, logvar_x, mu, logvar

    # inference
    def forward_inference(self, y, Nloop=1000, Nbatch=None):
        """
        Parameters
        ------------------------------
        y: torch.tensor
            An input signal. The shape is (1, Nchannel, Ndata).
        
        Nloop: int
            The number of samples to be sampled.
            Default 1000

        Nbatch: int
            The batch size. If it is None, Nbatch=Nloop.
            Default for None.

        Returns
        -------------------------------
        outlist: numpy.ndarray
            The samples. The shape is (Nloop, Npred), where Nphys is the number of dimensions of predicted parameters.
        """

        if Nbatch is None: Nbatch = Nloop
        kbatch = Nloop // Nbatch
        Nsize = kbatch * Nbatch
        predlist = torch.empty((Nsize, self.Nout))
        ydim = len(y.size()) - 1
        dims = [Nbatch]
        for _ in range(ydim): dims.append(1)
        # Convolution
        y = self.convolution(y)
        # Encode
        mu, logvar = self.prior(y)
        ytiled = torch.tile(y, dims=dims)
        std_enc = logvar.mul(0.5).exp_()
        flatten = nn.Flatten()
        ytiled = flatten(ytiled)
        for k in range(kbatch):
            # Sampling from the standard normal distribution and reparametrize.
            eps = torch.empty((Nbatch, self.Nhid)).normal_(0.0, 1.0)
            if self.cudaflg: eps = eps.cuda(self.gpudevice)
            z = eps.mul(std_enc).add_(mu)
            # Decode
            mu_x, logvar_x = self.generator(z, ytiled)
            # Random sampling
            eps = torch.randn_like(mu_x)
            if self.cudaflg: eps = eps.cuda(self.gpudevice)
            std_dec = logvar_x.mul(0.5).exp_()
            pred = eps.mul(std_dec).add_(mu_x)
            if self.cudaflg:
                predlist[k * Nbatch : (k+1) * Nbatch] = pred.cpu()

            """
            if 'outlist' in locals():
                outlist = np.vstack((outlist, pred.detach().numpy()))
            else:
                outlist = pred.detach().numpy()
            """
        return predlist

    def _get_output_of_encoder(self, y):
        # convolution and flattening
        y = self.convolution(y)
        # Encode the input into the posterior's parameters
        mu, logvar = self.prior(y)        
        return mu, logvar






class TSYConditionalVariationalAutoEncoder_BernoulliProbability_SharedConvolutionalLayers(nn.Module):

    def __init__(self, netstructure, cudaflg=False, device=None, kidx=-2):
        super(TSYConditionalVariationalAutoEncoder_BernoulliProbability_SharedConvolutionalLayers, self).__init__()

        # Check cuda
        self.cudaflg = cudaflg
        self.gpudevice = device
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

        ##### Convolutional layers (shared)#####################################################################
        convlayers = []
        for l in netstructure["Convolutional"]:
            layername = l["lname"]
            convlayers.append(gl.LayersDict[layername](**(l["params"])))
        convlayers.append(nn.Flatten())
        self.convlayers = nn.ModuleList(convlayers)
        ########################################################################################################
        
        ##### Prior ############################################################################################
        priorlayers = []
        for l in netstructure["Prior"]:
            layername = l["lname"]
            priorlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.priorlayers = nn.ModuleList(priorlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        # prior output: mean
        priormu_layers = []
        for l in netstructure["Prior mean"]:
            layername = l["lname"]
            priormu_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.priormu_layers = nn.ModuleList(priormu_layers)
        # prior output: logvar
        priorlogvar_layers = []
        for l in netstructure["Prior logvar"]:
            layername = l["lname"]
            priorlogvar_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.priorlogvar_layers = nn.ModuleList(priorlogvar_layers)
        #######################################################################################################

        ##### Recognition ################################################################
        recognitionlayers = []
        for l in netstructure["Recognition"]:
            layername = l["lname"]
            recognitionlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.recognitionlayers = nn.ModuleList(recognitionlayers)
        # output of Encoder should be divided into a mean and a variance of a Gaussian distribution.
        # recognition output: mean
        recognitionmu_layers = []
        for l in netstructure["Recognition mean"]:
            layername = l["lname"]
            recognitionmu_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.recognitionmu_layers = nn.ModuleList(recognitionmu_layers)
        # recognition output: logvar
        recognitionlogvar_layers = []
        for l in netstructure["Recognition logvar"]:
            layername = l["lname"]
            recognitionlogvar_layers.append(gl.LayersDict[layername](**(l["params"])))
        self.recognitionlogvar_layers = nn.ModuleList(recognitionlogvar_layers)
        #######################################################################################################

        ##### Generator #######################################################################################
        # Generator outputs the Bernoulli probability
        # Each elements of an output vector are assumed to take a value within [0.0, 1.0].
        generatorlayers = []
        for l in netstructure["Generator"]:
            layername = l["lname"]
            generatorlayers.append(gl.LayersDict[layername](**(l["params"])))
        self.generatorlayers = nn.ModuleList(generatorlayers)    # Dimension of an output must coincide with that of a target vector.
        ########################################################################################################

    def _prior_mean(self, x):
        for l in self.priormu_layers:
            x = l(x)
        return x

    def _prior_logvar(self, x):
        for l in self.priorlogvar_layers:
            x = l(x)
        return x

    def _recognition_mean(self, x):
        for l in self.recognitionmu_layers:
            x = l(x)
        return x

    def _recognition_logvar(self, x):
        for l in self.recognitionlogvar_layers:
            x = l(x)
        return x
            
    # Convolution and Flattening
    def convolution(self, x):
        for l in self.convlayers:
            x = l(x)
        return x 
    
    # prior
    def prior(self, x):
        # forward calculation
        for l in self.priorlayers:
            x = l(x)
        mu = self._prior_mean(x)
        logvar = self._prior_logvar(x)
        return mu, logvar

    # recognition
    def recognition(self, x, label):
        # Stack strain and label
        y = torch.cat([x, label], dim=-1)
        if y.requires_grad: y.retain_grad()
        # forward calculation
        for l in self.recognitionlayers:
            y = l(y)
        mu = self._recognition_mean(y)
        logvar = self._recognition_logvar(y)
        return mu, logvar

    # generator
    def generator(self, z, y):
        # Stack strain and latent variables
        y = torch.cat([y,z], dim=1)
        if y.requires_grad: y.retain_grad()
        # forward calculation
        for l in self.generatorlayers:
            y = l(y)
        return y

    # Forward calculation
    def forward_training(self, y, label):
        # convolution and flattening
        y = self.convolution(y)
        # Encode the input data into the posterior's parameters
        mu1, logvar1 = self.prior(y)
        mu2, logvar2 = self.recognition(y, label)
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar2)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)
        std2 = logvar2.mul(0.5).exp_()
        z = eps.mul(std2).add_(mu2)
        # Decode
        preds = self.generator(z, y)
        return preds, mu1, logvar1, mu2, logvar2

    # prediction
    def forward_prediction(self, y):
        # convolution and flattening 
        y = self.convolution(y)
        # Encode the input into the posterior's parameters
        mu, logvar = self.prior(y)        
        # Sampling from the standard normal distribution and reparametrize.
        eps = torch.randn_like(logvar)
        if self.cudaflg: eps = eps.cuda(self.gpudevice)
        std = logvar.mul(0.5).exp_()
        z = eps.mul(std).add_(mu)
        # Decode
        preds = self.generator(z, y)
        return preds, mu, logvar

    # inference
    def forward_inference(self, y, Nloop=1000, Nbatch=None):
        """
        Parameters
        ------------------------------
        y: torch.tensor
            An input signal. The shape is (1, Nchannel, Ndata).
        
        Nloop: int
            The number of samples to be sampled.
            Default 1000

        Nbatch: int
            The batch size. If it is None, Nbatch=Nloop.
            Default for None.

        Returns
        -------------------------------
        outlist: numpy.ndarray
            The samples. The shape is (Nloop, Npred), where Nphys is the number of dimensions of predicted parameters.
        """

        if Nbatch is None: Nbatch = Nloop
        kbatch = Nloop // Nbatch
        Nsize = kbatch * Nbatch
        predlist = torch.empty((Nsize, self.Nout))
        ydim = len(y.size()) - 1
        dims = [Nbatch]
        for _ in range(ydim): dims.append(1)
        # Convolution
        y = self.convolution(y)
        # Encode
        mu, logvar = self.prior(y)
        ytiled = torch.tile(y, dims=dims)
        std_enc = logvar.mul(0.5).exp_()
        flatten = nn.Flatten()
        ytiled = flatten(ytiled)
        for k in range(kbatch):
            # Sampling from the standard normal distribution and reparametrize.
            eps = torch.empty((Nbatch, self.Nhid)).normal_(0.0, 1.0)
            if self.cudaflg: eps = eps.cuda(self.gpudevice)
            z = eps.mul(std_enc).add_(mu)
            # Decode
            preds = self.generator(z, ytiled)
            if self.cudaflg:
                predlist[k * Nbatch : (k+1) * Nbatch] = preds.cpu()
        return predlist

    def _get_output_of_encoder(self, y):
        # convolution and flattening
        y = self.convolution(y)
        # Encode the input into the posterior's parameters
        mu, logvar = self.prior(y)
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
    net = TSYConditionalVariationalAutoEncoder(params)
    print("net is defined.")
    summary(net, inputsize=(1,2,1000))

    # trial
    inputs = torch.empty((10, 2, 1000)).normal_()
    print(f"Input size: {inputs.size()}")
    outputs = net(inputs)
    print(f"Output size: {outputs.size()}")
