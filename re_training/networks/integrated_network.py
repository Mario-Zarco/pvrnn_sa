"""
Integrated network composed of multi-layered PV-RNN dealing with vision and proprioception simultaneously which
- has vertical connection between layers
- is compatible with mini-batch learning
- samples prior from unit gaussian prior at t=0
Extended by Mario: an interoceptive dimension has been added
"""
import sys
sys.path.append("../")

from networks.pvrnn import PVRNN
from networks.output import Output
import torch
import torch.nn as nn


class Integrated(nn.Module):
    def __init__(self, net_param: dict, minibatch_size: int, n_minibatch: int, seq_len: int, motor_dim=2, vision_dim=1, intero_dim=1):
        super(Integrated, self).__init__()
        print("Initializing the network...")
        self.net_param       = net_param # hyper parameters are given with a dictionary
        self.minibatch_size  = minibatch_size
        self.n_minibatch     = n_minibatch
        self.seq_len     = seq_len
        
        self.motor_dim   = motor_dim
        self.vision_dim  = vision_dim
        self.intero_dim  = intero_dim

        self.device = "cpu"

        """initialize each module of the entire network.
        PV-RNN parts are loaded on CPU."""
        # associative module on CPU
        self.top = PVRNN(net_param["top_d_size"], net_param["top_z_size"], net_param["top_tau"], net_param["top_w"], net_param["top_w1"], net_param["top_wd1"], minibatch_size, n_minibatch, seq_len, device=self.device)

        # proprioception module on CPU
        self.prop = PVRNN(net_param["prop_d_size"], net_param["prop_z_size"], net_param["prop_tau"], net_param["prop_w"], net_param["prop_w1"], net_param["prop_wd1"], minibatch_size, n_minibatch, seq_len, input=True, input_dim=net_param["top_d_size"][-1], device=self.device)
        # proprioception output on CPU
        self.prop_out = Output(net_param["prop_d_size"][-1], motor_dim, act_func="tanh", device=self.device)

        # vision module - PV-RNN part on CPU
        self.vision = PVRNN(net_param["vision_d_size"], net_param["vision_z_size"], net_param["vision_tau"], net_param["vision_w"], net_param["vision_w1"], net_param["vision_wd1"], minibatch_size, n_minibatch, seq_len, input=True, input_dim=net_param["top_d_size"][-1], device=self.device)
        # vision module - PV-RNN part output on CPU
        self.vision_out = Output(net_param["vision_d_size"][-1], vision_dim, act_func="tanh", device=self.device)

        # interoception module - PV-RNN part on CPU
        self.intero = PVRNN(net_param["intero_d_size"], net_param["intero_z_size"], net_param["intero_tau"], net_param["intero_w"], net_param["intero_w1"], net_param["intero_wd1"], minibatch_size, n_minibatch, seq_len, input=True, input_dim=net_param["top_d_size"][-1], device=self.device)
        # interoception module - PV-RNN part output on CPU
        self.intero_out = Output(net_param["intero_d_size"][-1], vision_dim, act_func="tanh", device=self.device)

    def initialize(self, minibatch_ind: int):
        self.top.initialize(minibatch_ind)
        self.prop.initialize(minibatch_ind)
        self.vision.initialize(minibatch_ind)
        self.intero.initialize(minibatch_ind)

    # change the value of w and w1 in er
    def set_w(self, w_setting: dict):
        self.top.set_w(w_setting["top_w"], w_setting["top_w1"])
        self.prop.set_w(w_setting["prop_w"], w_setting["prop_w1"])
        self.vision.set_w(w_setting["vision_w"], w_setting["vision_w1"])
        self.intero.set_w(w_setting["intero_w"], w_setting["intero_w1"])

    # compute one time step forward computation with posterior
    def posterior_step(self, epo: int):
        self.top.posterior_step(epo, None)

        self.prop.posterior_step(epo, self.top.layers[-1].d)
        p = self.prop_out(self.prop.layers[-1].d)

        self.vision.posterior_step(epo, self.top.layers[-1].d)
        v = self.vision_out(self.vision.layers[-1].d)

        self.intero.posterior_step(epo, self.top.layers[-1].d)
        i = self.intero_out(self.intero.layers[-1].d)

        return p, v, i

    # compute one time step forward computation with prior
    def prior_step(self):
        self.top.prior_step()

        self.prop.prior_step(self.top.layers[-1].d)
        p = self.prop_out(self.prop.layers[-1].d)

        self.vision.prior_step(self.top.layers[-1].d)
        v = self.vision_out(self.vision.layers[-1].d)

        self.intero.prior_step(self.top.layers[-1].d)
        i = self.intero_out(self.intero.layers[-1].d)

        return p, v, i

    # # generate latent vision with posterior (lt) and embed a pixel image (l)
    # def posterior_enc_step(self, x):
    #     self.top.posterior_step()
    #     self.vision.posterior_step(self.top.layers[-1].d)
    #     lt = self.vision_out(self.vision.layers[-1].d)
    #     l  = self.encoder(x).cpu().view(self.minibatch_size, self.latent_size)
    #     return l, lt

    # generate proprioception and vision with posterior
    def posterior_forward(self, epo: int, minibatch_ind: int):
        ps = torch.zeros(self.minibatch_size, self.seq_len, self.motor_dim, device=self.device)
        vs = torch.zeros(self.minibatch_size, self.seq_len, self.vision_dim, device=self.device)
        ins = torch.zeros(self.minibatch_size, self.seq_len, self.intero_dim, device=self.device)

        self.initialize(minibatch_ind)
        
        top_kl    = torch.zeros(len(self.net_param["top_z_size"]), device=self.device)
        prop_kl   = torch.zeros(len(self.net_param["prop_z_size"]), device=self.device)
        vision_kl = torch.zeros(len(self.net_param["vision_z_size"]), device=self.device)
        intero_kl = torch.zeros(len(self.net_param["intero_z_size"]), device=self.device)

        top_wkl    = torch.zeros(len(self.net_param["top_z_size"]), device=self.device)
        prop_wkl   = torch.zeros(len(self.net_param["prop_z_size"]), device=self.device)
        vision_wkl = torch.zeros(len(self.net_param["vision_z_size"]), device=self.device)
        intero_wkl = torch.zeros(len(self.net_param["intero_z_size"]), device=self.device)

        # TOP
        top_d, top_mu_p, top_mu_q, top_sigma_p, top_sigma_q, top_kl_step, top_a, top_init_h, top_init_h_mu, top_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.top.n_layer):
            top_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.d_size[l], device=self.device))
            top_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device=self.device))
            top_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device=self.device))
            top_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device=self.device))
            top_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device=self.device))
            top_kl_step.append(torch.zeros(self.seq_len, device=self.device))
            top_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.top.z_size[l], device=self.device))
            top_init_h.append(torch.zeros(self.minibatch_size, self.top.d_size[l], device=self.device))
            top_init_h_mu.append(torch.zeros(self.top.d_size[l], device=self.device))
            top_bias.append(torch.zeros(self.top.d_size[l], device=self.device))

        # PROPRIOCEPTION
        prop_d, prop_mu_p, prop_mu_q, prop_sigma_p, prop_sigma_q, prop_kl_step, prop_a, prop_init_h, prop_init_h_mu, prop_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.prop.n_layer):
            prop_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.d_size[l], device=self.device))
            prop_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device=self.device))
            prop_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device=self.device))
            prop_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device=self.device))
            prop_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device=self.device))
            prop_kl_step.append(torch.zeros(self.seq_len, device=self.device))
            prop_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.prop.z_size[l], device=self.device))
            prop_init_h.append(torch.zeros(self.minibatch_size, self.prop.d_size[l], device=self.device))
            prop_init_h_mu.append(torch.zeros(self.prop.d_size[l], device=self.device))
            prop_bias.append(torch.zeros(self.prop.d_size[l], device=self.device))

        # EXTEROCEPTION
        vision_d, vision_mu_p, vision_mu_q, vision_sigma_p, vision_sigma_q, vision_kl_step, vision_a, vision_init_h, vision_init_h_mu, vision_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.vision.n_layer):
            vision_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.d_size[l], device=self.device))
            vision_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device=self.device))
            vision_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device=self.device))
            vision_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device=self.device))
            vision_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device=self.device))
            vision_kl_step.append(torch.zeros(self.seq_len, device=self.device))
            vision_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.vision.z_size[l], device=self.device))
            vision_init_h.append(torch.zeros(self.minibatch_size, self.vision.d_size[l], device=self.device))
            vision_init_h_mu.append(torch.zeros(self.vision.d_size[l], device=self.device))
            vision_bias.append(torch.zeros(self.vision.d_size[l], device=self.device))

        # INTEROCEPTION
        intero_d, intero_mu_p, intero_mu_q, intero_sigma_p, intero_sigma_q, intero_kl_step, intero_a, intero_init_h, intero_init_h_mu, intero_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.prop.n_layer):
            intero_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.intero.d_size[l], device=self.device))
            intero_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.intero.z_size[l], device=self.device))
            intero_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.intero.z_size[l], device=self.device))
            intero_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.intero.z_size[l], device=self.device))
            intero_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.intero.z_size[l], device=self.device))
            intero_kl_step.append(torch.zeros(self.seq_len, device=self.device))
            intero_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.intero.z_size[l], device=self.device))
            intero_init_h.append(torch.zeros(self.minibatch_size, self.intero.d_size[l], device=self.device))
            intero_init_h_mu.append(torch.zeros(self.intero.d_size[l], device=self.device))
            intero_bias.append(torch.zeros(self.intero.d_size[l], device=self.device))
        
        for t in range(self.seq_len):
            ps[:, t, :], vs[:, t, :], ins[:, t, :] = self.posterior_step(epo)
            
            # TOP
            for l, layer in enumerate(self.top.layers):
                top_d[l][:, t, :] = layer.d
                top_mu_p[l][:, t, :] = layer.mu_p
                top_mu_q[l][:, t, :] = layer.mu_q
                top_sigma_p[l][:, t, :] = layer.sigma_p
                top_sigma_q[l][:, t, :] = layer.sigma_q
                top_kl_step[l][t] = layer.kl
            
            # PROPRIOCEPTION
            for l, layer in enumerate(self.prop.layers):
                prop_d[l][:, t, :] = layer.d
                prop_mu_p[l][:, t, :] = layer.mu_p
                prop_mu_q[l][:, t, :] = layer.mu_q
                prop_sigma_p[l][:, t, :] = layer.sigma_p
                prop_sigma_q[l][:, t, :] = layer.sigma_q
                prop_kl_step[l][t] = layer.kl

            # EXTEROCEPTION
            for l, layer in enumerate(self.vision.layers):
                vision_d[l][:, t, :] = layer.d
                vision_mu_p[l][:, t, :] = layer.mu_p
                vision_mu_q[l][:, t, :] = layer.mu_q
                vision_sigma_p[l][:, t, :] = layer.sigma_p
                vision_sigma_q[l][:, t, :] = layer.sigma_q
                vision_kl_step[l][t] = layer.kl

            # INTEROCEPTION
            for l, layer in enumerate(self.intero.layers):
                intero_d[l][:, t, :] = layer.d
                intero_mu_p[l][:, t, :] = layer.mu_p
                intero_mu_q[l][:, t, :] = layer.mu_q
                intero_sigma_p[l][:, t, :] = layer.sigma_p
                intero_sigma_q[l][:, t, :] = layer.sigma_q
                intero_kl_step[l][t] = layer.kl
            
            top_kl     += self.top.kl
            prop_kl    += self.prop.kl
            vision_kl  += self.vision.kl
            intero_kl  += self.intero.kl

            top_wkl    += self.top.wkl
            prop_wkl   += self.prop.wkl
            vision_wkl += self.vision.wkl
            intero_wkl += self.intero.wkl
        
        top_wnll_init_h = self.top.wnll_init_h
        prop_wnll_init_h = self.prop.wnll_init_h 
        vision_wnll_init_h = self.vision.wnll_init_h
        intero_wnll_init_h = self.intero.wnll_init_h
        
        # TOP
        for l, layer in enumerate(self.top.layers):
            top_a[l] = self.top.state_dict()["layers." + str(l) + ".A.0"]
            top_init_h[l] = self.top.state_dict()["layers." + str(l) + ".init_h.0"]
            top_init_h_mu[l] = self.top.state_dict()["layers." + str(l) + ".init_h_mu.0"]
            top_bias[l] = layer.bias_d
        
        # PROPRIOCEPTION
        for l, layer in enumerate(self.prop.layers):
            prop_a[l] = self.prop.state_dict()["layers." + str(l) + ".A.0"]
            prop_init_h[l] = self.prop.state_dict()["layers." + str(l) + ".init_h.0"]
            prop_init_h_mu[l] = self.prop.state_dict()["layers." + str(l) + ".init_h_mu.0"]
            prop_bias[l] = layer.bias_d
        
        # EXTEROCEPTION
        for l, layer in enumerate(self.vision.layers):
            vision_a[l] = self.vision.state_dict()["layers." + str(l) + ".A.0"]
            vision_init_h[l] = self.vision.state_dict()["layers." + str(l) + ".init_h.0"]
            vision_init_h_mu[l] = self.vision.state_dict()["layers." + str(l) + ".init_h_mu.0"]
            vision_bias[l] = layer.bias_d

        # INTEROCEPTION
        for l, layer in enumerate(self.intero.layers):
            intero_a[l] = self.intero.state_dict()["layers." + str(l) + ".A.0"]
            intero_init_h[l] = self.intero.state_dict()["layers." + str(l) + ".init_h.0"]
            intero_init_h_mu[l] = self.intero.state_dict()["layers." + str(l) + ".init_h_mu.0"]
            intero_bias[l] = layer.bias_d

        summary_arch = {"p": ps, "v": vs, "i": ins,
                        "top_kl": top_kl_step, "prop_kl": prop_kl_step, "vision_kl": vision_kl_step, "intero_kl": intero_kl_step, 
                        "top_d": top_d, "top_mu_p": top_mu_p, "top_mu_q": top_mu_q, "top_sigma_p": top_sigma_p, "top_sigma_q": top_sigma_q, 
                        "prop_d": prop_d, "prop_mu_p": prop_mu_p, "prop_mu_q": prop_mu_q, "prop_sigma_p": prop_sigma_p, "prop_sigma_q": prop_sigma_q, 
                        "vision_d": vision_d, "vision_mu_p": vision_mu_p, "vision_mu_q": vision_mu_q, "vision_sigma_p": vision_sigma_p, "vision_sigma_q": vision_sigma_q, 
                        "intero_d": intero_d, "intero_mu_p": intero_mu_p, "intero_mu_q": intero_mu_q, "intero_sigma_p": intero_sigma_p, "intero_sigma_q": intero_sigma_q, 
                        "top_a": top_a, "prop_a": prop_a, "vision_a": vision_a, "intero_a": intero_a,  
                        "top_init_h": top_init_h, "top_init_h_mu": top_init_h_mu, 
                        "prop_init_h": prop_init_h, "prop_init_h_mu": prop_init_h_mu, 
                        "vision_init_h": vision_init_h, "vision_init_h_mu": vision_init_h_mu, 
                        "intero_init_h": intero_init_h, "intero_init_h_mu": intero_init_h_mu, 
                        "top_bias_d": top_bias, "prop_bias_d": prop_bias, "vision_bias_d": vision_bias, "intero_bias_d": intero_bias}
        
        return ps, vs, ins, top_kl, prop_kl, vision_kl, intero_kl, top_wkl, prop_wkl, vision_wkl, intero_wkl, top_wnll_init_h, prop_wnll_init_h, vision_wnll_init_h, intero_wnll_init_h, summary_arch


    # save a trained parameter
    def save_param(self, fn="para.pth"):
        para = self.state_dict()
        torch.save(para, fn)

    # load a trained parameter
    def load_param(self, fn="para.pth"):
        param = torch.load(fn)
        self.load_state_dict(param)

