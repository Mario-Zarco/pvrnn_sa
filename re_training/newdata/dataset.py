"""
dataset class to train interoceptive decoupling
"""
import torch

class DataSet():

    def __init__(self, motor, vision, intero, minibatch_size: int):
        """motor: npy file with the form of(n_seq, time-step, data)
           vision: npy file with the form of (n_seq, time-step, data)
           intero; npy fule with the form of (n_seq, time-step, data)
           minibatch_size: should hold n_seq%minibatch_size=0"""

        self.minibatch_size = minibatch_size
        self.n_seq, self.seq_len, self.motor_dim = motor.shape
        self.vision_dim = vision.shape[-1]
        self.intero_dim = intero.shape[-1]

        if self.n_seq % minibatch_size != 0:
            print("Choose minibatch_size s.t. n_seq % minibatch_size = 0")

        self.n_minibatch = int(self.n_seq / minibatch_size)
        print("#seq: {}, minibatch_size: {}, #minibatch: {}".format(self.n_seq, minibatch_size, self.n_minibatch))

        motor_minibatch, vision_minibatch, intero_minibatch = [], [], []
        for i in range(self.n_minibatch):
            motor_minibatch.append(torch.from_numpy(motor[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor))
            #vision_minibatch.append(torch.from_numpy(vision[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor).to("cuda"))
            vision_minibatch.append(torch.from_numpy(vision[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor))
            intero_minibatch.append(torch.from_numpy(intero[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor))
    
        self.motor_minibatch = motor_minibatch
        self.vision_minibatch = vision_minibatch
        self.intero_minibatch = intero_minibatch