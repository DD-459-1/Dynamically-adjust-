



import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.init as init
import numpy as np


class DAs_Yogi(Optimizer):

    def __init__(self, params, batch_num, lr = required, adjustment_rate = 0.4, beta_1_fin = 0.9,
                 beta_2_fin = 0.999, epsilon = 1e-8,k = 0.05, beta_1_ini = 0.3, weight_decay = 0):
        """
        Args:
            params: model parameters
            batch_num: the total number of batches
            d_model: dimension of data variables
            lr: learning rate
            adjustment_rate: proportion of data involved in dynamic adjustment
            method_flag: optimization method
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta_1_fin < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta_1_fin))
        if not 0.0 <= beta_2_fin < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta_2_fin))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr = lr, beta_1_fin = beta_1_fin, beta_2_fin = beta_2_fin, epsilon = epsilon,
                        adjustment_step = int(adjustment_rate * batch_num),batch_num = batch_num,
                        beta_1_ini = beta_1_ini ,beta_2_ini = torch.pow(torch.tensor(beta_1_ini), 0.9),
                        k = k,max_v = 0, weight_decay = weight_decay)
        super(DAs_Yogi, self).__init__(params, defaults)

    def _init_state(self,p):
        """
        Initialize optimizer state
        Args:
            p: Model parameters

        Returns:
        Initialized optimizer parameters
        """
        state = self.state[p]
        state["step"] = 0
        state['momentum'] = torch.zeros_like(p.data)
        state['velocity'] = torch.zeros_like(p.data)
        return state

    def _adjust_beta_exp(self, beta_fin, t, k, beta_ini):
        """
        Adjust the beta parameter to avoid local minimum point
        Returns:
        Adjusted beta parameter
        """
        s = self.defaults['adjustment_step']
        alpha = np.exp(-k * (s - t))
        return alpha * beta_fin + (1 - alpha) * beta_ini

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta_1_fin = group['beta_1_fin']
            beta_1_ini = group['beta_1_ini']
            beta_2_fin = group['beta_2_fin']
            beta_2_ini = group['beta_2_ini']
            lr = group['lr']
            epsilon = group['epsilon']
            adjustment_step = group['adjustment_step']
            batch_num = group['batch_num']
            k = group['k']


            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    self._init_state(p)
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                state["step"] += 1
                momentum = state['momentum']
                velocity = state['velocity']
                t = (state['step'] % batch_num) + 1
                step = 0.5 * state["step"] * np.sin(state["step"] * np.pi / 5)  + 1.5 * state["step"]

                beta_1_adj = self._adjust_beta_exp(beta_1_fin, t, k, beta_1_ini) if t <= adjustment_step else beta_1_fin
                beta_2_adj = self._adjust_beta_exp(beta_2_fin, t, k,beta_2_ini) if t <= adjustment_step else beta_2_fin

                # Update momentum and velocity with adjusted beta values
                momentum.mul_(beta_1_adj).add_(1 - beta_1_adj, grad)
                grad_squared = grad.mul(grad)
                velocity.addcmul_(
                    torch.sign(velocity - grad_squared),
                    grad_squared,
                    value=-(1 - beta_2_adj)
                )

                # Bias-corrected momentum and velocity calculations
                bias_correction1 = 1 - beta_1_fin ** step
                bias_correction2 = 1 - beta_2_fin ** step


                # if group['max_v'] < bias_correction2:
                #     group['max_v'] = bias_correction2
                # else:
                #     bias_correction2 = group['max_v']

                step_size = lr / bias_correction1
                denom = (velocity.sqrt() / np.sqrt(bias_correction2)).add_(epsilon)

                # Update parameters
                p.data.addcdiv_(-step_size, momentum, denom)

        return loss

