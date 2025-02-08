




import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn.init as init
import numpy as np


class Beta_iterator:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.length = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.length:
            current_data = self.data[self.index]
            self.index += 1
            # 检查当前元素是否是最后一个
            is_last = self.index == self.length
            return current_data, is_last
        else:
            raise StopIteration

class DA_Yogi(Optimizer):

    def __init__(self, params, batch_num, lr = required, adjustment_rate = 0.3, beta_1 = 0.9, beta_2 = 0.999,
                 epsilon = 1e-8, beta_1_ini = 0.3, weight_decay = 0):
        """
                Args:
                    params: model parameters
                    batch_num: the total number of batches
                    lr: learning rate
                    adjustment_rate: proportion of data involved in dynamic adjustment
                """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta_1 < 1.0:
            raise ValueError("Invalid beta1 parameter: {}".format(beta_1))
        if not 0.0 <= beta_2 < 1.0:
            raise ValueError("Invalid beta2 parameter: {}".format(beta_2))
        if not 0.0 <= beta_1_ini <= beta_1:
            raise ValueError("Invalid beta1 initial value: {}".format(beta_1_ini))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        adjustment_step = int(adjustment_rate * batch_num)
        defaults = {
            'batch_num': batch_num,
            'lr': lr,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'beta_1_ini': beta_1_ini,
            'beta_2_ini': torch.pow(torch.tensor(beta_1_ini),0.9),
            'adjustment_step': adjustment_step,
            'single_adj_step': 4,
            'epsilon': epsilon,
            'max_v': 0,
            'weight_decay': weight_decay,
            'probability': 0,
            'trigger_count': 0,
            'adj_enable': True,
            'beta_1_iter': None,
            'beta_2_iter': None
        }
        super(DA_Yogi, self).__init__(params, defaults)


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


    def _adjust_beta_iter(self, beta, beta_ini):
        """
                Adjust the beta parameter to avoid local minimum point
        """
        sin_adj_step = self.defaults['single_adj_step']
        adj_list = []
        for i in range(sin_adj_step):
            alpha = np.exp(i - sin_adj_step)
            adj_list.append(alpha * beta + (1 - alpha) * beta_ini)

        return Beta_iterator(adj_list)


    def _calculate_gradient_angle(self, grad, m):
        """
            Calculate the angle between the current gradient and the historical accumulated gradient for tensors of any shape.

            Args:
            grad (torch.Tensor): Current gradient tensor of any shape.
            m (torch.Tensor): Historical accumulated gradient tensor of any shape.

            Returns:
            float: Angle in degrees between the two gradients.
            """
        # Ensure both gradients are on the same device
        if grad.device != m.device:
            m = m.to(grad.device)

        # Flatten the tensors to 1D
        grad_flat = grad.flatten()
        m_flat = m.flatten()

        # Calculate the dot product of flattened tensors
        dot_product = torch.dot(grad_flat, m_flat)

        # Calculate norms of the flattened tensors
        norm_grad = torch.norm(grad_flat)
        norm_m = torch.norm(m_flat)

        # Calculate cosine of the angle
        cos_theta = dot_product / (norm_grad * norm_m)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp to prevent numerical issues

        # Calculate angle in radians and then convert to degrees
        angle_radians = torch.acos(cos_theta)
        angle_degrees = angle_radians * 180 / torch.pi

        return angle_degrees.item()

    def _generate_probability(self, grad, m, k = 50):
        angle = self._calculate_gradient_angle(grad, m)
        x = angle / 180
        return 1 / (1 + torch.exp(torch.tensor(-k * (x - 0.5))))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            beta_1 = group['beta_1']
            beta_2 = group['beta_2']
            beta_1_ini = group['beta_1_ini']
            beta_2_ini = group['beta_2_ini']
            lr = group['lr']
            epsilon = group['epsilon']
            adjustment_step = group['adjustment_step']
            single_adj_step = group['single_adj_step']

            # DA optrate
            if group['trigger_count'] < adjustment_step:
                if group['adj_enable']:
                    seed = torch.rand(1)
                    if seed < group['probability']:
                        group['adj_enable'] = True
                        group['beta_1_iter'] = self._adjust_beta_iter(beta_1, beta_1_ini)
                        group['beta_2_iter'] = self._adjust_beta_iter(beta_2, beta_2_ini)
                        beta_1, is_last = next(group['beta_1_iter'])
                        beta_2, is_last = next(group['beta_2_iter'])
                else:
                    beta_1, is_last = next(group['beta_1_iter'])
                    beta_2, is_last = next(group['beta_2_iter'])
                    if is_last:
                        group['adj_enable'] = False
                        group['trigger_count'] += single_adj_step

            next_probability = 0
            divid = 1e-8
            for i, p in enumerate(group['params']):
                 divid = i + 1
                 # t ← t + 1
                 state = self.state[p]
                 if len(state) == 0:
                     self._init_state(p)
                 state['step'] += 1
                 # g
                 if p.grad is None:
                     continue
                 grad = p.grad.data
                 if group['weight_decay'] != 0:
                     grad = grad.add(p.data, alpha=group['weight_decay'])
                 # last mt, vt
                 momentum = state['momentum']
                 velocity = state['velocity']

                 if group['trigger_count'] < adjustment_step:
                    next_probability += self._generate_probability(grad, momentum)

                 # Update momentum and velocity with adjusted beta values
                 momentum.mul_(beta_1).add_(1 - beta_1, grad)
                 grad_squared = grad.mul(grad)
                 velocity.addcmul_(
                     torch.sign(velocity - grad_squared),
                     grad_squared,
                     value = -(1 - beta_2)
                 )
                 # Bias-corrected momentum and velocity calculations
                 bias_correction1 = 1 - beta_1 ** state['step']
                 bias_correction2 = 1 - beta_2 ** state['step']
                 step_size = lr / bias_correction1
                 denom = (velocity.sqrt() / np.sqrt(bias_correction2)).add_(epsilon)
                 # Update parameters
                 p.data.addcdiv_(-step_size, momentum, denom)

            next_probability /= divid
            group['probability'] = next_probability
        return loss


