import math
import torch.optim as optim
from transformers import Adafactor
import re
from collections import defaultdict
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import AdafactorSchedule

class Classroom(Adafactor):
    def __init__(self, *args, training_variables=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.variables = training_variables
    
    @staticmethod
    def _get_lr(param_group, param_state, variables):
        # rel_step_sz = param_group["lr"]
        rel_step_sz = 0.00002
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        param_scale = param_scale * variables["distillation_loss"].item()
        return param_scale * rel_step_sz
    
    # Default Hugging Face Adafactor implementation with _get_lr modified to use the distillation loss
    # https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/optimization.py#L639
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state, self.variables) # add external variables

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss

# Default Hugging Face Adafactor Scheduler implementation modified to accomodate _get_lr changes
# https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/optimization.py#L733
class ClassroomSchedule(AdafactorSchedule):
    def get_lr(self):
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]], opt.variables)
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs

# from https://github.com/r-three/t-few
def get_optimizer(model, config, training_variables):
    """
    Construct optimizer based on config

    :param model:
    :param config:
    :return:
    """
    optim_name = config["name"]

    def param_name_to_group_name(param_name):
        if False:
            return ".".join(param_name.split(".")[:3])
            # only needed when the model has many trainable parameters, disabled in our expeirments
        else:
            return "."

    param_groups = defaultdict(lambda: {"params": []})
    trainable_param_names = set()
    for (param_name, param) in model.named_parameters():
        if re.fullmatch(config["trainable_param_names"], param_name):
            param_groups[param_name_to_group_name(param_name)]["params"].append(param)
            trainable_param_names.add(param_name)
        else:
            param.requires_grad = False

    param_groups = param_groups.values()
    if optim_name.lower() == "adam":
        optimizer = optim.Adam(param_groups, lr=config["lr"])
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(param_groups, lr=config["lr"], weight_decay=config["weight_decay"])
    elif optim_name.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config["lr"], weight_decay=config["weight_decay"], eps=1e-8)
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            scale_parameter=config["scale_parameter"],
            relative_step=False,
            warmup_init=False,
        )
    elif optim_name.lower() == "classroom":
        optimizer = Classroom(
            param_groups,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            scale_parameter=config["scale_parameter"],
            relative_step=False,
            warmup_init=False,
            training_variables=training_variables,
        )
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return optimizer

def get_scheduler(optimizer, config):
    """
    Get scheduler

    :param optimizer:
    :param config:
    :return:
    """
    scheduler_name = config["scheduler"]
    if "warmup_ratio" in config:
        num_warmup_steps = config["num_steps"] * config["warmup_ratio"]

    if scheduler_name == "polynomial_decay_with_warmup":
        return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, config["num_steps"])
    elif scheduler_name == "exponential_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    elif scheduler_name == "linear_decay_with_warmup":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, config["num_steps"])
    elif scheduler_name == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["num_steps"])
    elif scheduler_name == "adafactor":
        return AdafactorSchedule(optimizer, config["lr"])
    elif scheduler_name == "classroom":
        return ClassroomSchedule(optimizer, config["lr"])
    else:
        raise ValueError("Invalid Scheduler Name %s" % scheduler_name)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay
    from the initial lr set in the optimizer to end lr defined by `lr_end`,
    after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is
    based on the original BERT implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)
