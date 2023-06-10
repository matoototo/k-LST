from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/philschmid/knowledge-distillation-transformers-pytorch-sagemaker
class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

class ClassroomTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, training_variables=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()
        self.variables = training_variables

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_student, outputs_student = super().compute_loss(model, inputs, return_outputs=True)

        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)
        
        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))

        # Compute L1 loss instead of distillation loss
        # loss_function = nn.L1Loss(reduction="mean")
        # loss_logits = loss_function(outputs_student.logits, outputs_teacher.logits)
        
        # Save distillation loss for use in the optimizer
        self.variables["distillation_loss"] = loss_logits

        return (loss_student, outputs_student) if return_outputs else loss_student