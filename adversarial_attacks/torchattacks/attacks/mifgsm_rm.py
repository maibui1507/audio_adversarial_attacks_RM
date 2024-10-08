import torch
import torch.nn as nn
import numpy as np

from ..attack import Attack


class MIFGSM_RM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device, eps=8/255, alpha=2/255, steps=5, decay=1.0, probability=0.5):
        super().__init__("MIFGSM_RM", model, device)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.probability = probability
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            self.model.train()

            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            delta_masked = random_alternate_with_zero_torch(delta, self.probability)
            adv_images = torch.clamp(images + delta_masked, min=0, max=1).detach()
            #check label of adv_images by model.predict function
            self.model.eval()
            adv_outputs = self.model(adv_images)
            adv_labels =  adv_outputs.argmax(dim=1)
            if adv_labels == labels:
                # print("MIFGSM successfully attacked")
                break
            else:
                self.alpha = self.alpha * 1.2

        if adv_labels != labels:
            # print("MIFGSM RM failed to attack")
            return None
        return adv_images

def random_alternate_with_zero(input_array, probability=0.5):
    mask = np.random.rand(*input_array.shape) < probability
    output_array = input_array.copy()
    output_array[mask] = 0
    return output_array

def random_alternate_with_zero_torch(input_array, probability=0.5):
    mask = torch.rand(input_array.shape) < probability
    output_array = input_array.clone()
    output_array[mask] = 0
    return output_array