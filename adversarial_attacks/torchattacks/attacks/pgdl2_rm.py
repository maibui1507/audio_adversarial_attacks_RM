import torch
import torch.nn as nn

from ..attack import Attack


class PGDL2_RM(Attack):
    """
    """
    def __init__(self, model, device, eps=1.0, alpha=0.2, steps=40, random_start=True, eps_for_division=1e-10, probability=0.5):
        super().__init__("PGDL2", model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
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

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()
        batch_size = len(images)

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(adv_images.size(0), 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            self.model.train()

            outputs = self.model(adv_images)
            outputs = torch.cat([-outputs, outputs], dim=1)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            delta_masked = random_alternate_with_zero_torch(delta, self.probability)
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta_masked = delta_masked * factor.view(-1, 1)
            
            adv_images = torch.clamp(images + delta_masked, min=0, max=1).detach()
            #check label of adv_images by model.predict function
            self.model.eval()
            adv_outputs = self.model(adv_images)
            adv_labels =  adv_outputs.argmax(dim=1)
            if adv_labels == labels:
                # print("PGD L2 RM successfully attacked")
                break
            else:
                self.alpha = self.alpha * 1.2

        if adv_labels != labels:
            # print("PGD L2 RM failed to attack")
            return None
        
        return adv_images

def random_alternate_with_zero_torch(input_array, probability=0.5):
    mask = torch.rand(input_array.shape) < probability
    output_array = input_array.clone()
    output_array[mask] = 0
    return output_array