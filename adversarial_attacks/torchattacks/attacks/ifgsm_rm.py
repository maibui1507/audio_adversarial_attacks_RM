import torch
import torch.nn as nn

from ..attack import Attack


class IFGSM_RM(Attack):
    """
    """

    def __init__(self, model, device, eps=8/255, alpha=2/255, steps=5, probability=0.5):
        super().__init__("IFGSM_RM", model, device)
        self.eps = eps
        self.steps = steps
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

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            delta_masked = random_alternate_with_zero_torch(delta, self.probability)
            adv_images = torch.clamp(images + delta_masked, min=0, max=1).detach()
            #check label of adv_images by model.predict function
            self.model.eval()
            adv_outputs = self.model(adv_images)
            adv_labels =  adv_outputs.argmax(dim=1)
            if adv_labels == labels:
                # print("IFGSM RM successfully attacked")
                break
            else:
                self.alpha = self.alpha * 1.2
        if adv_labels != labels:
            # print("IFGSM RM failed to attack")
            return None
        
        return adv_images

def random_alternate_with_zero_torch(input_array, probability=0.5):
    mask = torch.rand(input_array.shape) < probability
    output_array = input_array.clone()
    output_array[mask] = 0
    return output_array