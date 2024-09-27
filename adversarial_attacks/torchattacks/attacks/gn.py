import torch

from ..attack import Attack


class GN(Attack):
    r"""
    Add Gaussian Noise.

    Arguments:
        model (nn.Module): model to attack.
        std (nn.Module): standard deviation (Default: 0.1).

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.GN(model)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, std=0.1, eps=0.3):
        super().__init__("GN", model)
        self.std = std
        self.eps = eps
        self._supported_mode = ['default']

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        adv_images = images + self.std*torch.randn_like(images)
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(delta + adv_images, min=0, max=1).detach()
        # adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        #check label of adv_images by model.predict function
        batch_size = images.shape[0]
        labels = torch.tensor([1] * batch_size).to(self.device)
        self.model.eval()
        adv_outputs = self.model(adv_images)
        adv_labels =  adv_outputs.argmax(dim=1)
        if adv_labels == labels:
            print("GN successfully attacked")
        else:
            print("GN failed to attack")
            return None
        
        return adv_images
