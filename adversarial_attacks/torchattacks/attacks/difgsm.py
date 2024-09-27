import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import Attack


class DIFGSM(Attack):
    r"""
    DI2-FGSM in the paper 'Improving Transferability of Adversarial Examples with Input Diversity'
    [https://arxiv.org/abs/1803.06978]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 0.0)
        steps (int): number of iterations. (Default: 20)
        resize_rate (float): resize factor used in input diversity. (Default: 0.9)
        diversity_prob (float) : the probability of applying input diversity. (Default: 0.5)
        random_start (bool): using random initialization of delta. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=20, decay=0.0, resize_rate=0.9, diversity_prob=0.5, random_start=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device, eps=8/255, alpha=2/255, steps=20, decay=0.0,
                 resize_rate=0.9, diversity_prob=0.5, random_start=False):
        super().__init__("DIFGSM", model, device)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    # def input_diversity(self, x):
    #     img_size = x.shape[-1]
    #     img_resize = int(img_size * self.resize_rate)

    #     if self.resize_rate < 1:
    #         img_size = img_resize
    #         img_resize = x.shape[-1]

    #     rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    #     rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    #     h_rem = img_resize - rnd
    #     w_rem = img_resize - rnd
    #     pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    #     pad_bottom = h_rem - pad_top
    #     pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    #     pad_right = w_rem - pad_left

    #     padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    #     return padded if torch.rand(1) < self.diversity_prob else x

    def input_diversity(self, x):
        audio_length = x.shape[-1]
        audio_resize = int(audio_length * self.resize_rate)

        if self.resize_rate < 1:
            audio_length = audio_resize
            audio_resize = x.shape[-1]

        rnd = torch.randint(low=audio_length, high=audio_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x.unsqueeze(1), size=[rnd], mode='linear', align_corners=False).squeeze(1)
        rem = audio_resize - rnd
        pad_left = torch.randint(low=0, high=rem.item(), size=(1,), dtype=torch.int32)
        pad_right = rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item()], value=0)

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            self.model.train()

            outputs = self.model(self.input_diversity(adv_images))

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
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            #check label of adv_images by model.predict function
            self.model.eval()
            adv_outputs = self.model(adv_images)
            adv_labels =  adv_outputs.argmax(dim=1)
            if adv_labels == labels:
                # print("DIFGSM successfully attacked")
                break
            else:
                self.alpha = self.alpha * 1.2
        if adv_labels != labels:
            # print("DIFGSM failed to attack")
            return None
        return adv_images
