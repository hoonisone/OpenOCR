from torch import nn

from . import build_loss


class GTCLoss(nn.Module):

    def __init__(
        self,
        gtc_loss,
        gtc_weight=1.0,
        ctc_weight=1.0,
        zero_infinity=True,
        **kwargs,
    ):
        super(GTCLoss, self).__init__()
        ctc_config = {'name': 'CTCLoss', 'zero_infinity': zero_infinity}
        self.ctc_loss = build_loss(ctc_config)
        self.gtc_loss = build_loss(gtc_loss)
        self.gtc_weight = gtc_weight
        self.ctc_weight = ctc_weight

    def forward(self, predicts, batch):
        if isinstance(batch, dict):
            # Keep list index compatibility with existing loss implementations:
            # - batch[:-2] includes image/label for gtc loss
            # - batch[-2:] is [ctc_label, ctc_length] for ctc loss
            batch = [
                batch.get('image', None),
                batch.get('label', None),
                batch.get('ctc_label', None),
                batch.get('ctc_length', None),
            ]

        ctc_loss = self.ctc_loss(predicts['ctc_pred'],
                                 [None] + batch[-2:])['loss']
        gtc_loss = self.gtc_loss(predicts['gtc_pred'], batch[:-2])['loss']
        return {
            'loss': self.ctc_weight * ctc_loss + self.gtc_weight * gtc_loss,
            'ctc_loss': ctc_loss,
            'gtc_loss': gtc_loss
        }
