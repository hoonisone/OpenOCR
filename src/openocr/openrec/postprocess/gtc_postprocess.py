from . import build_post_process



class GTCLabelDecode(object):
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_decode=None,
                 character_dict_path=None,
                 use_space_char=True,
                 only_gtc=False,
                 only_ctc=False,
                 with_ratio=False,
                 **kwargs):
        gtc_label_decode['character_dict_path'] = character_dict_path
        gtc_label_decode['use_space_char'] = use_space_char
        self.gtc_label_decode = build_post_process(gtc_label_decode)
        self.ctc_label_decode = build_post_process({
            'name':
            'CTCLabelDecode',
            'character_dict_path':
            character_dict_path,
            'use_space_char':
            use_space_char
        })
        self.gtc_character = self.gtc_label_decode.character
        self.ctc_character = self.ctc_label_decode.character
        self.only_gtc = only_gtc
        self.only_ctc = only_ctc
        self.with_ratio = with_ratio

    def get_character_num(self):
        return [len(self.gtc_character), len(self.ctc_character)]

    def __call__(self, preds, batch=None, *args, **kwargs):
        if isinstance(batch, dict):
            # Keep list index compatibility:
            # - batch[1] should be GT label for gtc decoder
            # - batch[-2:] should be [ctc_label, ctc_length] for ctc decoder
            batch = [
                batch.get("image", None),
                batch.get("label", None),
                batch.get("ctc_label", None),
                batch.get("ctc_length", None),
            ]
        elif self.with_ratio and isinstance(batch, (list, tuple)):
            if len(batch) > 0:
                batch = batch[:-1]

        gtc_batch = batch
        ctc_batch = None
        if batch is not None and isinstance(batch, (list, tuple)) and len(batch) >= 2:
            gtc_batch = batch[:-2]
            ctc_batch = [None] + list(batch[-2:])

    
        if self.only_ctc:
            ctc = self.ctc_label_decode(preds['ctc_pred'], ctc_batch if batch is not None else None)
            return ctc

        if self.only_gtc:
            gtc = self.gtc_label_decode(preds['gtc_pred'], gtc_batch if batch is not None else None)
            return gtc

        gtc = self.gtc_label_decode(preds['gtc_pred'], gtc_batch if batch is not None else None)

        ctc = self.ctc_label_decode(preds['ctc_pred'], ctc_batch if batch is not None else None)

        return [gtc, ctc]