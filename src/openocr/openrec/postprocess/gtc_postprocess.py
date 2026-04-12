from . import build_post_process

class GTCLabelDecode(object):
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_decode=None,
                 ctc_label_decode=None,
                 character_dict_path=None,
                 use_space_char=True,
                 only_gtc=False,
                 with_ratio=False,
                 **kwargs):
        
        gtc_label_decode['character_dict_path'] = character_dict_path
        gtc_label_decode['use_space_char'] = use_space_char
        
        ctc_label_decode['character_dict_path'] = character_dict_path
        ctc_label_decode['use_space_char'] = use_space_char


        self.gtc_label_decode = build_post_process(gtc_label_decode)
        self.ctc_label_decode = build_post_process(ctc_label_decode)

        self.gtc_character = self.gtc_label_decode.character
        self.ctc_character = self.ctc_label_decode.character
        self.only_gtc = only_gtc
        self.with_ratio = with_ratio

    
    @property
    def get_character_num(self):
        return {
            "gtc_num": len(self.gtc_character),
            "ctc_num": len(self.ctc_character)
        }

    def __call__(self, preds, batch=None, *args, **kwargs):
        if self.with_ratio:
            assert isinstance(batch, list), "with_ratio 관련해서 아직 잘 몰라서 여기에 대해서는 batch를 dict로 다루도록 반영하지 않음, 아마  with_ratio가 활성화되면 레이블이 추가되도록 하는 코드가 있을텐데 그 부분을 찾아서 batch가 dict일 때도 추가될 수 있도록 코드를 수정해야 함"
            batch = batch[:-1]


        if isinstance(batch, dict):
            gtc_batch = batch["gtc_label"]
        else:
            gtc_batch = batch[:-2]



        gtc = self.gtc_label_decode(preds['gtc_pred'], gtc_batch if batch is not None else None)
        if self.only_gtc:
            return {"gtc_pred": gtc}




        if isinstance(batch, dict):
            ctc_batch = batch["ctc_label"]
        else:
            ctc_batch = [None]+batch[-2:]
        ctc = self.ctc_label_decode(preds['ctc_pred'], ctc_batch if batch is not None else None)

        return {"gtc_pred": gtc, "ctc_pred": ctc, "pred": ctc}
