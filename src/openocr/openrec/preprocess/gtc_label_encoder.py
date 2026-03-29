from . import dynamic_import


class GTCLabelEncode:
    """Convert between text-label and text-index."""

    def __init__(self,
                 gtc_label_encode,
                 max_text_length,
                 character_dict_path=None,
                 use_space_char=False,
                 **kwargs):
        self.gtc_label_encode = dynamic_import(gtc_label_encode['name'])(
            max_text_length=max_text_length,
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            **gtc_label_encode)
        self.ctc_label_encode = dynamic_import('CTCLabelEncode')(
            max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        data_ctc = self.ctc_label_encode({'label': data['label']})
        new_data = self.gtc_label_encode(data)
        if data_ctc is None or data is None:
            return None
        new_data['ctc_label'] = data_ctc['label']
        new_data['ctc_length'] = data_ctc['length']
        return new_data

    @property
    def character_str(self):
        return self.gtc_label_encode.character_str