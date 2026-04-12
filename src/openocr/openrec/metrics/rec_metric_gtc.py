from .rec_metric import RecMetric

from . import build_metric

class RecGTCMetric(object):

    def __init__(self,
                ctc_metric, 
                gtc_metric, 
                use_ctc:bool = True,
                use_gtc:bool = True,
                main_indicator='acc',
                is_filter=False,
                ignore_space=True,
                stream=False,
                with_ratio=False,
                max_len=25,
                max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5


        ctc_metric["main_indicator"] = main_indicator
        ctc_metric["is_filter"] = is_filter
        ctc_metric["ignore_space"] = ignore_space
        ctc_metric["stream"] = stream
        ctc_metric["with_ratio"] = with_ratio
        ctc_metric["max_len"] = max_len
        ctc_metric["max_ratio"] = max_ratio
        
        
        gtc_metric["main_indicator"] = main_indicator
        gtc_metric["is_filter"] = is_filter
        gtc_metric["ignore_space"] = ignore_space
        gtc_metric["stream"] = stream
        gtc_metric["with_ratio"] = with_ratio
        gtc_metric["max_len"] = max_len
        gtc_metric["max_ratio"] = max_ratio


        self.gtc_metric = build_metric(gtc_metric)
        self.ctc_metric = build_metric(ctc_metric)

    def __call__(self,
                 pred_label,
                 batch=None,
                 training=False,
                 *args,
                 **kwargs):

        # metric = {}
        ctc_metric = self.ctc_metric(pred_label["ctc_pred"], batch["ctc_label"], training=training)
        gtc_metric = self.gtc_metric(pred_label["gtc_pred"], batch["gtc_label"], training=training)

        return {'ctc_metric': ctc_metric, 'gtc_metric': gtc_metric}
        # metric['ctc_acc'] = ctc_metric['acc']
        # metric['ctc_ned'] = ctc_metric['norm_edit_dis']

        # metric['gtc_acc'] = gtc_metric['acc']
        # metric['gtc_ned'] = gtc_metric['norm_edit_dis']
        # return ctc_metric

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        ctc_metric = self.ctc_metric.get_metric()
        gtc_metric = self.gtc_metric.get_metric()
        # ctc_metric['gtc_acc'] = gtc_metric['acc']
        # ctc_metric['gtc_norm_edit_dis'] = gtc_metric['norm_edit_dis']
        return {'ctc_metric': ctc_metric, 'gtc_metric': gtc_metric}
        # return ctc_metric
