import copy
from importlib import import_module
from torch import nn

name_to_module = {
    'ABINetLoss': '.abinet_loss',
    'ARLoss': '.ar_loss',
    'CDistNetLoss': '.cdistnet_loss',
    'CELoss': '.ce_loss',
    'CPPDLoss': '.cppd_loss',
    'CTCLoss': '.ctc_loss',
    'IGTRLoss': '.igtr_loss',
    'LISTERLoss': '.lister_loss',
    'LPVLoss': '.lpv_loss',
    'MGPLoss': '.mgp_loss',
    'PARSeqLoss': '.parseq_loss',
    'RobustScannerLoss': '.robustscanner_loss',
    'SEEDLoss': '.seed_loss',
    'SMTRLoss': '.smtr_loss',
    'SRNLoss': '.srn_loss',
    'VisionLANLoss': '.visionlan_loss',
    'CAMLoss': '.cam_loss',
    'MDiffLoss': '.mdiff_loss',
    'UniRecLoss': '.unirec_loss',
    'CMERLoss': '.cmer_loss',
    'GTCLoss': '.gtc_loss',
}


def build_loss(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')

    assert module_name in name_to_module, Exception(
        '{} is not supported. The losses in {} are supportes'.format(
            module_name, list(name_to_module.keys())))
    module_path = name_to_module[module_name]
    module = import_module(module_path, package=__package__)
    module_class = getattr(module, module_name)

    return module_class(**config)


