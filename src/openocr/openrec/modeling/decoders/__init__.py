import torch.nn as nn
from importlib import import_module

__all__ = ['build_decoder']

class_to_module = {
    'ABINetDecoder': '.abinet_decoder',
    'ASTERDecoder': '.aster_decoder',
    'CDistNetDecoder': '.cdistnet_decoder',
    'CPPDDecoder': '.cppd_decoder',
    'RCTCDecoder': '.rctc_decoder',
    'CTCDecoder': '.ctc_decoder',
    'DANDecoder': '.dan_decoder',
    'IGTRDecoder': '.igtr_decoder',
    'LISTERDecoder': '.lister_decoder',
    'LPVDecoder': '.lpv_decoder',
    'MGPDecoder': '.mgp_decoder',
    'NRTRDecoder': '.nrtr_decoder',
    'PARSeqDecoder': '.parseq_decoder',
    'RobustScannerDecoder': '.robustscanner_decoder',
    'SARDecoder': '.sar_decoder',
    'SMTRDecoder': '.smtr_decoder',
    'SMTRDecoderNumAttn': '.smtr_decoder_nattn',
    'SRNDecoder': '.srn_decoder',
    'VisionLANDecoder': '.visionlan_decoder',
    'MATRNDecoder': '.matrn_decoder',
    'CAMDecoder': '.cam_decoder',
    'OTEDecoder': '.ote_decoder',
    'BUSDecoder': '.bus_decoder',
    'DptrParseq': '.dptr_parseq_clip_b_decoder',
    'MDiffDecoder': '.mdiff_decoder',
    'GTCDecoder': '.gtc_decoder',
    'GTCDecoderTwo': '.gtc_decoder',
}


def build_decoder(config):

    module_name = config.pop('name')

    # Check if the class is defined in current module (e.g., GTCDecoder)
    # if module_name in globals():
    #     module_class = globals()[module_name]
    # else:
    if module_name not in class_to_module:
        raise ValueError(f'Unsupported decoder: {module_name}')
    module_str = class_to_module[module_name]
    # Dynamically import the module and get the class
    module = import_module(module_str, package=__package__)
    module_class = getattr(module, module_name)

    return module_class(**config)


