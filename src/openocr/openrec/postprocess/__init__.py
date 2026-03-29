import copy
from importlib import import_module

__all__ = ['build_post_process']

# 定义类名到模块路径的映射
module_mapping = {
    'CTCLabelDecode': '.ctc_postprocess',
    'CharLabelDecode': '.char_postprocess',
    'CELabelDecode': '.ce_postprocess',
    'CPPDLabelDecode': '.cppd_postprocess',
    'NRTRLabelDecode': '.nrtr_postprocess',
    'ABINetLabelDecode': '.abinet_postprocess',
    'ARLabelDecode': '.ar_postprocess',
    'IGTRLabelDecode': '.igtr_postprocess',
    'VisionLANLabelDecode': '.visionlan_postprocess',
    'SMTRLabelDecode': '.smtr_postprocess',
    'SRNLabelDecode': '.srn_postprocess',
    'LISTERLabelDecode': '.lister_postprocess',
    'MPGLabelDecode': '.mgp_postprocess',
    'UniRecLabelDecode': '.unirec_postprocess',
    'CMERLabelDecode': '.cmer_postprocess',
    'GTCLabelDecode': '.gtc_postprocess' 
}


def build_post_process(config, global_config=None):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if global_config is not None:
        config.update(global_config)

    assert module_name in module_mapping, Exception(
        'post process only support {}'.format(list(module_mapping.keys())))

    module_path = module_mapping[module_name]

    # 处理当前模块中的类
    if module_path == '.':
        module_class = globals()[module_name]
    else:
        # 动态导入模块
        module = import_module(module_path, package=__package__)
        module_class = getattr(module, module_name)

    return module_class(**config)

