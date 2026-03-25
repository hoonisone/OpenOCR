from openocr.tools.engine.config import Config
from openocr.tools.engine.trainer import Trainer
from openocr.tools.utility import ArgsParser


def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        '--eval',
        action='store_true',
        default=True,
        help='Whether to perform evaluation in train',
    )
    args = parser.parse_args()
    return args


def main():
    FLAGS = parse_args()
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt')
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)
    trainer = Trainer(cfg,
                      mode='train_eval' if FLAGS['eval'] else 'train',
                      task='det')
    trainer.train()


if __name__ == '__main__':
    main()
