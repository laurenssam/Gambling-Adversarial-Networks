from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # evaluation
        parser.add_argument('--evaluate', type=int, default=1, help="whether to evaluate")
        parser.add_argument('--model_eval', type=str, default='seg', help='')
        parser.add_argument('--name_eval', type=str, default='facades_segmentation', help='train, val, test, etc')
        parser.add_argument('--suffix2', type=str, default='', help='suffix for saving directory')
        parser.add_argument('--class_id', type=str, default='train', help='class to be evaluated')
        parser.add_argument('--test', type=str, default="test", help='for checking mode')
        parser.add_argument('--gambler', type=int, default=0, help='for checking mode')
        parser.add_argument('--structure', type=int, default=0, help='for checking mode')
        parser.add_argument('--argmax', type=int, default=0, help='for checking mode')
        parser.add_argument('--visualize_features', type=int, default=0, help='for checking mode')


        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
