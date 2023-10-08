import argparse


class SwtArgParser(argparse.ArgumentParser):
    """Argument parser for running orientation detection model."""

    def __init__(self):
        super().__init__()

        self.add_argument('--image', default=r"images\0459_60.png", 
                          help='The image file to process.')

        self.add_argument('--bright_on_dark', default=False, action='store_true',
                          help='Enables bright on dark selection.')

        self.add_argument('--filter_percent', default=75, 
                          help='Filter out the lines with the shortest length 75/points with the smallest value 75')
        # 值越大，过滤掉的线越多，如75表示仅留下最亮的25%的线条和点，但同时所得的mask区域较小

        self.add_argument('--dilate_kernel', default=5,
                          help='The size of the kernel used to dilate the mask.')


        #self.add_argument(
        #    "--validation_dir", "-vd", default="/tmp",
        #    help="[default: %(default)s] The location of the validation data.",
        #    metavar="<VD>",
        #)

        #self.set_defaults(
        #    validation_dir=os.path.join('dataset', 'eval')
        #    )
