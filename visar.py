#!/usr/bin/env python
'''
SAR visualizer.
'''

import os
import sys

from sar import parser
from sar import viz


def main(log_type, in_sar_log, output_path):
    insar = parser.Parser(in_sar_log)
    sar_viz = viz.Visualization(insar.get_sar_info(), **{log_type: True})
    sar_viz.save(output_path, output_type=viz.Visualization.PDF_OUTPUT)


def set_include_path():
    include_path = os.path.abspath("./")
    sys.path.append(include_path)

if __name__ == "__main__":
    set_include_path()

    if sys.argv[1] not in viz.Visualization.SAR_TYPES:
        raise Exception('Type \'{}\' is not one of {}'
                        .format(sys.argv[1], viz.Visualization.SAR_TYPES))

    if not os.path.isfile(sys.argv[2]):
        raise Exception('Cannot find SAR log file {}'.format(sys.argv[2]))

    main(sys.argv[1], sys.argv[2], sys.argv[3])
