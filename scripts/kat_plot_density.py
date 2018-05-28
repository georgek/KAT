#!/usr/bin/env python3

import sys
import argparse
import logging
import functools

import numpy as np
import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

import kat_plot as k
import kat_plot_colormaps as cmaps

from PyQt5 import QtWidgets
import matplotlib.backends.backend_qt5agg


class KatDensityWindow(k.KatPlotWindow):
    def __init__(self, matrix, args, make_figure_fun):
        super().__init__(matrix, args, make_figure_fun)

        def kat_density_plot_menus():
            options_menu = self.menuBar().addMenu("&Options")
            colour_menu = options_menu.addMenu("Colour map")
            colour_group = QtWidgets.QActionGroup(self, exclusive=True)
            for cmap in cmaps.__all__:
                a = colour_group.addAction(QtWidgets.QAction(
                    cmap.capitalize(),
                    self,
                    checkable=True))
                colour_menu.addAction(a)
                a.triggered.connect(functools.partial(self.set_cmap, cmap))
                a.triggered.connect(self.redraw)
                if cmap == args.cmap:
                    a.setChecked(True)
            contour_menu = options_menu.addMenu("Contours")
            contour_group = QtWidgets.QActionGroup(self, exclusive=True)
            for contour_option in k.CONTOUR_OPTIONS:
                a = contour_group.addAction(QtWidgets.QAction(
                    contour_option.capitalize(),
                    self, checkable=True))
                contour_menu.addAction(a)
                a.triggered.connect(functools.partial(self.set_contour_option,
                                                      contour_option))
                a.triggered.connect(self.redraw)
                if contour_option == args.contours:
                    a.setChecked(True)
        self.make_menus(kat_density_plot_menus)

        self.make_axis_dock()
        self.add_axis_slider("x", self.set_x_max,
                             self.args.x_max,
                             self.matrix.shape[1],
                             0)
        self.add_axis_slider("y", self.set_y_max,
                             self.args.y_max,
                             self.matrix.shape[0],
                             1)
        self.add_axis_slider("z", self.set_z_max,
                             self.args.z_max,
                             int(max((np.percentile(self.matrix, 99)+1)*5,
                                     self.args.z_max*2)),
                             2)

        self.make_labels_dock()
        self.add_labels_input("Title", self.set_title,   self.args.title,   0)
        self.add_labels_input("X",     self.set_x_label, self.args.x_label, 1)
        self.add_labels_input("Y",     self.set_y_label, self.args.y_label, 2)
        self.add_labels_input("Z",     self.set_z_label, self.args.z_label, 3)

        self.make_output_dock()

    def set_x_label(self, v):
        self.args.x_label = str(v)

    def set_y_label(self, v):
        self.args.y_label = str(v)

    def set_z_label(self, v):
        self.args.z_label = str(v)

    @k.only_type(int)
    def set_x_max(self, v):
        self.args.x_max = v

    @k.only_type(int)
    def set_y_max(self, v):
        self.args.y_max = v

    @k.only_type(int)
    def set_z_max(self, v):
        self.args.z_max = v

    def set_cmap(self, cmap):
        if cmap in cmaps.__all__:
            self.args.cmap = cmap

    def set_contour_option(self, contour_option):
        if contour_option in k.CONTOUR_OPTIONS:
            self.args.contours = contour_option


def get_args():
    (parent_parser,
     input_group,
     output_group,
     plot_group,
     misc_group) = k.get_common_argparser()

    input_group.add_argument("matrix_file", type=str,
                             help="The input matrix file from KAT")
    plot_group.add_argument("-a", "--x_label", type=str,
                            help="Label for x-axis")
    plot_group.add_argument("-b", "--y_label", type=str,
                            help="Label for y-axis")
    plot_group.add_argument("-c", "--z_label", type=str,
                            help="Label for z-axis")
    plot_group.add_argument("-x", "--x_max", type=int,
                            help="Maximum value for x-axis")
    plot_group.add_argument("-y", "--y_max", type=int,
                            help="Maximum value for y-axis")
    plot_group.add_argument("-z", "--z_max", type=int,
                            help="Maximum value for z-axis")
    plot_group.add_argument("--contours", choices=k.CONTOUR_OPTIONS,
                            default="normal")
    plot_group.add_argument("--cmap", choices=cmaps.__all__,
                            default="viridis",
                            help="Colour map (theme)")
    misc_group.add_argument("--not_rasterised", dest="rasterised",
                            action="store_false",
                            help="Don't rasterise graphics (slow).")
    misc_group.set_defaults(rasterised=True)

    parser = argparse.ArgumentParser(
        add_help=False,
        parents=[parent_parser],
        description="""Create K-mer Density Plots.

    Creates a scatter plot, where the density or "heat" at each point
    represents the number of distinct K-mers at that point.  Typically this is
    used to visualise a matrix produced by the "kat comp" tool to compare
    multiplicities from two K-mer hashes produced by different NGS reads, or
    to visualise the GC vs K-mer multiplicity matrices produced by the "kat
    gcp" tool.""")

    return parser.parse_args()


def find_peaks(matrix):
    msum = np.sum(matrix)
    xsums = np.sum(matrix, 0)
    ysums = np.sum(matrix, 1)
    peakx = k.find_peaks(xsums)
    peaky = k.find_peaks(ysums)
    # ignore peaks at 1
    peakx = peakx[peakx != 1]
    peaky = peaky[peaky != 1]
    peakz = matrix[peaky, :][:, peakx]

    xmax = len(xsums)
    ymax = len(ysums)
    for i in range(1, len(xsums), int(len(xsums)/40) + 1):
        if np.sum(xsums[:i]) >= msum * 0.995:
            xmax = i
            break
    for i in range(1, len(ysums), int(len(ysums)/40) + 1):
        if np.sum(ysums[:i]) >= msum * 0.995:
            ymax = i
            break

    zmax = int(np.max(peakz) * 1.1)

    return xmax, ymax, zmax


def make_figure(figure, matrix, args):
    if args.contours == "smooth":
        matrix_smooth = ndimage.gaussian_filter(matrix, sigma=2.0, order=0)

    ax = figure.add_subplot(111)
    pcol = ax.pcolormesh(matrix, vmin=0, vmax=args.z_max,
                         cmap=cmaps.cmaps[args.cmap],
                         rasterized=args.rasterised)
    ax.axis([0, args.x_max, 0, args.y_max])
    cbar = figure.colorbar(pcol, label=k.wrap(args.z_label))
    cbar.solids.set_rasterized(args.rasterised)
    if args.z_max > 0:
        levels = np.arange(args.z_max/8, args.z_max, args.z_max/8)
        if args.contours == "normal":
            ax.contour(matrix, colors="white",
                       alpha=0.6, levels=levels)
        elif args.contours == "smooth":
            ax.contour(matrix_smooth, colors="white",
                       alpha=0.6, levels=levels)

    ax.set_title(k.wrap(args.title))
    ax.set_xlabel(k.wrap(args.x_label))
    ax.set_ylabel(k.wrap(args.y_label))
    ax.grid(True, color="white", alpha=0.2, linestyle="dashed")


def main(args):
    try:
        with open(args.matrix_file) as input_file:
            header = k.readheader(input_file)
            matrix = np.loadtxt(input_file)
    except FileNotFoundError as e:
        sys.exit(e)

    if "Transpose" in header and header["Transpose"] == '1':
        matrix = np.transpose(matrix)

    logging.info("%d by %d matrix file loaded.",
                 matrix.shape[0], matrix.shape[1])

    if args.x_max is None or args.y_max is None or args.z_max is None:
        xmax, ymax, zmax = find_peaks(matrix)
        logging.info("Automatically detected axis limits:")
        logging.info("xmax: %d", xmax)
        logging.info("ymax: %d", ymax)
        logging.info("zmax: %d", zmax)

    if args.x_max is None:
        args.x_max = xmax
    if args.y_max is None:
        args.y_max = ymax
    if args.z_max is None:
        args.z_max = zmax

    if args.title is None and "Title" in header:
        args.title = header["Title"]
    else:
        args.title = "Density Plot"

    if args.x_label is None and "XLabel" in header:
        args.x_label = header["XLabel"]
    else:
        args.x_label = "X"

    if args.y_label is None and "YLabel" in header:
        args.y_label = header["YLabel"]
    else:
        args.y_label = "Y"

    if args.z_label is None and "ZLabel" in header:
        args.z_label = header["ZLabel"]
    else:
        args.z_label = "Z"

    if args.output:
        figure = k.new_figure(args.width, args.height)
        if args.output_type is not None:
            output_name = args.output + '.' + args.output_type
        else:
            output_name = args.output
        make_figure(figure, matrix, args)
        figure.savefig(k.correct_filename(output_name), dpi=args.dpi)
    else:
        app = QtWidgets.QApplication(sys.argv)
        main = KatDensityWindow(matrix, args, make_figure)
        main.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)
    main(args)
