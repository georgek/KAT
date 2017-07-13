#!/usr/bin/env python3

import sys
import argparse
import threading
import logging
import functools

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.backends.backend_qt4agg

import kat_plot_colormaps as cmaps
from kat_plot_misc import *

from PyQt4 import QtCore, QtGui


def onlyInt(fun):
    """Only runs the function if the last argument can be coerced into an int,
otherwise does nothing.
    """
    @functools.wraps(fun)
    def wrapper(*args):
        try:
            v = int(args[-1])
        except ValueError:
            return
        fun(*args[:-1], v)
    return wrapper


class MainWindow(QtGui.QMainWindow):
    def __init__(self, figure, matrix, header, args):
        super().__init__()
        self.figure = figure
        self.matrix = matrix
        self.header = header
        self.args = args

        self.setWindowTitle("KAT plot")
        self.setWindowIcon(QtGui.QIcon("kat_logo.png"))

        self.canvas = matplotlib.backends.backend_qt4agg.FigureCanvasQTAgg(self.figure)
        self.setCentralWidget(self.canvas)

        self.redraw_event = threading.Event()
        self.end_event = threading.Event()
        self.drawthread = threading.Thread(target=self.async_redraw,
                                           args=(self.redraw_event, self.end_event))

        axisdock = QtGui.QDockWidget("Axis limits")
        axisdock.setAutoFillBackground(True)
        palette = axisdock.palette()
        palette.setColor(axisdock.backgroundRole(), QtCore.Qt.white)
        axisdock.setPalette(palette)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, axisdock)

        sliders = QtGui.QWidget()
        # sliders.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum,
        #                                         QtGui.QSizePolicy.Expanding))
        sliders_grid = QtGui.QGridLayout(sliders)

        def add_slider(lab, fun, init, maximum, col):
            logging.debug("add_slider: %s", locals())
            label = QtGui.QLabel(lab, sliders)
            label.setAlignment(QtCore.Qt.AlignCenter)
            textbox = QtGui.QLineEdit(sliders)
            textbox.setText(str(init))
            # let the user type a higher number as this will be corrected by
            # the slider
            textbox.setValidator(QtGui.QIntValidator(1, maximum*10))
            sld = QtGui.QSlider(QtCore.Qt.Vertical, sliders)
            sld.setMinimum(1)
            sld.setMaximum(maximum)
            sld.setSliderPosition(init)
            sld.setTickInterval(maximum/10)
            sld.setTickPosition(QtGui.QSlider.TicksRight)
            sld.setFocusPolicy(QtCore.Qt.NoFocus)
            textbox.textChanged[str].connect(functools.partial(self.updateSlider, sld))
            sld.valueChanged[int].connect(fun)
            sld.valueChanged[int].connect(functools.partial(self.updateTextBox, textbox))
            sld.valueChanged.connect(self.redraw)
            sliders_grid.addWidget(label,   0, col)
            sliders_grid.addWidget(textbox, 1, col)
            sliders_grid.addWidget(sld,     2, col)

        add_slider("x", self.setXmax,
                   args.x_max,
                   matrix.shape[1],
                   0)
        add_slider("y", self.setYmax,
                   args.y_max,
                   matrix.shape[0],
                   1)
        add_slider("z", self.setZmax,
                   args.z_max,
                   int(max((np.percentile(matrix, 99)+1)*5, args.z_max*2)),
                   2)

        axisdock.setWidget(sliders)

        make_figure(self.figure, self.matrix, self.header, self.args)
        self.drawthread.start()

    def closeEvent(self, event):
        self.end_event.set()
        self.redraw()
        super().closeEvent(event)

    def redraw(self):
        self.redraw_event.set()


    def async_redraw(self, redraw_event, end_event):
        while True:
            redraw_event.wait()
            redraw_event.clear()
            if end_event.isSet(): return
            self.figure.clear()
            make_figure(self.figure, self.matrix, self.header, self.args)
            self.canvas.draw()


    def updateTextBox(self,textBox,val):
        textBox.setText(str(val))


    @onlyInt
    def updateSlider(self,slider,val):
        slider.setSliderPosition(val)


    @onlyInt
    def setXmax(self,v):
        self.args.x_max = v


    @onlyInt
    def setYmax(self,v):
        self.args.y_max = v


    @onlyInt
    def setZmax(self,v):
        self.args.z_max = v


def get_args():
    parser = argparse.ArgumentParser(
        description="""Create K-mer Density Plots.

    Creates a scatter plot, where the density or "heat" at each point
    represents the number of distinct K-mers at that point.  Typically this is
    used to visualise a matrix produced by the "kat comp" tool to compare
    multiplicities from two K-mer hashes produced by different NGS reads, or
    to visualise the GC vs K-mer multiplicity matrices produced by the "kat
    gcp" tool.""")

    parser.add_argument("matrix_file", type=str,
                        help="The input matrix file from KAT")

    parser.add_argument("-o", "--output", type=str,
                        help="The path to the output file.")
    parser.add_argument("-p", "--output_type", type=str,
                        help="The plot file type to create (default is " \
                        "based on given output name).")
    parser.add_argument("-t", "--title", type=str,
                        help="Title for plot")
    parser.add_argument("-a", "--x_label", type=str,
                        help="Label for x-axis")
    parser.add_argument("-b", "--y_label", type=str,
                        help="Label for y-axis")
    parser.add_argument("-c", "--z_label", type=str,
                        help="Label for z-axis")
    parser.add_argument("-x", "--x_max", type=int,
                        help="Maximum value for x-axis")
    parser.add_argument("-y", "--y_max", type=int,
                        help="Maximum value for y-axis")
    parser.add_argument("-z", "--z_max", type=int,
                        help="Maximum value for z-axis")
    parser.add_argument("-w", "--width", type=int, default=8,
                        help="Width of canvas")
    parser.add_argument("-l", "--height", type=int, default=6,
                        help="Height of canvas")
    parser.add_argument("--contours", choices=["none", "normal", "smooth"],
                        default="normal")
    parser.add_argument("--not_rasterised", dest="rasterised",
                        action="store_false",
                        help="Don't rasterise graphics (slower).")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolution in dots per inch of output graphic.")
    parser.set_defaults(rasterised=True)
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action="store_true",
                        help="Print extra information")
    parser.set_defaults(verbose=False)

    return parser.parse_args()


def find_peaks(matrix):
    msum = np.sum(matrix)
    xsums = np.sum(matrix, 0)
    ysums = np.sum(matrix, 1)
    peakx = findpeaks(xsums)
    peaky = findpeaks(ysums)
    # ignore peaks at 1
    peakx = peakx[peakx != 1]
    peaky = peaky[peaky != 1]
    peakz = matrix[peaky,:][:,peakx]

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

    return xmax,ymax,zmax


def make_figure(figure, matrix, header, args):
    if args.title is not None:
        title = args.title
    elif "Title" in header:
        title = header["Title"]
    else:
        title = "Density Plot"

    if args.x_label is not None:
        x_label = args.x_label
    elif "XLabel" in header:
        x_label = header["XLabel"]
    else:
        x_label = "X"

    if args.y_label is not None:
        y_label = args.y_label
    elif "YLabel" in header:
        y_label = header["YLabel"]
    else:
        y_label = "Y"

    if args.z_label is not None:
        z_label = args.z_label
    elif "ZLabel" in header:
        z_label = header["ZLabel"]
    else:
        z_label = "Z"

    if args.contours == "smooth":
        matrix_smooth = ndimage.gaussian_filter(matrix, sigma=2.0, order=0)

    ax = figure.add_subplot(111)
    pcol = ax.pcolormesh(matrix, vmin=0, vmax=args.z_max,
                         cmap=cmaps.viridis,
                         rasterized=args.rasterised)
    ax.axis([0,args.x_max,0,args.y_max])
    cbar = figure.colorbar(pcol, label=wrap(z_label))
    cbar.solids.set_rasterized(args.rasterised)
    if args.z_max > 0:
        levels = np.arange(args.z_max/8, args.z_max, args.z_max/8)
        if args.contours == "normal":
            ax.contour(matrix, colors="white",
                       alpha=0.6, levels=levels)
        elif args.contours == "smooth":
            ax.contour(matrix_smooth, colors="white",
                       alpha=0.6, levels=levels)

    ax.set_title(wrap(title))
    ax.set_xlabel(wrap(x_label))
    ax.set_ylabel(wrap(y_label))
    ax.grid(True, color="white", alpha=0.2)


def main(args):
    with open(args.matrix_file) as input_file:
        header = readheader(input_file)
        matrix = np.loadtxt(input_file)

    figure = matplotlib.pyplot.figure(figsize=(args.width, args.height),
                                      facecolor="white",
                                      tight_layout=True)

    if "Transpose" in header and header["Transpose"] == '1':
        matrix = np.transpose(matrix)

    if args.verbose:
        print("{:d} by {:d} matrix file loaded.".format(matrix.shape[0],
                                                        matrix.shape[1]))

    if args.x_max is None or args.y_max is None or args.z_max is None:
        xmax,ymax,zmax = find_peaks(matrix)
        if args.verbose:
            print("Automatically detected axis limits:")
            print("xmax: ", xmax)
            print("ymax: ", ymax)
            print("zmax: ", zmax)

    if args.x_max is None:
        args.x_max = xmax
    if args.y_max is None:
        args.y_max = ymax
    if args.z_max is None:
        args.z_max = zmax

    if args.output:
        if args.output_type is not None:
            output_name = args.output + '.' + args.output_type
        else:
            output_name = args.output
        make_figure(figure,matrix,header,args)
        figure.savefig(correct_filename(output_name), dpi=args.dpi)
    else:
        app = QtGui.QApplication(sys.argv)
        main = MainWindow(figure,matrix,header,args)
        main.show()
        sys.exit(app.exec_())


if __name__ == '__main__':
    args = get_args()
    main(args)
