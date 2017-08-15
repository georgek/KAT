#!/usr/bin/env python3

import sys
import argparse
import threading
import logging
import functools
import textwrap

import numpy as np
import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot

import kat_plot_colormaps as cmaps

from PyQt4 import QtCore, QtGui
import matplotlib.backends.backend_qt4agg

KAT_VERSION = "2.3.4"

ABOUT = """<h3>About KAT</h3>

<p>This is KAT version {:s}.</p>

<p>KAT is a suite of tools that analyse jellyfish hashes or sequence files
(fasta or fastq) using kmer counts.</p>

<p>KAT is licensed under the GNU General Public Licence v3.</p>

<p>If you use KAT in your work and wish to cite us please use the following citation:</p>

<p>Daniel Mapleson, Gonzalo Garcia Accinelli, George Kettleborough, Jonathan
Wright, and Bernardo J. Clavijo.  <b>KAT: A K-mer Analysis Toolkit to quality
control NGS datasets and genome assemblies</b> <i>Bioinformatics</i>,
2016. doi: <a
href="http://dx.doi.org/10.1093/bioinformatics/btw663">10.1093/bioinformatics/btw663</a></p>
""".format(KAT_VERSION)

ONLINE_DOC_URL = "http://kat.readthedocs.io/en/latest/"

CONTOUR_OPTIONS = ["none", "normal", "smooth"]


def get_common_argparser():
    parser = argparse.ArgumentParser(add_help=False)
    input_group = parser.add_argument_group("Input options")
    output_group = parser.add_argument_group("Output options")
    output_group.add_argument("-o", "--output", type=str,
                              help="The path to the output file.")
    output_group.add_argument("-p", "--output_type", type=str,
                              help="The plot file type to create (default is " \
                              "based on given output name).")
    output_group.add_argument("-w", "--width", type=int, default=20,
                              help="Width of canvas (cm)")
    output_group.add_argument("-l", "--height", type=int, default=15,
                              help="Height of canvas (cm)")
    output_group.add_argument("-r", "--resolution", type=int, default=300,
                              help="Resolution in dots per inch (dpi) of output graphic.")
    plot_group = parser.add_argument_group("Plot options")
    plot_group.add_argument("-t", "--title", type=str,
                            help="Title for plot")
    misc_group = parser.add_argument_group("Miscellaneous options")
    misc_group.add_argument("-v", "--verbose", dest="verbose",
                            action="store_true",
                            help="Print extra information")
    misc_group.set_defaults(verbose=False)
    misc_group.add_argument("--debug", dest="debug",
                            action="store_true",
                            help=argparse.SUPPRESS)
    misc_group.set_defaults(debug=False)
    misc_group.add_argument("-h", "--help", action="help",
                            help="show this help message and exit")

    return parser,input_group,output_group,plot_group,misc_group


def only_type(typ):
    """Only runs the function if the last argument can be coerced into type,
otherwise does nothing.
    """
    def only_type_decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args):
            try:
                v = typ(args[-1])
            except ValueError:
                return
            fun(*(args[:-1] + (v,)))
        return wrapper
    return only_type_decorator


def update_text_box(textBox, val):
    textBox.setText(str(val))


@only_type(int)
def update_slider(slider, val):
    slider.setSliderPosition(val)


CM_PER_INCH = 2.54


def cm2inch(v):
    return v/CM_PER_INCH


def readheader(input_file):
    header = {}
    for line in input_file:
        if line[0:2] == "# ":
            s = line[2:-1].split(":")
            n = s[0]
            v = ":".join(s[1:])
            header[n] = v
        elif line[:-1] == "###":
            break
        else:
            break
    return header


def find_peaks(a):
    a = np.squeeze(np.asarray(a))
    ad = np.sign(np.diff(a))
    # remove zeros to find end of plateaus
    ad[ad == 0] = 1
    return np.where(np.diff(ad) == -2)[0] + 1


def correct_filename(filename):
    split = filename.split('.')
    if len(split) > 1:
        ext = split[-1]
    else:
        ext = ''
    types = matplotlib.pyplot.gcf().canvas.get_supported_filetypes().keys()
    if ext in types:
        return filename
    elif "png" in types:
        return filename + ".png"
    elif "pdf" in types:
        return filename + ".pdf"
    else:
        return filename + "." + types[0]


def wrap(name):
    return "\n".join(textwrap.wrap(name, 60))


def new_figure(width, height):
    "Makes new figure with width and height in cm."
    figure = matplotlib.pyplot.figure(figsize=(cm2inch(width),
                                               cm2inch(height)),
                                      facecolor="white",
                                      tight_layout=True)
    return figure


class KatPlotWindow(QtGui.QMainWindow):
    def __init__(self, matrix, args, make_figure_fun):
        super().__init__()
        self.matrix = matrix
        self.args = args
        self.make_figure = make_figure_fun

        self.setWindowTitle("KAT plot")
        self.setWindowIcon(QtGui.QIcon("kat_logo.png"))

        self.figure = new_figure(self.args.width, self.args.height)
        self.canvas = matplotlib.backends.backend_qt4agg.FigureCanvasQTAgg(self.figure)
        self.setCentralWidget(self.canvas)

        self.redraw_event = threading.Event()
        self.end_event = threading.Event()
        self.drawthread = threading.Thread(target=self.async_redraw,
                                           args=(self.redraw_event, self.end_event))

        self.dockwidth = 2

        logging.info("Screen dpi: %s, %s",
                     self.logicalDpiX(),
                     self.logicalDpiY())

        self.drawthread.start()
        self.redraw()


    def closeEvent(self, event):
        self.end_event.set()
        self.redraw()
        super().closeEvent(event)


    def open_online_docs(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(ONLINE_DOC_URL))


    def about_window(self):
        mbox = QtGui.QMessageBox(self)
        mbox.setWindowTitle("About KAT")
        mbox.setText(ABOUT)
        mbox.setIconPixmap(self.windowIcon().pixmap(100, 100))
        mbox.exec_()


    def save_as(self):
        filename = QtGui.QFileDialog.getSaveFileName(self)
        logging.info("Filename given: %s", filename)
        figure = matplotlib.pyplot.figure(figsize=(cm2inch(self.args.width),
                                                   cm2inch(self.args.height)),
                                          facecolor="white",
                                          tight_layout=True)
        self.make_figure(figure, self.matrix, self.args)
        figure.savefig(correct_filename(filename), dpi=self.args.resolution)


    def redraw(self):
        self.redraw_event.set()


    def async_redraw(self, redraw_event, end_event):
        while True:
            redraw_event.wait()
            redraw_event.clear()
            if end_event.is_set(): return
            self.figure.clear()
            self.make_figure(self.figure, self.matrix, self.args)
            self.canvas.draw()


    def make_menus(self, extra_menu_fun):
        """Makes the standard menus and calls extra_menu_fun before the help menu
which can be used to add extra menus.
        """
        file_menu = self.menuBar().addMenu("&File")
        a = QtGui.QAction("Save as...", self)
        file_menu.addAction(a)
        a.triggered.connect(self.save_as)
        a = QtGui.QAction("&Quit", self)
        file_menu.addAction(a)
        a.triggered.connect(self.close)

        extra_menu_fun()

        help_menu = self.menuBar().addMenu("&Help")
        a = QtGui.QAction("KAT documentation", self)
        help_menu.addAction(a)
        a.triggered.connect(self.open_online_docs)
        help_menu.addSeparator()
        a = QtGui.QAction("About Qt", self)
        help_menu.addAction(a)
        a.triggered.connect(functools.partial(QtGui.QMessageBox.aboutQt, self))
        a = QtGui.QAction("About KAT", self)
        help_menu.addAction(a)
        a.triggered.connect(self.about_window)


    def make_axis_dock(self):
        self.axisdock = QtGui.QDockWidget("Axis limits")
        self.axisdock.setAutoFillBackground(True)
        self.axisdock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                             QtGui.QDockWidget.DockWidgetMovable)
        palette = self.axisdock.palette()
        palette.setColor(self.axisdock.backgroundRole(), QtCore.Qt.white)
        self.axisdock.setPalette(palette)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.axisdock)

        self.sliders = QtGui.QWidget()
        self.sliders.setFixedWidth(self.x_px_dim(self.dockwidth))
        self.sliders.setFixedHeight(self.y_px_dim(2))
        self.sliders_grid = QtGui.QGridLayout(self.sliders)
        self.axisdock.setWidget(self.sliders)


    def add_axis_slider(self, lab, fun, init, maximum, col):
        logging.debug("add_slider: %s", locals())
        label = QtGui.QLabel(lab, self.sliders)
        label.setAlignment(QtCore.Qt.AlignCenter)
        textbox = QtGui.QLineEdit(self.sliders)
        textbox.setText(str(init))
        textbox.setCursorPosition(0)
        # let the user type a higher number as this will be corrected by
        # the slider
        textbox.setValidator(QtGui.QIntValidator(1, maximum*10))
        sld = QtGui.QSlider(QtCore.Qt.Vertical, self.sliders)
        sld.setMinimum(1)
        sld.setMaximum(maximum)
        sld.setSliderPosition(init)
        sld.setTickInterval(maximum/10)
        sld.setTickPosition(QtGui.QSlider.TicksRight)
        sld.setFocusPolicy(QtCore.Qt.NoFocus)
        textbox.textChanged[str].connect(functools.partial(update_slider, sld))
        sld.valueChanged[int].connect(fun)
        sld.valueChanged[int].connect(functools.partial(update_text_box, textbox))
        sld.valueChanged.connect(self.redraw)
        self.sliders_grid.addWidget(label,   0, col)
        self.sliders_grid.addWidget(textbox, 1, col)
        self.sliders_grid.addWidget(sld,     2, col)


    def make_labels_dock(self):
        self.labelsdock = QtGui.QDockWidget("Labels")
        self.labelsdock.setAutoFillBackground(True)
        self.labelsdock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                                    QtGui.QDockWidget.DockWidgetMovable)
        palette = self.labelsdock.palette()
        palette.setColor(self.labelsdock.backgroundRole(), QtCore.Qt.white)
        self.labelsdock.setPalette(palette)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.labelsdock)

        self.labelsopts = QtGui.QWidget()
        self.labelsopts.setFixedWidth(self.x_px_dim(self.dockwidth))
        self.labelsopts.setFixedHeight(self.y_px_dim(1.5))
        self.labelsopts_grid = QtGui.QGridLayout(self.labelsopts)
        self.labelsdock.setWidget(self.labelsopts)


    def add_labels_input(self, lab, fun, init, row):
        label = QtGui.QLabel(lab, self.labelsopts)
        textbox = QtGui.QLineEdit(self.labelsopts)
        textbox.setText(str(init))
        textbox.setCursorPosition(0)
        textbox.textChanged[str].connect(fun)
        textbox.textChanged.connect(self.redraw)
        self.labelsopts_grid.addWidget(label,   row, 0, 1, 1)
        self.labelsopts_grid.addWidget(textbox, row, 1, 1, 1)


    def make_output_dock(self):
        self.outputdock = QtGui.QDockWidget("Output")
        self.outputdock.setAutoFillBackground(True)
        self.outputdock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable |
                                    QtGui.QDockWidget.DockWidgetMovable)
        palette = self.outputdock.palette()
        palette.setColor(self.outputdock.backgroundRole(), QtCore.Qt.white)
        self.outputdock.setPalette(palette)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.outputdock)

        self.outputopts = QtGui.QWidget()
        self.outputopts.setFixedWidth(self.x_px_dim(self.dockwidth))
        self.outputopts.setFixedHeight(self.y_px_dim(1.5))
        self.outputopts_grid = QtGui.QGridLayout(self.outputopts)
        self.outputdock.setWidget(self.outputopts)
        self.add_output_input("Width",     "cm", self.set_width, self.args.width,
                              0, QtGui.QDoubleValidator())
        self.add_output_input("Height",    "cm", self.set_height,self.args.height,
                              1, QtGui.QDoubleValidator())
        self.add_output_input("Resolution","dpi",self.set_dpi,   self.args.resolution,
                              2, QtGui.QIntValidator())
        self.add_save_button()


    def add_output_input(self, lab, unit, fun, init, row, validator):
        label = QtGui.QLabel(lab, self.outputopts)
        unit = QtGui.QLabel(unit, self.outputopts)
        textbox = QtGui.QLineEdit(self.outputopts)
        textbox.setText(str(init))
        textbox.setCursorPosition(0)
        textbox.setValidator(validator)
        textbox.textChanged[str].connect(fun)
        self.outputopts_grid.addWidget(label,   row, 0, 1, 1)
        self.outputopts_grid.addWidget(textbox, row, 1, 1, 1)
        self.outputopts_grid.addWidget(unit,    row, 2, 1, 1)


    def add_save_button(self):
        savebutton = QtGui.QPushButton("&Save as...")
        savebutton.clicked.connect(self.save_as)
        self.outputopts_grid.addWidget(savebutton, 3, 0, 1, 3)


    def x_px_dim(self, len):
        return len * self.logicalDpiX()


    def y_px_dim(self, len):
        return len * self.logicalDpiY()


    @only_type(float)
    def set_width(self, v):
        logging.info("output width changed: %f", v)
        self.args.width = v


    @only_type(float)
    def set_height(self, v):
        logging.info("output height changed: %f", v)
        self.args.height = v


    @only_type(int)
    def set_dpi(self, v):
        logging.info("output resolution changed: %f", v)
        self.args.resolution = v


    def set_title(self, v):
        self.args.title = str(v)
