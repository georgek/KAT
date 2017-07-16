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
import matplotlib.backends.backend_qt4agg

import kat_plot_colormaps as cmaps

from PyQt4 import QtCore, QtGui

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
    figure = matplotlib.pyplot.figure(figsize=(width, height),
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
        figure = matplotlib.pyplot.figure(figsize=(self.args.width,
                                                   self.args.height),
                                          facecolor="white",
                                          tight_layout=True)
        self.make_figure(figure, self.matrix, self.args)
        figure.savefig(correct_filename(filename), dpi=self.args.dpi)


    def redraw(self):
        self.redraw_event.set()


    def async_redraw(self, redraw_event, end_event):
        while True:
            redraw_event.wait()
            redraw_event.clear()
            if end_event.isSet(): return
            self.figure.clear()
            self.make_figure(self.figure, self.matrix, self.args)
            self.canvas.draw()
