
from PyQt5 import QtCore, QtGui, QtWidgets
from Main_window import Ui_MainWindow
from tag_window import Ui_Dialog
from video_window import Ui_Window
from navigation_window import Ui_Dialog_navigation
from tag_class import Tag
import sys
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QVideoProbe
import scipy.io as spio
from Data_analysis import DataAnalysis
import pyqtgraph as pg
import numpy as np
import cv2
from openCVQimage import OpenCVQImage
import pandas as pd

# Initializing the application using template
class Visualizer:

    def __init__(self, Ui_template):
        self.app = QtGui.QApplication([])
        self.win = QtGui.QMainWindow()
        self.ui = Ui_template()
        self.ui.setupUi(self.win)
        # tag dialog
        self.dialog_box = QtGui.QDialog()
        self.tag_dialog = Ui_Dialog()
        self.tag_dialog.setupUi(self.dialog_box)
        # navigator dialog
        self.nav_dialog_box = QtGui.QDialog()
        self.nav_dialog = Ui_Dialog_navigation()
        self.nav_dialog.setupUi(self.nav_dialog_box)

        # Initializing the video player and output
        self.mediaPlayer= cv2.VideoCapture(0)
        # Initializing video probe
        self._timer = QtCore.QTimer()
        self._stim_timer = QtCore.QElapsedTimer()
        self._timer.stop()

        # Connecting the timer to the frame grabbing
        self._timer.timeout.connect(self.grabFrame)
        self._timer.timeout.connect(self.update_plots)
        self._timer.timeout.connect(self.update_trigger_plot)
        self._timer.timeout.connect(self.positionChanged)
        self._timer.timeout.connect(self.display_stimulus_data)
        # Connecting the toolbar actions
        self.ui.actionLoad_Video.triggered.connect(self.load_video)
        self.ui.actionLoad_data.triggered.connect(self.load_data)
        self.ui.play_push.setIcon(self.win.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.ui.play_push.clicked.connect(self.play_all)
        self.ui.spinBox_offset.valueChanged.connect(self.set_offset)
        self.ui.fps_spin_box.valueChanged.connect(self.set_fps)
        self.ui.actionTag.triggered.connect(self.open_tag_dialog)
        #
        self.ui.listWidget.itemClicked.connect(self.create_checked_vector)
        # Tagging window
        btn = self.tag_dialog.button_box.button(self.tag_dialog.button_box.Apply)
        btn.clicked.connect(self.add_tag_to_db)
        self.tag_dialog.start_time.selectionChanged.connect(lambda:  self.update_tag_values('start'))
        self.tag_dialog.end_time.selectionChanged.connect(lambda: self.update_tag_values('end'))
        self.tag_dialog.Delete_row.clicked.connect(self.delete_db_row)
        self.tag_dialog.load_csv.clicked.connect(self.load_csv_to_db)
        self.tag_dialog.tableWidget.cellClicked.connect(self.cellClick)
        # tagging class self.tag_dialog.Export.clicked.connect(self.export_to_csv)

        self.tag_db = Tag()
        # Slider interactions
        self.ui.positionSlider.sliderMoved.connect(self.setPosition)

        # Next and previous frames
        self.ui.prev_frame_push.clicked.connect(self.prev_frame)
        self.ui.next_frame_push.clicked.connect(self.next_frame)

        # Go to frame
        self.ui.spinBoxGoFrame.valueChanged.connect(self.go_to_frame)
        # Go to time
        self.ui.go_to_time_spin.valueChanged.connect(self.go_to_time)
        # Range
        self.ui.spinBox_min.valueChanged.connect(self.set_y_range)
        self.ui.spinBox_max.valueChanged.connect(self.set_y_range)
        # Menu items
        self.ui.actionPreprocessed_data.triggered.connect(self.load_prepros)
        self.ui.actionRaw_data.triggered.connect(self.load_raw)
        self.ui.actionICA.triggered.connect(self.load_ica)
        self.ui.actionRMS.triggered.connect(self.load_rms)
        self.ui.actionTime_of_signal.triggered.connect(self.display_time)
        self.ui.actionFrames.triggered.connect(self.display_frame)
        self.ui.actionSlice_navigator.triggered.connect(self.navigation_panel)
        # Video screen  /    Adding a scene
        self.scene = QtWidgets.QGraphicsScene()
        self.ui.videoWindow.setScene(self.scene)
        self.frame = QtGui.QPixmap()
        self.scene.addPixmap(self.frame)
        # initial offset between data and video
        self.n_of_frames = 0
        # Time Span and amplitude configurations
        # number of seconds to display
        self.numOfSec  = 3
        # min Y max Y
        self.dispRange = [0,100]
        # Display methood
        self.display_frame_bool = False
        # Update y axes range display
#        self.ui.spinBox_min.setValue(self.dispRange[0])
#        self.ui.spinBox_max.setValue(self.dispRange[1])
        # For Timer Stimulus LCD
        self.prev_stim = 1000
        # Load data from intan file

    def load_video(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.win, "Load video",
                                                         QtCore.QDir.currentPath())
        if fileName != '':
            self.mediaPlayer.open(fileName)
            # find fps
            self.fps = self.mediaPlayer.get(5) # Prop id = FPS
            self.ui.fps_spin_box.setValue(self.fps)
            self._timer.setInterval(1000 / self.fps)
            self.durationChanged()

    def load_data(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.win, "Load data",
                                                         QtCore.QDir.currentPath())
        if fileName != '':
            # Applying Data Analysis on the file
            #  new_mat = spio.loadmat(fileName, appendmat=False)
            # initializing the parameters
            self.data_analyze = DataAnalysis(fileName)
            self.sig = self.data_analyze.data_vec
            self.sampling_frequency = self.data_analyze.sample_rate
            self.number_of_channels = len(self.sig[:, 0])
            self.time_vector = self.data_analyze.time_sync
            self.trigger_vector = self.data_analyze.trigger_vector
            # initialize a plt handle for each channel
            # creating the plots list
            self.plot_id = [channel for channel in range(self.number_of_channels)]
            self.plt = [0 for channel in range(self.number_of_channels+1)]
            self.plt_trig = [0 for channel in range(2)]
            self.curve = [0 for channel in range(self.number_of_channels+1)]
            # Initiate checkbox list
            self.cb = [0 for channel in range(self.number_of_channels+1)]
            self.prev_num_of_channels = self.number_of_channels
            # Create plots
            self.re_initiate_plots()
            # Create checboxes
            self.initialize_list()
            # Insert plot triggers
            self.insert_plot_triggers()

    def load_raw(self):
        self.sig = np.array(self.data_analyze.data_vec)
        self.prev_num_of_channels = self.number_of_channels
        self.number_of_channels = len(self.sig)
        self.re_initiate_plots()
        self.initialize_list()
        self.insert_plot_triggers()

    def load_prepros(self):
        self.sig = np.array(self.data_analyze.filtered_data)
        self.prev_num_of_channels = self.number_of_channels
        self.number_of_channels = len(self.sig)
        self.re_initiate_plots()
        self.initialize_list()
        self.insert_plot_triggers()

    def load_ica(self):
        self.sig = np.array(self.data_analyze.apply_ica()).T
        self.prev_num_of_channels = self.number_of_channels
        self.number_of_channels = len(self.sig)
        self.re_initiate_plots()
        self.initialize_list()
        self.insert_plot_triggers()

    def load_rms(self):
        self.sig = np.array(self.data_analyze.apply_rms())
        self.prev_num_of_channels = self.number_of_channels
        self.number_of_channels = len(self.sig)
        self.re_initiate_plots()
        self.initialize_list()
        self.insert_plot_triggers()

    def re_initiate_plots(self):
        self._timer.stop()
        if (self.prev_num_of_channels != 0) & (type(self.plt[0]) != int):
            for channel in range(self.prev_num_of_channels):
                self.ui.graphicsView.removeItem(self.plt[channel])
            self.ui.graphicsView.nextRow()
        for channel in range(self.number_of_channels):
            self.plt[channel] = self.ui.graphicsView.addPlot()
            self.curve[channel] = self.plt[channel].plot()
            self.plt[channel].setDownsampling(ds=20 ,mode='peak', auto=False)
            self.plt[channel].setClipToView(True)
            #self.plt[channel].setLabel('bottom', 'Time', 's')
            self.plt[channel].setRange(xRange=[-1 * self.numOfSec/2, self.numOfSec/2])
            self.plt[channel].setRange(yRange=self.dispRange)
            self.plt[channel].setLabel(axis = 'left', text = 'Ch %s' % (self.plot_id[channel]+1))
            self.plt[channel].addLine(x=0)
                # Creating buffer - bufferSize = one sec
            if (channel % 3 == 2) & (channel != 1) & (channel != self.number_of_channels-1):
                self.ui.graphicsView.nextRow()

    def insert_plot_triggers(self):
        channel = self.number_of_channels
        if self.plt_trig[0] != 0 & (self.ui.listWidget.item(0).checkState() == 0):
            self.ui.graphicsView.removeItem(self.plt_trig[0])
            self.plt_trig[0] = 0
        if ((self.ui.listWidget.item(0).checkState() == 2) & (self.plt_trig[0] == 0)):
            if (channel % 3) == 0:
                self.ui.graphicsView.nextRow()
            self.plt_trig[0] = self.ui.graphicsView.addPlot()
            self.plt_trig[0].setLabel(axis = 'left', text = 'Triggers')
            self.plt_trig[1] =  self.plt_trig[0].plot()
            #self.curve[channel].setPen()
            self.plt_trig[1].setData(x = self.time_vector, y = self.trigger_vector)

            self.trigger_line = self.plt_trig[0].addLine(x=0)
            self.plt_trig[0].setRange(xRange=[40, 600])
            self.plt_trig[0].setRange(yRange=[0, 100])
            self.plt_trig[0].setDownsampling(ds = 50,mode='subsample', auto = False)
            self.plt_trig[0].setClipToView(True)


    def grabFrame(self):
        if self.mediaPlayer.isOpened():
            _frame = self.mediaPlayer.read()
            if _frame[1] is not None:
                self.scene.clear()
                _frame = cv2.cvtColor(_frame[1] ,cv2.COLOR_BGR2RGB)
                qimg = QtGui.QImage(_frame.data, _frame.shape[1], _frame.shape[0], QtGui.QImage.Format_RGB888)
                self.frame = self.frame.fromImage(qimg)
                self.scene.addPixmap(self.frame)
                # Fit view - Try once after load file
                self.ui.videoWindow.ensureVisible(self.scene.sceneRect())
                self.ui.videoWindow.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def play_all(self):
        self._timer.setSingleShot(False)
        if (self.mediaPlayer.isOpened()):
            if self._timer.isActive() == True:
                self._timer.stop()

            else:
                self._timer.start()

    def setPosition(self):
        # get 3 id - relative position of the video [0,1]
        # 1 next frame to be captured
        if (self.mediaPlayer.isOpened()):
            self.mediaPlayer.set(1, self.ui.positionSlider.value())

    def positionChanged(self):
        if (self.mediaPlayer.isOpened()):
            # get 2 id - relative position of the video [0,1]
            self.ui.positionSlider.setValue(self.mediaPlayer.get(1))
            # Position number in frames
            if self.display_frame_bool == True:
                self.ui.lcdNumber.display(self.mediaPlayer.get(1))
            else:
                self.ui.lcdNumber.display(self.current_sig_location)

    def durationChanged(self):
        self.ui.positionSlider.setRange(0, self.mediaPlayer.get(7)) # 8 number of frame in the video file

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.ui.play_push.setIcon(
                self.win.style().standardIcon(QtWidgets.QStyle.SP_MediaPause))
        else:
            self.ui.play_push.setIcon(
                self.win.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

    def update_plots(self):
        try:
            if (('plt' in dir(self)) & (self.mediaPlayer.isOpened())):
                vid_indices = self.mediaPlayer.get(1) # next frame position
                sig_indices = int(np.floor(vid_indices*(self.sampling_frequency/self.fps)))
                plotCenter = int(self.numOfSec * self.sampling_frequency/2)
                for channel in range(self.number_of_channels):
                    self.curve[channel].setData(x = self.time_vector[max(0,(sig_indices + self.n_of_frames) + 1 - plotCenter):sig_indices+1+ self.n_of_frames + plotCenter]
                        , y = self.sig[self.plot_id[channel],max(0,(sig_indices +self.n_of_frames) + 1 - plotCenter):sig_indices + 1+ self.n_of_frames + plotCenter])
                    self.curve[channel].setPos(-(sig_indices+self.n_of_frames)/self.sampling_frequency,0) # position is the right most time sample.
                self.current_sig_location = self.time_vector[sig_indices + self.n_of_frames + 1]
        except:
            print('Couldnt update plot location')
    def set_offset(self):
        self.n_of_frames = self.ui.spinBox_offset.value()

    def set_fps(self):
        self.fps = self.ui.fps_spin_box.value()
        self._timer.setInterval(1000 / self.fps)

    def next_frame(self):
        if (self.mediaPlayer.isOpened()):
            if self._timer.isActive() == True:
                self._timer.stop()
            self.mediaPlayer.set(1, self.mediaPlayer.get(1))
            self._timer.setSingleShot(True) # Trigger relevant finctions
            self._timer.start()

    def prev_frame(self):
        if self.mediaPlayer.isOpened():
            if self._timer.isActive() is True:
                self._timer.stop()
            if(self.mediaPlayer.get(1)-2) > 0:
                self.mediaPlayer.set(1, self.mediaPlayer.get(1)-2)
                self._timer.setSingleShot(True) # Trigger relevant functions
                self._timer.start()

    def set_y_range(self):
        if self.mediaPlayer.isOpened():
            self.dispRange = [self.ui.spinBox_min.value(),
                              self.ui.spinBox_max.value()]
            for channel in range(self.number_of_channels):
                self.plt[channel].setRange(yRange=self.dispRange)

    def go_to_frame(self):
        if self.mediaPlayer.isOpened():
            frame_number = self.ui.spinBoxGoFrame.value()
            self.mediaPlayer.set(1, frame_number)

    def initialize_list(self):
        if self.ui.listWidget.count != self.number_of_channels:
            self.ui.listWidget.clear()
            self.cb[0] = QtGui.QListWidgetItem()
            self.cb[0].setText('Triggers')
            self.cb[0].setFlags(self.cb[0].flags() | QtCore.Qt.ItemIsUserCheckable)
            self.cb[0].setCheckState(QtCore.Qt.Checked)
            self.ui.listWidget.addItem(self.cb[0])
        for channel in range(1,self.number_of_channels+1):
            self.cb[channel] = QtGui.QListWidgetItem()
            self.cb[channel].setText('Channel %s' % (channel))
            self.cb[channel].setFlags(self.cb[channel].flags() | QtCore.Qt.ItemIsUserCheckable)
            self.cb[channel].setCheckState(QtCore.Qt.Checked)
            self.ui.listWidget.addItem(self.cb[channel])

    def create_checked_vector(self):
        if self._timer.isActive() is True:
            self._timer.stop()
        self.plot_id.clear()
        for channel in range(1,len(self.sig[:, 0])+1):
            if self.ui.listWidget.item(channel).checkState() == 2:
                self.plot_id.append(channel-1)
        self.prev_num_of_channels = self.number_of_channels
        self.number_of_channels = len(self.plot_id)
        self.re_initiate_plots()
        self.insert_plot_triggers()

    def check_all(self):
        for channel in range(0,len(self.sig[:, 0])):
            self.ui.listWidget.item(channel).setCheck(True)

    def un_check_all(self):
        for channel in range(0,len(self.sig[:, 0])):
            self.ui.listWidget.item(channel).setCheck(False)

    def update_trigger_plot(self):
        try:
            channel = self.number_of_channels
            if self.ui.listWidget.item(0).checkState() == 2:
                vid_indices = self.mediaPlayer.get(1)  # next frame position
                sig_indices = int((vid_indices - 1) * (self.sampling_frequency / self.fps))
                self.cur_pos_ind = sig_indices + self.n_of_frames
                self.trigger_line.setValue(self.time_vector[sig_indices + self.n_of_frames])
                axis = self.plt_trig[0].getAxis('bottom').range
                if (axis[0] > self.current_sig_location) or (axis[1] < self.current_sig_location):
                    self.plt_trig[0].setRange(
                        xRange=[self.current_sig_location - 1, self.current_sig_location + (axis[1] - axis[0])])
        except:
            return

    def display_time(self):
        self.display_frame_bool = False

    def display_frame(self):
        self.display_frame_bool = True

    def go_to_time(self):
        try:
            if self.mediaPlayer.isOpened():
                next_value = self.ui.go_to_time_spin.value()
                cur_value = self.current_sig_location
                added_frame = int((next_value-cur_value)*self.fps)
                self.mediaPlayer.set(1, max(1, self.mediaPlayer.get(1) + added_frame))
        except:
            return 'Cant go to requested time'

    def display_stimulus_data(self):
        try:
            if self.mediaPlayer.isOpened():
                self.current_stim = self.trigger_vector[self.cur_pos_ind]
                self.initial_stim_time = self.time_vector[np.argmax(self.trigger_vector == self.current_stim)]
                if (self.current_stim != self.prev_stim):
                    self.ui.lcdNumber_2.display(self.current_stim - 64)
                    self.prev_stim = self.current_stim
                diff = self.current_sig_location - self.initial_stim_time
                self.ui.lcdNumber_stim.display(diff)
        except:
            return

    def change_down_sampling_rate(self):
        for channel in range(self.number_of_channels):
            self.plt[channel].setDownsampling(ds=20, mode='peak', auto=False)

    def open_tag_dialog(self):
        self.dialog_box.show()

    def update_tag_values(self, string):
        try:
            if string is 'start':
                self.tag_dialog.start_time.setText(str(self.current_sig_location))
                self.tag_dialog.stim_number.setText(str(self.current_stim-64))
                self.tag_dialog.stim_time.setText(str(self.current_sig_location-self.initial_stim_time))
            elif string is 'end':
                self.tag_dialog.end_time.setText(str(self.current_sig_location))

        except:
            pass
        # self.tag_dialog.start_time.setText('Start Time')
        self.dialog_box.show()

    def add_tag_to_db(self):

        try:
            start = float(self.tag_dialog.start_time.text())
            end = float(self.tag_dialog.end_time.text())
            classification = self.tag_dialog.classification.text()
            electrodes = self.tag_dialog.electrodes.text()
            stimulus = int(self.tag_dialog.stim_number.text())
            stim_time = float(self.tag_dialog.stim_time.text())
            # update database
            self.tag_db.add_item(classification, start, end-start, electrodes, stimulus,
                                 stim_time)
            # update table
            self.set_items_in_tag_table()
        except(ValueError, TypeError):
            print('Could not add entry to DB')
        return 'added entry to data base'

    def set_items_in_tag_table(self):

        for micro_expression in self.tag_db.database:
            self.tag_dialog.tableWidget.setRowCount(len(self.tag_db.database))
            for counter, value in enumerate(micro_expression, start=0):
                self.tag_dialog.tableWidget.setItem(micro_expression.ME_number, counter, QtWidgets.QTableWidgetItem(str(value)))

    def delete_db_row(self):
        row_to_delete = self.tag_dialog.tableWidget.selectionModel().selectedRows()
        for row in row_to_delete:
            self.tag_dialog.tableWidget.removeRow(row.row())
            self.tag_db.remove_item(row.row())

    def load_csv_to_db(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.win, "Load csv",
                      QtCore.QDir.currentPath())
        self.df = pd.read_table(fileName, header=[0,1])
        for idx, row in self.df.iterrows():
            start = row[0]
            end = row[1]
            self.tag_db.add_item('UC', start, end-start, 'NaN', 'NaN',
                                 'NaN')
        self.set_items_in_tag_table()


    def export_to_csv(self):
        self.tag_db.export()

    def navigation_panel(self):
        self.nav_dialog_box.show()

    def cellClick(self, row, col):
        loc = self.tag_dialog.tableWidget.item(row, 2)
        self.ui.go_to_time_spin.setValue(float(loc.text()))

if __name__ == '__main__':

    vis = Visualizer(Ui_MainWindow)
    vis.win.show()
    sys.exit(vis.app.instance().exec_())
