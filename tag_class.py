import collections
import csv
import pickle

class Tag:

    def __init__(self, database=None):

        if database is None:
            self.database = list()

        else:
            self.database = database

        self.subject_class = collections.namedtuple('Database', ['ME_number', 'Type', 'Video_time', 'Length', 'Electrode',
                                                    'Stimulus_number', 'Stimulus_time'])

    def add_item(self, type='UC', video_time=0, length=0, electrode=-1, stimulus_num=0, stimulus_time=0):

        last_subj = self.subject_class(len(self.database), type, video_time, length, electrode, stimulus_num,
                                       stimulus_time)
        self.database.append(last_subj)
        # add exception

    def remove_item(self, item):
        try:
            del self.database[item]
        except:
            return

    def export(self):

        with open('database.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(['ME number', 'Type', 'Video time', 'Length', 'Electrodes',
                                                    'Stimulus number', 'Stimulus_time'])
            for row in self.database:
                wr.writerow(row)

    def import_from_file(self):
        try:
            with open('database.csv', 'r') as myfile:
                self.database = list()
                reader = csv.reader(myfile)
                for ind, row in enumerate(reader):
                    if ind > 0:
                        self.add_item(row[1], row[2], row[3], row[4], row[5], row[6])
        except:
            return


if __name__ == '__main__':
    t = Tag()
    t.add_item()
    t.add_item()
    t.export()
    t.import_from_file()