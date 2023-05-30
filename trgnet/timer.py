import time
from enum import Enum, auto

import matplotlib.pyplot as plt


class Status(Enum):
    CREATED = auto()
    STARTED = auto()
    FINISHED = auto()


class Record:
    def __init__(self, identifier):
        self.identifier = identifier
        self.start_time = None
        self.end_time = None
        self.status = Status.CREATED

    def start(self):
        self.start_time = time.time()
        self.status = Status.STARTED

    def stop(self):
        self.end_time = time.time()
        self.status = Status.FINISHED

    def total_time(self):
        if self.status == Status.FINISHED:
            return self.end_time - self.start_time
        return None


class Timer:
    def __init__(self):
        self.records = {}

    def start(self, topic):
        if not self.records.get(topic):
            self.records[topic] = []
        current_record = (
            self.records[topic][-1] if len(self.records[topic]) > 0 else None
        )
        if current_record is None or current_record.status == Status.FINISHED:
            new_record = Record(len(self.records[topic]))
            new_record.start()
            self.records[topic].append(new_record)
        else:
            if current_record.status == Status.CREATED:
                current_record.start()
            elif current_record.status == Status.STARTED:
                raise Exception(f"A record on the topic '{topic}' has already started.")

    def stop(self, topic):
        records = self.records.get(topic, [])
        current_record = records[-1] if len(records) > 0 else None

        if current_record is not None and current_record.status == Status.STARTED:
            current_record.stop()
        else:
            raise Exception(f"The topic '{topic}' does not have an active record.")

    def __mean(self, topic):
        records = self.records.get(topic, [])
        if len(records) > 0:
            total_sum = sum([r.total_time() for r in records])
            return total_sum / len(records)
        return 0

    def __max(self, topic):
        records = self.records.get(topic, [])
        return max([r.total_time() for r in records]) if len(records) > 0 else 0

    def __min(self, topic):
        records = self.records.get(topic, [])
        return min([r.total_time() for r in records]) if len(records) > 0 else 0

    def __plot(self, topic, color):
        records = self.records.get(topic, [])
        plt.hist(
            [r.total_time() for r in records],
            alpha=0.8,
            bins=30,
            color=color,
            label=topic,
        )

    def report(self, show=True):
        colors = ["b", "g", "r", "c", "m", "y", "k"]
        print("")
        for i, topic in enumerate(self.records):
            print(f"- {topic}:")
            print(f"  - number of records: {len(self.records[topic])}")
            print(f"  - mean: {self.__mean(topic)}")
            print(f"  - max: {self.__max(topic)}")
            print(f"  - min: {self.__min(topic)}")
            if show:
                self.__plot(topic, colors[i % len(colors)])
        if show:
            plt.legend()
            plt.show()
