import struct
import time

import pyaudio

import analyzer


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

print "Starting PyAudio input"
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

classifier_filename = "classifier.xml"
classifier = analyzer.SavedNeuralNetworkClassifier(classifier_filename)

realtime_analyzer = analyzer.RealtimeAnalyzer(RATE, classifier)


def timestamp():
    return float(time.time() + 0.5)

print timestamp()
print "Now recording"
i = 0
start_time = timestamp()
while True:
    i += 1
    data = stream.read(CHUNK)
    data = struct.unpack("%dh" % CHUNK, data)
    realtime_analyzer.push(data)
    if i >= 100:
        rate = CHUNK * i / (timestamp() - start_time)
        start_time = timestamp()
        print (rate)
        i = 0
