import math
import inspect

import numpy as np
import matplotlib.pyplot as plt
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from scipy.io import wavfile
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader

EPSILON = np.finfo(np.float).eps
FRAME_TIME_LENGTH = 180  # length of frame in milliseconds
# DIVISIONS = np.array([40, 70, 110, 150, 200, 250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000, 11025])
# DIVISIONS = np.array([500, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7000, 10000])
DIVISIONS = np.array([500, 1000, 2500, 5000, 7000])
MOVING_AVERAGE_LENGTH = 3  # length in number of FFT intervals
MOVING_THRESHOLD_LENGTH = 70
NETWORK_LEARNING_RATE = 0.3
NETWORK_MOMENTUM = 0.1
NETWORK_HIDDEN_NEURONS = 20
NETWORK_ITERATIONS = 50


class AudioBuffer:
    def __init__(self, fft_sample_length, overlap_sample_length):
        self.data = []
        self.fft_sample_length = fft_sample_length
        self.overlap_sample_length = overlap_sample_length
        self.step = fft_sample_length - overlap_sample_length

    def push(self, samples):
        """
        Adds samples to end of buffer data.
        :param samples:
        """
        self.data.extend(samples)

    def available(self):
        return len(self.data) >= self.fft_sample_length

    def read(self):
        output = self.data[:self.fft_sample_length]
        self.data = self.data[self.step:]
        return output


class DataBuffer:
    def __init__(self, length=float("inf")):
        self.length = length
        self.data = []

    def push(self, item):
        self.data.append(item)
        self._trim()

    def push_multiple(self, items):
        self.data.extend(items)
        self._trim()

    def _trim(self):
        length = len(self.data)
        if length > self.length:
            self.data = self.data[length - self.length:]


class Classifier(object):  # interface for a generic classifier
    def __init__(self):
        pass

    def train(self, data):
        pass

    def run(self, feature_vector):
        pass


class NeuralNetworkClassifier(Classifier):
    def __init__(self, n_inputs, n_outputs, n_hidden=NETWORK_HIDDEN_NEURONS):
        super(NeuralNetworkClassifier, self).__init__()
        self.network = buildNetwork(n_inputs, n_hidden, n_hidden, n_hidden, n_outputs)
        self.dataset = SupervisedDataSet(n_inputs, n_outputs)

    def train(self, data, iterations=NETWORK_ITERATIONS):
        for item in data:
            self.dataset.addSample(item[0], item[1])
        trainer = BackpropTrainer(self.network, self.dataset, learningrate=NETWORK_LEARNING_RATE,
                                  momentum=NETWORK_MOMENTUM)
        error = 0
        for i in xrange(iterations):
            error = trainer.train()
            print (i + 1), error
        return error

    def run(self, feature_vector):
        return self.network.activate(feature_vector)

    def export(self, filename):
        NetworkWriter.writeToFile(self.network, filename)


class SavedNeuralNetworkClassifier(Classifier):
    def __init__(self, filename):
        super(SavedNeuralNetworkClassifier, self).__init__()
        self.network = NetworkReader.readFrom(filename)

    def run(self, feature_vector):
        return self.network.activate(feature_vector)


class FeatureVectorBuffer(DataBuffer):
    def __init__(self, length=float("inf")):
        DataBuffer.__init__(self, length)
        self.results = DataBuffer(length)

    def add_vector(self, feature_vector):
        DataBuffer.push(self, feature_vector)
        result = self.classify(feature_vector)
        self.results.push(result)

    def classify(self, feature_vector):
        pass


class FeatureVectorExtractor:
    def __init__(self, rate):
        self.rate = rate

        calculator = FFTSizeCalculator(rate)
        self.fft_sample_length = calculator.fft_sample_length
        self.overlap_sample_length = calculator.overlap_sample_length

        self.audio_buffer = AudioBuffer(fft_sample_length=self.fft_sample_length,
                                        overlap_sample_length=self.overlap_sample_length)
        # self.buffers = {
        #     "raw_slices": DataBuffer(),
        #     "slices": DataBuffer(),
        #     "zero_crossing_rates": DataBuffer(),
        #     "rolloff_freqs": DataBuffer(),
        #     "slices_bins": DataBuffer()
        # }
        self.buffers = {name: DataBuffer() for name in
                        ["raw_slices", "slices", "zero_crossing_rates", "rolloff_freqs", "slices_bins",
                         "third_octave", "averages", "thresholds", "ratios", "magnitude", "stddev"]}

        self.classifier = FeatureVectorBuffer()
        self.fft = FFT(self.rate)
        self.original_freqs = self.fft.freqs
        self.freqs = self.high_pass_filter_freqs(self.original_freqs, 500)
        self.bin_divisions_indexes = self.find_indexes(self.freqs, DIVISIONS)

    def plot_spectrogram(self, bins, freqs, slices, logscale=False, axes=plt):
        power = slices.T
        if logscale:
            z = np.log10(power)
        else:
            z = power
        axes.pcolormesh(bins, freqs, z)

    def find_indexes(self, freqs, divisions):
        # Determine where the divisions are in the freqs list

        indexes = []
        i = 0
        for div in divisions:
            while i < len(freqs) and freqs[i] < div:
                i += 1
            indexes.append(i)

        return indexes

    def freq_bins(self, slice):
        # Divide slice into frequency bins, returns new slice

        indexes = self.bin_divisions_indexes

        output = []
        prev_index = indexes[0]
        for i in xrange(1, len(indexes)):
            part = slice[prev_index:indexes[i] + 1]
            average = sum(part) / len(part)
            output.append(average)
            prev_index = indexes[i]

        output = np.array(output)

        return output

    def slice_rolloff_freq(self, slice, threshold=0.9):
        target = threshold * sum(slice)
        partial = 0.0
        i = 0
        length = len(slice)
        while partial < target and i < length - 1:
            partial += slice[i]
            i += 1
        return i

    def avg_zero_crossing_rate(self, sound_data):
        signs = np.sign(np.array(sound_data))
        total = 0
        for i in xrange(1, len(signs)):
            if signs[i - 1] != signs[i]:
                total += 1
        rate = float(total) / len(sound_data)
        return rate

    def normalize(self, slice):
        raw_slices = np.array(self.buffers["raw_slices"].data)
        end = len(raw_slices)

        # Take the moving average to smooth out the data
        start = max(0, end - MOVING_AVERAGE_LENGTH)
        actual_length = end - start
        average = sum(raw_slices[start:end]) / actual_length
        self.buffers["averages"].push(average)

        # Find the sliding minimum value in each frequency band as threshold
        averages = self.buffers["averages"].data
        start2 = max(0, len(self.buffers["averages"].data) - MOVING_THRESHOLD_LENGTH)
        possible_thresholds = np.array(averages[start2:]).T
        threshold = []
        for band in possible_thresholds:
            threshold.append(np.amin(band))

        new_slices = slice - threshold  # normalize
        new_slices = new_slices.clip(0)  # clip at threshold
        new_slices /= 10  # scale downwards
        return new_slices, threshold, average

    def high_pass_filter(self, slice, freqs, cutoff_frequency):
        """
        Zeros the frequencies below the specified frequency
        (or the next lowest present)
        and returns the remaining higher frequencies (data and labels)
        :param slices:
        """
        # Find the index to cut off at
        index = self.find_indexes(freqs, [cutoff_frequency])[0]

        # Perform the filtering
        new_slice = slice[index:]

        return new_slice

    def high_pass_filter_freqs(self, freqs, cutoff_frequency):
        index = self.find_indexes(freqs, [cutoff_frequency])[0]
        new_freqs = freqs[index:]
        new_freqs = np.array(new_freqs)
        return new_freqs

    def pairwise_differences(self, items):
        length = len(items)
        ratios = []
        for i in xrange(length):
            for j in xrange(i + 1, length):
                ratios.append(items[i] - items[j])
        return ratios

    def autocorrelation_coefficient(self, series):
        series1 = series - np.average(series)
        series2 = series1[::-1]
        corr = np.correlate(np.abs(series), np.abs(series2))
        return float(corr) / max(np.var(series), EPSILON) / 100

    def analyze(self, data):
        raw_slice = self.fft.run(data)

        # Decibel scale
        raw_slice = 10 * np.log10(raw_slice) + 60
        raw_slice = raw_slice.clip(0)

        # High-pass filter
        raw_slice = self.high_pass_filter(raw_slice, self.original_freqs, 500)

        # Add raw slices to buffer for use in calculating moving average
        self.buffers["raw_slices"].push(raw_slice)

        # Normalize the slices for analysis purposes
        slice, threshold, average = self.normalize(raw_slice)
        self.buffers["slices"].push(slice)
        self.buffers["thresholds"].push(threshold)
        slices = [slice]

        # Calculate zero-crossing rate
        zero_crossing_rate = self.avg_zero_crossing_rate(data)
        self.buffers["zero_crossing_rates"].push(zero_crossing_rate)

        # Calculate rolloff frequencies
        rolloff_freq = self.freqs[self.slice_rolloff_freq(slice)]
        rolloff_freq /= np.amax(self.freqs)  # make a proportion of the maximum frequency
        self.buffers["rolloff_freqs"].push(rolloff_freq)

        # Divide each slice into frequency bins
        slice_bins = self.freq_bins(slice)
        self.buffers["slices_bins"].push(slice_bins)

        # Extract the third octave
        third_octave_indexes = self.find_indexes(self.freqs, [700, 1300])
        third_octave = slice[third_octave_indexes[0]:third_octave_indexes[1]]
        self.buffers["third_octave"].push(third_octave)

        # Third octave autocorrelation
        # third_octave_autocorrelation = self.autocorrelation_coefficient(slice)
        # self.buffers["third_octave_autocorrelation"].push(third_octave_autocorrelation)

        # Pairwise differences (ratio of magnitude) between frequency bins
        ratios = self.pairwise_differences(slice_bins)
        self.buffers["ratios"].push(ratios)

        # Overall magnitude of sound
        magnitude = np.average(slice)
        self.buffers["magnitude"].push(magnitude)

        # Standard deviation of frequency spectrum
        stddev = np.std(slice)
        self.buffers["stddev"].push(stddev)

        # Create feature vectors
        vector = []
        # vector.extend(slice_bins)
        vector.extend(ratios)
        vector.append(zero_crossing_rate)
        # vector.append(third_octave_autocorrelation)
        vector.append(stddev)
        vector.append(rolloff_freq)
        vector.append(magnitude)
        vector = np.array(vector)
        self.process_vector(vector)

        # Return vector
        return vector

    def _raw_data_in_slices(self, data):
        num = int((len(data) - self.fft_sample_length) / self._step_length()) + 1
        prev_index = 0
        for i in xrange(num):
            section = data[prev_index:prev_index + self.fft_sample_length]
            prev_index += self._step_length()
            yield section

    def push(self, samples):
        self.audio_buffer.push(samples)
        vectors = []
        while self.audio_buffer.available():
            vector = self.analyze(self.audio_buffer.read())
            vectors.append(vector)
        return vectors

    def display(self, plot_filename=None, buffer_list=None):
        if buffer_list is None:
            length = len(self.buffers)
        else:
            length = len(buffer_list)
        fig, axes = plt.subplots(length)
        i = 0
        for name in self.buffers.keys():
            if buffer_list is None or name in buffer_list:
                print (name)
                try:
                    axis = axes[i]
                except TypeError:
                    axis = axes
                self._display_buffer(self.buffers[name], axis)
                i += 1
                # self._display_buffer(self.classifier, axes[-1])  # Display feature vector
        if plot_filename is not None:
            plt.savefig(plot_filename)
        plt.show()

    def _display_buffer(self, buffer, axis):
        buffer_data = buffer.data
        if type(buffer_data[0]) is np.ndarray:
            # print as spectrogram
            # shifted_buffer_data = np.array(buffer_data) - np.amin(buffer_data)
            # shifted_buffer_data = shifted_buffer_data.clip(EPSILON)
            shifted_buffer_data = np.array(buffer_data[1:])
            self.plot_spectrogram(np.array(range(len(buffer_data) - 1)), np.array(range(len(buffer_data[0]))),
                                  shifted_buffer_data, axes=axis)
        else:
            # plot as standard (x,y)
            axis.plot(range(len(buffer_data) - 1), buffer_data[1:])

    def process_vector(self, vector):
        # print vector
        self.classifier.add_vector(vector)


class FFT:
    # FFT algorithm based on code from matplotlib
    # Code simplified for use as a single real-valued FFT
    # Used under a BSD compatible license
    # Copyright (c) 2002-2009 John D. Hunter; All Rights Reserved

    def __init__(self, rate):
        self.rate = rate

        calculator = FFTSizeCalculator(rate)
        self.fft_sample_length = calculator.fft_sample_length
        self.overlap_sample_length = calculator.overlap_sample_length
        self.step = calculator.step

        self.numFreqs = self.fft_sample_length / 2 + 1
        self.windowVals = np.hanning(self.fft_sample_length)
        self.freqs = float(self.rate) / self.fft_sample_length * np.arange(self.numFreqs)

    def run(self, x):
        assert len(x) == self.fft_sample_length

        windowed_x = x * self.windowVals
        fx = np.fft.rfft(windowed_x)

        # Get square of magnitude of complex vector
        fx = fx.real ** 2 + fx.imag ** 2

        # Scaling and normalizing output
        fx /= (np.abs(self.windowVals)**2).sum()
        fx[1:-1] *= 2
        fx /= self.rate

        fx = fx.real

        return fx


class FFTSizeCalculator:
    def __init__(self, rate):
        self.rate = rate
        frame_samples_length = int(float(FRAME_TIME_LENGTH) / float(1000) * float(self.rate))
        self.fft_sample_length = int(2 ** self._nextpow2(frame_samples_length))
        self.overlap_sample_length = int(0.3 * frame_samples_length)
        self.step = self.fft_sample_length - self.overlap_sample_length

    def _nextpow2(self, num):
        return int(np.ceil(np.log2(num)))


class RealtimeAnalyzer:
    def __init__(self, rate, classifier):
        """

        :param rate: int
        :param classifier: Classifier
        """
        self.classifier = classifier
        self.extractor = FeatureVectorExtractor(rate)
        self.buffer = DataBuffer(100)
        # self.buffer.push_multiple(np.zeros(100))
        # plt.ion()
        # self.line, = plt.plot(xrange(100), xrange(100))
        # plt.draw()

    def push(self, samples):
        feature_vectors = self.extractor.push(samples)
        if feature_vectors is not None:
            for vector in feature_vectors:
                result = self.classifier.run(vector)
                self._output(result)
            # self._plot()

    def _output(self, result):
        output = 0
        for item in result:
            output += item
        output /= float(2)
        if not math.isnan(output):
            scale = 20
            value = min(max(int(output * scale), 0), scale)
            self.buffer.push(value)
            # sys.stdout.flush()
            print ("[{0}{1}] {2}".format('#' * value, ' ' * (scale - value), output))

    def _plot(self):
        self.line.set_ydata(self.buffer.data)
        plt.draw()


VIRTUAL_BUFFER_SIZE = 1000


class FileProcessor(object):
    def _process_file(self, filename, display=False, **kargs):
        rate, data = wavfile.read(filename)
        extractor = FeatureVectorExtractor(rate)
        feature_vectors = extractor.push(data)
        if display:
            extractor.display(**kargs)
        return feature_vectors


class BatchFileTrainer(FileProcessor):
    def __init__(self, classifier):
        self.classifier = classifier
        self.data = []

    def add(self, filename, results):
        feature_vectors = self._process_file(filename)

        # Create a stretched results list the same length as feature_vectors
        target_length = len(feature_vectors)
        source_length = len(results)
        results_stretched = []
        for i in xrange(target_length):
            results_stretched.append(results[int(float(i) / target_length * source_length)])

        data = []
        for i in xrange(1, target_length):  # Remove the first vector, which usually has problems
            item = [feature_vectors[i], results_stretched[i]]
            print (item)
            data.append(item)

        print ("# of feature vectors:", target_length - 1)

        self.data.extend(data)

    def train(self):
        # Create a new classifier if necessary
        if inspect.isclass(self.classifier):
            self.classifier = self.classifier(len(self.data[0][0]), len(self.data[0][1]))
        return self.classifier.train(self.data)


class FileAnalyzer(FileProcessor):
    def __init__(self, classifier):
        self.classifier = classifier

    def analyze(self, filename, save_filename=None, **kargs):
        vectors = self._process_file(filename, **kargs)
        results = []
        text = ""
        for vector in vectors:
            print (vector)
            result = self.classifier.run(vector)
            text += ",".join([str(item) for item in vector]) + "," + ",".join([str(item) for item in result]) + "\n"
            results.append(result)
        if save_filename is not None:
            with open(save_filename, "w") as f:
                f.write(text)
        return results