import numpy as np
import pickle
import os
from haversine import haversine
from math import floor
from collections import defaultdict
import random
import gzip
from tqdm import tqdm 
import pandas as pd
import time
import multiprocessing
from multiprocessing import Pool

# dataset already been preprocessed
def parse(path):
    if 'gz' in path:
        f = gzip.open(path, 'rb')
        for l in f.readlines():
            yield(eval(l.decode('ascii')))
    else:
        f = open(path, 'rb')
        for l in f.readlines():
            yield(eval(l))

def process(line):
    return eval(line)

class dataInterpreter(object):
    def __init__(self, inputAtts, targetAtts=['derived_speed'], includeUser=True, includeSport=False, includeGender=False, includeTemporal=False, fn="endomondoHR_proper.json", scaleVals=True, trimmed_workout_len=450, scaleTargets="scaleVals", trainValidTestSplit=[.8,.1,.1], zMultiple=5, trainValidTestFN=None):
        self.filename = fn
        self.data_path = "./data"
        # path to raw data file and auxiliary on-disk indices
        self.original_data_path = os.path.join(self.data_path, self.filename)
        self.metaDataFn = fn.split(".")[0] + "_metaData.pkl"
        self.file_index_path = os.path.join(self.data_path, self.filename.split(".")[0] + "_file_index.pkl")

        self.scaleVals = scaleVals
        self.trimmed_workout_len = trimmed_workout_len
        if scaleTargets == "scaleVals":
            scaleTargets = scaleVals
        self.scale_targets = scaleTargets # set to false when scale only inputs
        self.smooth_window = 1 # window size = 1 means no smoothing
        self.perform_target_smoothing = True

        self.isNominal = ['gender', 'sport']
        self.isDerived = ['time_elapsed', 'distance', 'derived_speed', 'since_begin', 'since_last']
        self.isSequence = ['altitude', 'heart_rate', 'latitude', 'longitude'] + self.isDerived

        self.inputAtts = inputAtts
        self.includeUser = includeUser
        self.includeSport = includeSport
        self.includeGender = includeGender
        self.includeTemporal = includeTemporal

        self.targetAtts = ["tar_" + tAtt for tAtt in targetAtts]

        print("input attributes: ", self.inputAtts)
        print("target attributes: ", self.targetAtts)

        self.trainValidTestSplit = trainValidTestSplit
        self.trainValidTestFN = trainValidTestFN
        self.zMultiple = zMultiple

    def preprocess_data(self):
        """
        Lightweight preprocessing entrypoint.

        This version avoids loading the full dataset into memory. Instead it:
        1) loads the train/valid/test split,
        2) collects the set of workout ids actually used (including temporal context),
        3) streams over the raw data file to build summary statistics / encoders,
        4) builds a compact on-disk index mapping workout id -> file offset for fast access.
        """

        # load index for train/valid/test
        self.loadTrainValidTest()

        # collect all workout ids that will ever be touched
        self._collect_relevant_workouts()

        # build or load metadata (encoders + means/stds) in a streaming fashion
        self.buildMetaData()

        # build or load a small file index for random access during training
        self._ensure_file_index()

        # final input / output dimensions for the model
        self.input_dim = len(self.inputAtts)
        self.output_dim = len(self.targetAtts) # each continuous target has dimension 1, so total length = total dimension
      
    def map_workout_id(self):
        """
        Deprecated: in the original implementation this converted workout ids to
        integer indices into an in‑memory list of all workouts.

        The modern, memory‑efficient pipeline keeps workouts on disk and works
        directly with workout ids, so this method is now a no‑op kept only for
        backward compatibility.
        """
        return
    
    
    def load_meta(self): 
        self.buildMetaData() 

    def randomizeDataOrder(self, dataIndices):
        return np.random.permutation(dataIndices)

    
    def generateByIdx(self, index):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim
        
        # in the streaming version, `index` is assumed to be a workout id
        workoutid = index
        current_input = self._prepare_workout(workoutid)

        # use float32 to reduce memory footprint and match modern DL frameworks
        inputs = np.zeros([inputDataDim, self.trimmed_workout_len], dtype=np.float32)
        outputs = np.zeros([targetDataDim, self.trimmed_workout_len], dtype=np.float32)
        for idx, att in enumerate(inputAtts):
            if att == 'time_elapsed':
                inputs[idx, :] = np.ones([1, self.trimmed_workout_len]) * current_input[att][self.trimmed_workout_len-1] # given the total workout length
            else:
                inputs[idx, :] = current_input[att][:self.trimmed_workout_len]
        for att in targetAtts:
            outputs[0, :] = current_input[att][:self.trimmed_workout_len]
        inputs = np.transpose(inputs)
        outputs = np.transpose(outputs)

        # embedding inputs as int32 indices
        if self.includeUser:
            user_inputs = np.ones([self.trimmed_workout_len, 1], dtype=np.int32) * self.oneHotMap['userId'][current_input['userId']]
        if self.includeSport:
            sport_inputs = np.ones([self.trimmed_workout_len, 1], dtype=np.int32) * self.oneHotMap['sport'][current_input['sport']]
        if self.includeGender:
            gender_inputs = np.ones([self.trimmed_workout_len, 1], dtype=np.int32) * self.oneHotMap['gender'][current_input['gender']]

        # build context input    
        if self.includeTemporal and workoutid in self.contextMap:
            context_wid = self.contextMap[workoutid][2][-1] # id of previous workout
            context_input = self._prepare_workout(context_wid)

            context_since_last = np.ones([1, self.trimmed_workout_len], dtype=np.float32) * self.contextMap[workoutid][0]
            # consider what context?
            context_inputs = np.zeros([inputDataDim, self.trimmed_workout_len], dtype=np.float32)
            context_outputs = np.zeros([targetDataDim, self.trimmed_workout_len], dtype=np.float32)
            for idx, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    context_inputs[idx, :] = np.ones([1, self.trimmed_workout_len]) * context_input[att][self.trimmed_workout_len-1]
                else:
                    context_inputs[idx, :] = context_input[att][:self.trimmed_workout_len]
            for att in targetAtts:
                context_outputs[0, :] = context_input[att][:self.trimmed_workout_len]
            context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
            context_input_2 = np.transpose(context_outputs)

        inputs_dict = {'input':inputs}
        if self.includeUser:       
            inputs_dict['user_input'] = user_inputs
        if self.includeSport:       
            inputs_dict['sport_input'] = sport_inputs
        if self.includeGender:
            inputs_dict['gender_input'] = gender_inputs
        if self.includeTemporal:
            inputs_dict['context_input_1'] = context_input_1
            inputs_dict['context_input_2'] = context_input_2

        return (inputs_dict, outputs, workoutid)
    
    # yield input and target data
    def dataIteratorSupervised(self, trainValidTest):
        targetAtts = self.targetAtts
        inputAtts = self.inputAtts

        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        # run on train, valid or test?
        if trainValidTest == 'train':
            indices = self.trainingSet
        elif trainValidTest == 'valid':
            indices = self.validationSet
        elif trainValidTest == 'test':
            indices = self.testSet
        else:
            raise (Exception("invalid dataset type: must be 'train', 'valid', or 'test'"))

        # loop each data point; in the streaming version `indices` are workout ids
        for workoutid in indices:
            current_input = self._prepare_workout(workoutid)
 
            # use float32 for continuous inputs / targets
            inputs = np.zeros([inputDataDim, self.trimmed_workout_len], dtype=np.float32)
            outputs = np.zeros([targetDataDim, self.trimmed_workout_len], dtype=np.float32)
            for i, att in enumerate(inputAtts):
                if att == 'time_elapsed':
                    inputs[i, :] = np.ones([1, self.trimmed_workout_len]) * current_input[att][self.trimmed_workout_len-1] # given the total workout length
                else:
                    inputs[i, :] = current_input[att][:self.trimmed_workout_len]
            for att in targetAtts:
                outputs[0, :] = current_input[att][:self.trimmed_workout_len]
            inputs = np.transpose(inputs)
            outputs = np.transpose(outputs)

            # embedding inputs as int32 indices
            if self.includeUser:
                user_inputs = np.ones([self.trimmed_workout_len, 1], dtype=np.int32) * self.oneHotMap['userId'][current_input['userId']]
            if self.includeSport:
                sport_inputs = np.ones([self.trimmed_workout_len, 1], dtype=np.int32) * self.oneHotMap['sport'][current_input['sport']]
            if self.includeGender:
                gender_inputs = np.ones([self.trimmed_workout_len, 1], dtype=np.int32) * self.oneHotMap['gender'][current_input['gender']]
   
            # build context input    
            if self.includeTemporal and workoutid in self.contextMap:
                context_wid = self.contextMap[workoutid][2][-1] # id of previous workout
                context_input = self._prepare_workout(context_wid)

                context_since_last = np.ones([1, self.trimmed_workout_len], dtype=np.float32) * self.contextMap[workoutid][0]
                # consider what context?
                context_inputs = np.zeros([inputDataDim, self.trimmed_workout_len], dtype=np.float32)
                context_outputs = np.zeros([targetDataDim, self.trimmed_workout_len], dtype=np.float32)
                for i, att in enumerate(inputAtts):
                    if att == 'time_elapsed':
                        context_inputs[i, :] = np.ones([1, self.trimmed_workout_len]) * context_input[att][self.trimmed_workout_len-1]
                    else:
                        context_inputs[i, :] = context_input[att][:self.trimmed_workout_len]
                for att in targetAtts:
                    context_outputs[0, :] = context_input[att][:self.trimmed_workout_len]
                context_input_1 = np.transpose(np.concatenate([context_inputs, context_since_last], axis=0))
                context_input_2 = np.transpose(context_outputs)
            
            inputs_dict = {'input':inputs}
            if self.includeUser:       
                inputs_dict['user_input'] = user_inputs
            if self.includeSport:       
                inputs_dict['sport_input'] = sport_inputs
            if self.includeGender:
                inputs_dict['gender_input'] = gender_inputs
            if self.includeTemporal:
                inputs_dict['context_input_1'] = context_input_1
                inputs_dict['context_input_2'] = context_input_2
                
            yield (inputs_dict, outputs, workoutid)


    # feed into Keras' fit_generator (automatically resets)
    def generator_for_autotrain(self, batch_size, num_steps, trainValidTest):
        print("batch size = {}, num steps = {}".format(batch_size, num_steps))
        print("start new generator epoch: " + trainValidTest)

        # get the batch generator based on mode: train/valid/test
        if trainValidTest=="train":
            data_len = len(self.trainingSet)
        elif trainValidTest=="valid":
            data_len = len(self.validationSet)
        elif trainValidTest=="test":
            data_len = len(self.testSet)
        else:
            raise(ValueError("trainValidTest is not a valid value"))
        batchGen = self.dataIteratorSupervised(trainValidTest)
        epoch_size = int(data_len / batch_size)
        inputDataDim = self.input_dim
        targetDataDim = self.output_dim

        if epoch_size == 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")
            
        for i in range(epoch_size):
            # float32 continuous tensors + integer ids for embeddings
            inputs = np.zeros([batch_size, num_steps, inputDataDim], dtype=np.float32)
            outputs = np.zeros([batch_size, num_steps, targetDataDim], dtype=np.float32)
            workoutids = np.zeros([batch_size], dtype=np.int64)

            if self.includeUser:
                user_inputs = np.zeros([batch_size, num_steps, 1], dtype=np.int32)
            if self.includeSport:
                sport_inputs = np.zeros([batch_size, num_steps, 1], dtype=np.int32)
            if self.includeGender:
                gender_inputs = np.zeros([batch_size, num_steps, 1], dtype=np.int32)
            if self.includeTemporal:
                context_input_1 = np.zeros([batch_size, num_steps, inputDataDim + 1], dtype=np.float32)
                context_input_2 = np.zeros([batch_size, num_steps, targetDataDim], dtype=np.float32)

            # inputs_dict = {'input':inputs}
            inputs_dict = {'input':inputs, 'workoutid':workoutids}
            for j in range(batch_size):
                current = next(batchGen)
                inputs[j,:,:] = current[0]['input']
                outputs[j,:,:] = current[1]
                workoutids[j] = current[2]

                if self.includeUser:
                    user_inputs[j,:,:] = current[0]['user_input']
                    inputs_dict['user_input'] = user_inputs
                if self.includeSport:
                    sport_inputs[j,:,:] = current[0]['sport_input']
                    inputs_dict['sport_input'] = sport_inputs
                if self.includeGender:
                    gender_inputs[j,:,:] = current[0]['gender_input']
                    inputs_dict['gender_input'] = gender_inputs
                if self.includeTemporal:
                    context_input_1[j,:,:] = current[0]['context_input_1']
                    context_input_2[j,:,:] = current[0]['context_input_2']
                    inputs_dict['context_input_1'] = context_input_1
                    inputs_dict['context_input_2'] = context_input_2
            # yield one batch
            yield (inputs_dict, outputs)

    def loadTrainValidTest(self):
        with open(self.trainValidTestFN, "rb") as f:
            self.trainingSet, self.validationSet, self.testSet, self.contextMap = pickle.load(f)
            print("train/valid/test set size = {}/{}/{}".format(len(self.trainingSet), len(self.validationSet), len(self.testSet)))
            print("dataset split loaded")       

    def _collect_relevant_workouts(self):
        """
        Build a set of all workout ids that will ever be accessed during
        training, validation, testing, or temporal context lookup.
        """
        all_ids = set()
        all_ids.update(self.trainingSet)
        all_ids.update(self.validationSet)
        all_ids.update(self.testSet)

        for wid, ctx in self.contextMap.items():
            all_ids.add(wid)
            # ctx is (since_last, since_begin, [prev_wids])
            for prev_wid in ctx[2]:
                all_ids.add(prev_wid)

        self.all_workout_ids = all_ids
        print("Total unique workouts used (train/valid/test + context): {}".format(len(self.all_workout_ids)))

    # derive 'time_elapsed', 'distance', 'new_workout', 'derived_speed'
    def deriveData(self, att, currentDataPoint, wid):
        if att == 'time_elapsed':
            # Derive the time elapsed from the start
            timestamps = currentDataPoint['timestamp']
            initialTime = timestamps[0]
            return [x - initialTime for x in timestamps]
        elif att == 'distance':
            # Derive the distance
            lats = currentDataPoint['latitude']
            longs = currentDataPoint['longitude']
            indices = range(1, len(lats)) 
            distances = [0]
            # Gets distance traveled since last time point in kilometers
            distances.extend([haversine([lats[i-1],longs[i-1]], [lats[i],longs[i]]) for i in indices]) 
            return distances
        # derive the new_workout list
        elif att == 'new_workout': 
            workoutLength = self.trimmed_workout_len
            newWorkout = np.zeros(workoutLength)
            # Add the signal at start
            newWorkout[0] = 1 
            return newWorkout
        elif att == 'derived_speed':
            distances = self.deriveData('distance', currentDataPoint, wid)
            timestamps = currentDataPoint['timestamp']
            indices = range(1, len(timestamps))
            times = [0]
            times.extend([timestamps[i] - timestamps[i-1] for i in indices])
            derivedSpeeds = [0]
            for i in indices:
                try:
                    curr_speed = 3600 * distances[i] / times[i]
                    derivedSpeeds.append(curr_speed)
                except:
                    derivedSpeeds.append(derivedSpeeds[i-1])
            return derivedSpeeds
        elif att == 'since_last':
            if wid in self.contextMap:
                total_time = self.contextMap[wid][0]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len, dtype=np.float32) * total_time
        elif att == 'since_begin':
            if wid in self.contextMap:
                total_time = self.contextMap[wid][1]
            else:
                total_time = 0
            return np.ones(self.trimmed_workout_len, dtype=np.float32) * total_time
        else:
            raise(Exception("No such derived data attribute"))

        
    # computing z-scores and multiplying them based on a scaling paramater
    # produces zero-centered data, which is important for the drop-in procedure
    def scaleData(self, data, att, zMultiple=2):
        mean, std = self.variableMeans[att], self.variableStds[att]
        diff = [d - mean for d in data]
        zScore = [d / std for d in diff] 
        return [x * zMultiple for x in zScore]

    # perform fixed-window median smoothing on a sequence
    def median_smoothing(self, seq, context_size):
        # seq is a list
        if context_size == 1: # if the window is 1, no smoothing should be applied
            return seq
        seq_len = len(seq)
        if context_size % 2 == 0:
            raise Exception("Context size must be odd for median smoothing")

        smoothed_seq = []
        # loop through sequence and smooth each position
        for i in range(seq_len): 
            cont_diff = (context_size - 1) / 2
            context_min = int(max(0, i-cont_diff))
            context_max = int(min(seq_len, i+cont_diff))
            median_val = np.median(seq[context_min:context_max])
            smoothed_seq.append(median_val)

        return smoothed_seq
    
    def buildEncoder(self, classLabels):
        # Constructs a dictionary that maps each class label to a list 
        # where one entry in the list is 1 and the remainder are 0
        encodingLength = len(classLabels)
        encoder = {}
        mapper = {}
        for i, label in enumerate(classLabels):
            encoding = [0] * encodingLength
            encoding[i] = 1
            encoder[label] = encoding
            mapper[label] = i
        return encoder, mapper
    
    
    def writeSummaryFile(self):
        metaDataForWriting=metaDataEndomondo(self.numDataPoints, self.encodingLengths, self.oneHotEncoders,  
                                             self.oneHotMap, self.isSequence, self.isNominal, self.isDerived, 
                                             self.variableMeans, self.variableStds)
        with open(self.metaDataFn, "wb") as f:
            pickle.dump(metaDataForWriting, f)
        print("Summary file written")
        
    def loadSummaryFile(self):
        try:
            print("Loading metadata")
            with open(self.metaDataFn, "rb") as f:
                metaData = pickle.load(f)
        except:
            raise(IOError("Metadata file: " + self.metaDataFn + " not in valid pickle format"))
        self.numDataPoints = metaData.numDataPoints
        self.encodingLengths = metaData.encodingLengths
        self.oneHotEncoders = metaData.oneHotEncoders
        self.oneHotMap = metaData.oneHotMap
        self.isSequence = metaData.isSequence 
        self.isNominal = metaData.isNominal
        self.variableMeans = metaData.variableMeans
        self.variableStds = metaData.variableStds
        print("Metadata loaded")

    def _build_file_index(self):
        """
        Build a compact mapping from workout id -> byte offset in the raw data
        file so that we can lazily load individual workouts on demand.
        """
        print("Building file index (id -> offset)")
        file_index = {}
        with open(self.original_data_path, 'r') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                try:
                    currData = eval(line_stripped)
                except Exception:
                    continue

                wid = currData.get('id')
                if wid in getattr(self, "all_workout_ids", set()):
                    file_index[wid] = offset

        with open(self.file_index_path, "wb") as f:
            pickle.dump(file_index, f)
        self.file_index = file_index
        print("File index built for {} workouts".format(len(self.file_index)))

    def _load_file_index(self):
        with open(self.file_index_path, "rb") as f:
            self.file_index = pickle.load(f)
        print("File index loaded for {} workouts".format(len(self.file_index)))

    def _ensure_file_index(self):
        """
        Ensure that `self.file_index` is populated, building it on first use.
        """
        if hasattr(self, "file_index") and getattr(self, "file_index", None):
            return
        if os.path.isfile(self.file_index_path):
            self._load_file_index()
        else:
            self._build_file_index()

    def _load_workout_by_id(self, wid):
        """
        Lazily load a single workout by its id using the file index.
        """
        self._ensure_file_index()
        if wid not in self.file_index:
            raise KeyError("Workout id {} not found in file index".format(wid))
        offset = self.file_index[wid]
        with open(self.original_data_path, 'r') as f:
            f.seek(offset)
            line = f.readline()
        currData = eval(line.strip())
        return currData

    def _prepare_workout(self, wid):
        """
        Load, derive and (optionally) scale all attributes needed for a single
        workout. This is invoked lazily by the data iterator / generators.
        """
        currentDataPoint = self._load_workout_by_id(wid)

        # derive attributes needed for inputs/targets
        for att in self.isDerived:
            if att not in currentDataPoint:
                currentDataPoint[att] = self.deriveData(att, currentDataPoint, wid)

        # scale continuous input attributes
        if self.scaleVals:
            for att in self.isSequence:
                in_data = currentDataPoint[att]
                currentDataPoint[att] = np.array(self.scaleData(in_data, att, self.zMultiple), dtype=np.float32)
        else:
            for att in self.isSequence:
                currentDataPoint[att] = np.array(currentDataPoint[att], dtype=np.float32)

        # build scaled targets
        targetAtts = ['heart_rate', 'derived_speed']
        for tAtt in targetAtts:
            if tAtt not in currentDataPoint:
                continue
            if self.perform_target_smoothing:
                tar_data = self.median_smoothing(currentDataPoint[tAtt], self.smooth_window)
            else:
                tar_data = currentDataPoint[tAtt]
            if self.scale_targets:
                tar_data = self.scaleData(tar_data, tAtt, self.zMultiple)
            currentDataPoint["tar_" + tAtt] = np.array(tar_data, dtype=np.float32)

        return currentDataPoint

        
    def derive_data(self):
        """
        Deprecated: the old implementation eagerly materialised all derived
        attributes into an in‑memory list of workouts. We now derive features
        on the fly in `_prepare_workout` to keep memory usage low.
        """
        return

    # Generate meta information about data
    def buildMetaData(self):
        """
        Build or load metadata (encoders, means, stds) in a streaming manner,
        without loading the entire dataset into memory.
        """
        if os.path.isfile(self.metaDataFn):
            self.loadSummaryFile()
            return

        print("Building data schema (streaming)")
        print("is sequence: {}".format(self.isSequence))

        # sum of variables for mean computation
        variableSums = defaultdict(float)

        # number of categories for each categorical variable
        classLabels = defaultdict(set)

        # first pass: build class label sets and variable sums
        with open(self.original_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    currData = eval(line)
                except Exception:
                    continue

                wid = currData.get('id')
                if wid not in getattr(self, "all_workout_ids", set()):
                    continue

                # update number of users
                user = currData['userId']
                classLabels['userId'].add(user)

                # update categorical attributes
                for att in self.isNominal:
                    val = currData.get(att, None)
                    if val is not None:
                        classLabels[att].add(val)

                # base continuous attributes
                for att in ['altitude', 'heart_rate', 'latitude', 'longitude']:
                    if att in currData:
                        variableSums[att] += float(np.sum(currData[att]))

                # derived continuous attributes
                for att in self.isDerived:
                    derived_seq = self.deriveData(att, currData, wid)
                    variableSums[att] += float(np.sum(derived_seq))

        # build encoders
        oneHotEncoders = {}
        oneHotMap = {}
        encodingLengths = {}
        for att in self.isNominal:
            oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att])
            encodingLengths[att] = len(classLabels[att])

        att = 'userId'
        oneHotEncoders[att], oneHotMap[att] = self.buildEncoder(classLabels[att])
        encodingLengths[att] = 1

        for att in self.isSequence:
            encodingLengths[att] = 1

        # summary information
        self.numDataPoints = len(getattr(self, "all_workout_ids", []))

        # normalize continuous: altitude, heart_rate, latitude, longitude and all derives
        self.computeMeanStd(variableSums, self.numDataPoints, self.isSequence)

        self.oneHotEncoders = oneHotEncoders
        self.oneHotMap = oneHotMap
        self.encodingLengths = encodingLengths
        # Save that summary file so that it can be used next time
        self.writeSummaryFile()

 
    def computeMeanStd(self, varSums, numDataPoints, attributes):
        """
        Second streaming pass to compute standard deviations based on the
        variable sums from the first pass.
        """
        print("Computing variable means and standard deviations (streaming)")

        # assume each data point has 500 time steps, as in the original code
        numSequencePoints = max(1, numDataPoints * 500)

        variableMeans = {}
        for att in varSums:
            variableMeans[att] = varSums[att] / numSequencePoints

        varResidualSums = defaultdict(float)

        with open(self.original_data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    currData = eval(line)
                except Exception:
                    continue

                wid = currData.get('id')
                if wid not in getattr(self, "all_workout_ids", set()):
                    continue

                # loop each continuous attribute
                for att in attributes:
                    if att in ['altitude', 'heart_rate', 'latitude', 'longitude']:
                        dataPointArray = np.array(currData[att], dtype=np.float32)
                    else:
                        dataPointArray = np.array(self.deriveData(att, currData, wid), dtype=np.float32)

                    diff = np.subtract(dataPointArray, variableMeans[att])
                    sq = np.square(diff)
                    varResidualSums[att] += float(np.sum(sq))

        variableStds = {}
        for att in varResidualSums:
            variableStds[att] = np.sqrt(varResidualSums[att] / numSequencePoints)

        self.variableMeans = variableMeans
        self.variableStds = variableStds
        
        
    # scale continuous data
    def scale_data(self, scaling=True):
        """
        Deprecated: scaling is now done on the fly in `_prepare_workout`.
        This method is kept only for backward compatibility.
        """
        return


class metaDataEndomondo(object):
    def __init__(self, numDataPoints, encodingLengths, oneHotEncoders, oneHotMap, isSequence, isNominal, isDerived,
                 variableMeans, variableStds):
        self.numDataPoints = numDataPoints
        self.encodingLengths = encodingLengths
        self.oneHotEncoders = oneHotEncoders
        self.oneHotMap = oneHotMap
        self.isSequence = isSequence
        self.isNominal = isNominal
        self.isDerived = isDerived
        self.variableMeans = variableMeans
        self.variableStds = variableStds


if __name__ == "__main__":

    data_path = "endomondoHR_proper.json"
    attrFeatures = ['userId', 'sport', 'gender']
    trainValidTestSplit = [0.8, 0.1, 0.1]
    targetAtts = ["derived_speed"]
    inputAtts = ["distance", "altitude", "time_elapsed"]
    endo_reader = dataInterpreter(inputAtts)
    endo_reader.preprocess_data()

