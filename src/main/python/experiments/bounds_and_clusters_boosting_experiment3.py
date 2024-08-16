from datetime import datetime
import os
import pickle
from typing import Any
import warnings
from joblib.parallel import delayed
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib
from sklweka.dataset import load_arff, to_nominal_labels
from sklweka.classifiers import WekaEstimator
import sklweka.jvm as jvm
from sklearn.utils import shuffle
import logging
import logging.config
import logging.handlers
from progressparallel import ProgressParallel

import settings
import json
from tqdm import tqdm
import javabridge
import settings

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold

import multiprocessing as mp
import time
import random
import queue as qu

from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon, mannwhitneyu, rankdata
from stats_tools import p_val_matrix_to_vec, p_val_vec_to_matrix, reorder_names, chi_homogenity
import string
from timeit import default_timer as timer
from ptranks.ranks.wra import wra
import pymannkendall as mk
import traceback
import threading as th
import psutil
from results_storage.results_storage import ResultsStorage
from enum import Enum
import xarray as xr

CLASSIFIERS_CONFIG_FILE_NAME_STUMP="exp_Boosting_Classifiers_{}.json"
METHOD_NAMES=["BCs1", "BCs2", "BC", "Ta", "Si","RT","RTP"]

def generate_metrics():
    metrics = {
        "BAC":(balanced_accuracy_score,{}),
        "Kappa":(cohen_kappa_score,{}),
        "F1": (f1_score,{"average":"micro"}),
        "PRC": (precision_score,{"average":"micro"}),
        "REC": (recall_score, {"average":"micro"})
    }
    return metrics

class Dims(Enum):
    COMM_SIZES = "comm_sizes"
    BASE_CLASSIFIERS = "base_classifiers"
    METHODS = "methods"
    FOLDS = "folds"
    METRICS = "metrics"


class ExperimentManager:

    def __init__(self, comm_sizes, loggin_queue) -> None:
        self.comm_sizes = comm_sizes
        self.logging_queue = loggin_queue
        self.n_base_classifiers = 0
        self.n_methods = 0

        qh = logging.handlers.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        # if not root.hasHandlers():
        root.addHandler(qh)

        self.logger = logging.getLogger()


    @staticmethod
    def _generate_configs(comm_sizes, n_base_classifiers,n_methods):
        """
        This needs to be delegated into a separate process.
        """

        jvm.start(class_path=settings.CLASSPATH)

        classifier_creator = javabridge.JClassWrapper("classifiercreators.ClassifierCreator")
        n_base_classifiers.value = classifier_creator.nBaseClassifiers()
        n_methods.value = classifier_creator.nMethodsBoosting()
        
        for comm_size in comm_sizes:
        
            classifier_creator = javabridge.JClassWrapper("classifiercreators.ClassifierCreator")
            classifier_config_file_path = os.path.join(settings.CONFIGPATH, CLASSIFIERS_CONFIG_FILE_NAME_STUMP.format(comm_size))
            classifier_creator.createClassifiersConfigBoosting(classifier_config_file_path,comm_size)

        jvm.stop()

    def generate_configs(self):
        self.logger.info("Generating configs")
        n_base_classifiers_val = mp.Value('i')
        n_methods_val = mp.Value('i')
        numbers_gen_process = mp.Process(target=ExperimentManager._generate_configs, args=(self.comm_sizes,n_base_classifiers_val,n_methods_val,))
        numbers_gen_process.start()
        numbers_gen_process.join()

        self.n_base_classifiers = n_base_classifiers_val.value
        self.n_methods = n_methods_val.value

        self.logger.info("Configs generated.")


    def run_experiment_for_datafile(self, data_file_path, results_dir, overwrite=False, append=True):
        data_file_basename = os.path.splitext(data_file_path)[0]
        self.logger.info("Processing set: {}".format(data_file_basename))
        out_dump_filename = os.path.join(results_dir, data_file_basename+".pickle")

        if os.path.exists(out_dump_filename) and not overwrite:
            self.logger.info("Results for {} are present. Skipping".format(data_file_basename))
            return 


class ExperimentRunner:
    def __init__(self, logging_queue,comm_sizes, data_file_path, results_dir, overwrite=False, append=True,
                 n_splits=5, n_repeats=1, n_jobs=1, max_heap_size=None, reset_counter=10,
                   memory_percent_threshold=70 ) -> None:
        
        self.logging_queue = logging_queue
        self.comm_sizes = comm_sizes
        self.data_file_path = data_file_path
        self.results_dir = results_dir
        self.overwrite = overwrite
        self.append = append
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.max_heap_size = max_heap_size
        self.reset_counter = reset_counter
        self.memory_percent_threshold = memory_percent_threshold

        self.data_file_basename = os.path.splitext( os.path.basename(self.data_file_path) )[0]
        self.dump_file_path = os.path.join( results_dir, "{}.pickle".format(self.data_file_basename))

        self.results_storage = None
        self.skf:RepeatedStratifiedKFold = None
        self.input_queue = None
        self.output_queue = None


        qh = logging.handlers.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        # if not root.hasHandlers():
        root.addHandler(qh)

        self.logger = logging.getLogger()

    def _get_data(self):
        X, y, meta = load_arff(self.data_file_path, class_index="last")
        y = np.asanyarray(to_nominal_labels(y)) 
        X,y = shuffle(X,y,random_state=0)
        return X, y
    
    def _generate_results_storage(self):

        comm_size_names = [k for k in self.comm_sizes ]
        fold_names = [k for k in range(self.skf.get_n_splits())]
        bc_names = set()
        method_names = set()
        metric_names = [k for k in generate_metrics()]


        config_file_path = os.path.join(settings.CONFIGPATH, CLASSIFIERS_CONFIG_FILE_NAME_STUMP.format(self.comm_sizes[0]))
        file_handler = open(config_file_path,"r")
        json_repr = json.load(file_handler)
        file_handler.close()

        #Duplicating names!!!
        methods = [ (r["name"], r) for r in json_repr ]
        for m_name , _ in methods:
            bc_name, meth_name  = m_name.split("_")
            bc_names.add(bc_name)
            method_names.add(meth_name)

        coords={
            Dims.COMM_SIZES.value: comm_size_names,
            Dims.FOLDS.value: fold_names,
            Dims.BASE_CLASSIFIERS.value: list(bc_names),
            Dims.METHODS.value: list(method_names),
            Dims.METRICS.value: metric_names,
        }
        
        self.results_storage = ResultsStorage.init_coords(coords=coords, name="Storage")
        if os.path.exists(self.dump_file_path):
            with open(self.dump_file_path,"rb") as fh:
                loaded_storage = pickle.load(fh)
            loaded_storage.name = "loaded"
            self.results_storage = ResultsStorage.merge_with_loaded(loaded_obj=loaded_storage, new_obj=self.results_storage)

    def _put_configs_into_queue(self,X,y):

        for com_size in ResultsStorage.coords_need_recalc(self.results_storage, Dims.COMM_SIZES.value):
            self.logger.info("Committee size for {}: {}".format(self.data_file_basename,com_size))
            config_file_path = os.path.join(settings.CONFIGPATH, CLASSIFIERS_CONFIG_FILE_NAME_STUMP.format(com_size))
            file_handler = open(config_file_path,"r")
            json_repr = json.load(file_handler)
            file_handler.close()
            
            methods = [ (r["name"], r) for r in json_repr ]

            for met_name_full, method_description in tqdm(methods,leave=False,desc="Committees"):
                b_class_name, meth_name = met_name_full.split("_")

                for split_idx, (train_index, test_index) in enumerate( self.skf.split(X, y)):
                    is_put = np.any(
                            np.isnan(
                        self.results_storage.loc[{
                        Dims.COMM_SIZES.value: com_size,
                        Dims.BASE_CLASSIFIERS.value: b_class_name, 
                        Dims.METHODS.value: meth_name,
                        Dims.FOLDS.value: split_idx,
                    }]))
                    if is_put:
                        self.logger.info("Estimator: {} queued for {}; comm size {}".format(met_name_full,
                                    self.data_file_basename, com_size))
                        self.input_queue.put( (method_description,train_index, test_index, split_idx,
                                            b_class_name, meth_name, com_size) )
                    else:
                        self.logger.info("Estimator: {} not queued for {}; comm size {}".format(met_name_full,
                                    self.data_file_basename, com_size))

    def run_experiment(self):

        self.logger.info("Processing set: {}".format(self.data_file_basename))
        if os.path.exists(self.dump_file_path) and not self.overwrite:
            self.logger.info("Results for {} are present. Skipping".format(self.data_file_basename))
            return

        metrics = generate_metrics()
        self.skf = RepeatedStratifiedKFold(n_splits=self.n_splits,n_repeats=self.n_repeats,random_state=0)
        self.input_queue = mp.JoinableQueue()
        self.output_queue = mp.Queue()

        X, y = self._get_data()
        n_processes = mp.cpu_count() if self.n_jobs <=0 else self.n_jobs

        weka_process_pool = WekaProcessPool(input_queue=self.input_queue, output_queue=self.output_queue, X=X, y=y,
                                                classpath=settings.CLASSPATH, metrics=metrics,
                                                logging_queue=self.logging_queue,
                                                max_heap_size=self.max_heap_size,
                                                n_processes=n_processes, logger=self.logger, 
                                                reset_counter=self.reset_counter,
                                                memory_percent_threshold=self.memory_percent_threshold)
        self.logger.info("Weka Pool Starting!")
        weka_process_pool.start_pool()
        self.logger.info("Weka Pool Started!")

        self.logger.info("Generate ResultsStorage")
        self._generate_results_storage()

        result_process = ResultsGatheringProcess(
            name="Result Gatherer",
            results_queue=self.output_queue,
            logging_queue=self.logging_queue,
            dump_file_path=self.dump_file_path,
            results_storage=self.results_storage
        )
        self.logger.info("Result gathering process is about to start, Set {} ".format(result_process))
        result_process.start()
        self.logger.info("Result gathering thread was started: {}".format(result_process))

        self.logger.info("Putting configs into the queue!")
        self._put_configs_into_queue(X, y)

        #finishing processing
        self.input_queue.join()
        self.logger.info("Weka Pool Stopping!")
        weka_process_pool.stop_pool()
        self.logger.info("Weka Pool Stopped!")

        self.logger.info("{}: Sending poision pill to collecting process.".format(self.data_file_basename))

        self.output_queue.put(None)
        self.output_queue.close()
        
        self.logger.info("{}: Waiting for result proces to join".format(self.data_file_basename))

        result_process.join()
        self.logger.info("Result gathering process has been joined: {}".format(result_process))



class WekaProcess(mp.Process):

    def __init__(self, group: None = None, target = None,
                  name: str  = None, args = {}, kwargs  = {}, *,
                    daemon: bool  = None,
                    X = None, y=None, input_queue=None,output_queue = None,
                      classpath=None, metrics=None, logging_queue=None,
                      max_heap_size=None, reset_counter=100, 
                      memory_percent_threshold=70) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon,)

        self.X = X
        self.y = y
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.classpath = classpath
        self.metrics = metrics
        self.logger = None
        self.logging_queue = logging_queue
        self.max_heap_size = max_heap_size
        self.reset_counter = reset_counter
        self.memory_percent_threshold = memory_percent_threshold

    def run(self) -> None:

        qh = logging.handlers.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        # if not root.hasHandlers():
        root.addHandler(qh)

        self.logger = logging.getLogger()

        self.logger.debug("PID {}: Has started.".format( mp.current_process().pid))

        
        if jvm.started is None:
            self.logger.debug("PID {}: Trying to start JVM".format( mp.current_process().pid))
            jvm.start(class_path=self.classpath, max_heap_size=self.max_heap_size)
        else:
            self.logger.debug("PID {}: JVM has already started".format( mp.current_process().pid))

        self.logger.debug("PID {} Entering the main loop".format( mp.current_process().pid))

        while(True):
            try:
                if self.reset_counter < 0:
                    self.logger.debug("PID {} scheduled JVM stopping".format( mp.current_process().pid))
                    jvm.stop()
                    break

                config = self.input_queue.get()

                if config is None:
                    self.logger.debug("PID {} Has acquired poison pill. ".format( mp.current_process().pid))
                    self.input_queue.task_done()
                    self.input_queue.close()
                    self.output_queue.close()
                    self.logger.debug("PID {} stopping JVM".format( mp.current_process().pid))
                    jvm.stop()
                    break
                else:
                    if psutil.virtual_memory().percent > self.memory_percent_threshold:
                        self.reset_counter -= 1

                    self.logger.debug("PID {} Reset Counter: {}".format( mp.current_process().pid, self.reset_counter))
                    method_description, train_index, test_index, split_idx, b_class_name, meth_name, com_size_name = config

                    self.logger.debug("PID {} Acquired another job".format( mp.current_process().pid))

                    X_train, X_test = self.X[train_index], self.X[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]

                    method = WekaEstimator(classname=method_description["classname"], options=method_description["options"])
                    self.logger.debug("PID {} Fit for: {} ".format( mp.current_process().pid, method_description ))
                    method.fit(X_train, y_train)
                    self.logger.debug("PID {} Predict for: {} ".format( mp.current_process().pid, method_description))
                    y_pred = method.predict(X_test)
        
                    self.logger.debug("PID {} Putting results to output queue".format( mp.current_process().pid))

                    for metric_id, metric_name in enumerate(self.metrics):
                        metric_fun, metric_params = self.metrics[metric_name]

                        metric_val = metric_fun(y_test, y_pred, **metric_params)

                        self.output_queue.put( (split_idx, metric_name, b_class_name, meth_name, com_size_name,metric_val ) )

                    self.logger.debug("PID {} All calculated results have been put to output queue".format( mp.current_process().pid))

                    self.input_queue.task_done()

            except qu.Empty:
                self.logger.debug("PID {} empty queue sleeping".format( mp.current_process().pid))
                time.sleep(random.random())
            except Exception as ex:
                self.logger.error("PID {} Exception {}, traceback: {};  for config {}".format( mp.current_process().pid,
                                                                                        ex,traceback.format_exc() ,config[0]))

        self.logger.debug("PID {}: Stopping".format( mp.current_process().pid))        
        return None
    
class LoggerProcess(mp.Process):
    def __init__(self, group = None, target= None, name = None, args={}, kwargs={}, *, daemon= None,
                 output_dir=None, logging_queue=None,log_file_stump="LOG") -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

        self.output_dir = output_dir
        self.logging_queue = logging_queue
        self.log_file_stump = log_file_stump

    
    def run(self) -> None:

        date_string = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        log_file = os.path.join(self.output_dir,"{}_{}.log".format(self.log_file_stump,date_string))
        open(log_file,"w+").close()

        log_format_str = "%(asctime)s;%(levelname)s;%(message)s"
        log_date_format = "%Y-%m-%d %H:%M:%S"
        
        logging.basicConfig(filename=log_file, level=logging.INFO,force=True,format=log_format_str, datefmt=log_date_format)
        logging.captureWarnings(True)

        logger = logging.getLogger()

        while True:
            try:
                record = self.logging_queue.get()
                if record is None:
                    break
                logger.handle(record)
            except qu.Empty:
                time.sleep(random.random())

        return None
    
class ResultsGatheringProcess(mp.Process):
    def __init__(self, group: None = None, name: str = None, args = {}, kwargs= {}, *, daemon: bool = None,
                 results_queue=None, logging_queue=None, dump_file_path=None, results_storage:ResultsStorage = None) -> None:
        super().__init__(group, None, name, args, kwargs, daemon=daemon)

        self.results_queue = results_queue
        self.logging_queue = logging_queue
        self.results_storage:ResultsStorage = results_storage

        self.dump_file_path = dump_file_path
        
    def _make_readable(self,seconds):
        h = int(seconds // 3600)
        m = int((seconds - h * 3600) // 60)
        s = seconds - (h * 3600) - (m * 60)
        return f"{h:0>4d}:{m:0>2d}:{s:0>2f}"

    def _calc_time(self, start_time, curr_time, to_be_gathered,gathered_so_far):

        elapsed_time = curr_time - start_time
        elapsed_time_str = self._make_readable(elapsed_time)

        predicted = int( (elapsed_time * to_be_gathered)/gathered_so_far)
        predicted_str = self._make_readable(predicted)

        secs_per_iter =  elapsed_time/gathered_so_far
        secs_per_iter_str = self._make_readable(secs_per_iter)

        self.logger.debug("PID: {}; Result Gathering Process -- elapsed: {}; total: {}; per_iter: {} ".format(mp.current_process().pid,elapsed_time_str, predicted_str, secs_per_iter_str))        


    def run(self) -> None:

        qh = logging.handlers.QueueHandler(self.logging_queue)
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        # if not root.hasHandlers():
        root.addHandler(qh)

        self.logger = logging.getLogger()

        self.logger.debug("PID {}: Result Gathering Process has started.".format( mp.current_process().pid))

        if self.results_storage is None:
            self.logger.debug("PID {}: Result Gathering Process None storage?.".format( mp.current_process().pid))
        to_be_gathered = ResultsStorage.count_nans(self.results_storage)
        gathered = 0

        #Do not reach this log

        self.logger.debug("PID {}: Result Gathering Process entering main loop.".format( mp.current_process().pid))

        start_time = timer()        
        while True:
            try:
                self.logger.debug("PID: {} Result gathering Process -- get from queue.".format(mp.current_process().pid))
                record = self.results_queue.get()

                if record is None:
                    self.logger.debug("PID: {} Quitting gathering Process.".format(mp.current_process().pid))
                    
                    with open(self.dump_file_path, "wb") as fh:
                        pickle.dump(self.results_storage, file=fh )
                    self.logger.info("PID: {}; Numeric results for {} has been saved.".format(mp.current_process().pid,
                                                                                         self.dump_file_path))
                    break

                gathered += 1
                self.logger.debug("PID: {}; Result Gathering Process -- Collecting another result ({}/{}): {}"\
                                  .format(mp.current_process().pid,gathered,to_be_gathered,record))

                curr_time = timer()
                self._calc_time(start_time, curr_time, to_be_gathered, gathered)
                
                split_idx, metric_id, b_class_idx, meth_idx, com_size_idx,metric_val = record
                self.results_storage.loc[ {
                    Dims.COMM_SIZES.value:com_size_idx,
                    Dims.BASE_CLASSIFIERS.value:b_class_idx,
                    Dims.METHODS.value:meth_idx,
                    Dims.METRICS.value:metric_id,
                    Dims.FOLDS.value:split_idx
                }] = metric_val
                
            except qu.Empty:
                self.logger.debug("PID: {}; Result quque empty".format(mp.current_process().pid))
                time.sleep(random.random())
            except Exception as ex:
                self.logger.error("PID {} Exception in main process loop: {}".format( mp.current_process().pid, ex))
                self.logger.error("PID {} Emergency JVM stop".format( mp.current_process().pid))
                jvm.stop()

        self.logger.debug("PID: {}; Result gathering at exit".format(mp.current_process().pid))
        return None

class ExperimentLogger:
    def __init__(self, logging_dir="./",logging_level=logging.INFO,log_file_stump="LOG"):
        self.logging_dir = logging_dir
        self.logging_level = logging_level
        self.logging_queue = None
        self.logger = None
        self.logger_process = None
        self.log_file_stump = log_file_stump

    def start_logging(self):
        if self.logger is None:
            self.logging_queue = mp.Queue()
            qh = logging.handlers.QueueHandler(self.logging_queue)
            root = logging.getLogger()
            root.setLevel(self.logging_level)
            # if not root.hasHandlers():
            root.addHandler(qh)
            self.logger = root

            self.logger_process = LoggerProcess(output_dir=self.logging_dir, 
                                        logging_queue=self.logging_queue,
                                        log_file_stump= self.log_file_stump
                                        )
            self.logger_process.start()

        return self.logger
    
    def get_logger(self):
        return self.start_logging()
    
    def get_logging_queue(self):
        self.start_logging()
        return self.logging_queue

    def stop_logging(self):
        self.logging_queue.put(None)# poison pill
        self.logging_queue.close()
        self.logger_process.join()

class WekaProcessPool:

    def __init__(self, input_queue, output_queue, X, y,
                                   classpath, metrics, logging_queue,
                                   max_heap_size,n_processes, logger, reset_counter=10,
                                   memory_percent_threshold=70) -> None:
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.X = X
        self.y = y
        self.classpath = classpath
        self.metrics = metrics
        self.logging_queue = logging_queue
        self.max_heap_size = max_heap_size
        self.n_processes = n_processes
        self.logger = logger
        self.processes = None
        self.reset_counter = reset_counter
        self.reset_thread = None
        self.is_running = False
        self.memory_percent_threshold = memory_percent_threshold

    def reset_thread_fun(self):

        self.logger.info("Resetting thread is running.")
        while self.is_running:
            
            time.sleep(1)
            self.logger.info("Resetting thread is checking for dead processes.")
            for i in range( len( self.processes)):
                if not self.processes[i].is_alive():
                    self.logger.info("Pool process needs restart")
                    wp = WekaProcess(name="Weka Process",input_queue=self.input_queue, output_queue=self.output_queue, X=self.X, y=self.y,
                                   classpath=self.classpath, metrics=self.metrics, logging_queue=self.logging_queue,
                                   max_heap_size=self.max_heap_size, reset_counter=self.reset_counter, 
                                   memory_percent_threshold=self.memory_percent_threshold)
                    self.logger.info("Before Restarting WekaProcess {}.".format(wp))
                    wp.start()
                    self.logger.info("Restarted WekaProcess {}.".format(wp))
                    self.processes[i] = wp

    
    def start_pool(self):
        self.logger.info("Weka Pool start_pool")
        self.is_running = True
        self.processes = []

        for i in range(self.n_processes):
            wp = WekaProcess(name="Weka Process",input_queue=self.input_queue, output_queue=self.output_queue, X=self.X, y=self.y,
                                   classpath=self.classpath, metrics=self.metrics, logging_queue=self.logging_queue,
                                   max_heap_size=self.max_heap_size, reset_counter=self.reset_counter,
                                   memory_percent_threshold=self.memory_percent_threshold)
            self.logger.info("Before Starting WekaProcess {}.".format(wp))
            wp.start()
            self.logger.info("Started WekaProcess {}.".format(wp))
            self.processes.append(wp)

        self.logger.info("Creating reset thread")
        self.reset_thread = th.Thread(target=self.reset_thread_fun)
        self.logger.info("Starting reset thread")
        self.reset_thread.start()
        self.logger.info("Started reset thread: {}".format(self.reset_thread))


    def stop_pool(self):
        self.logger.info("Stopping restart thread.")
        self.is_running = False

        self.reset_thread.join()

        self.logger.info("Restarting thread has been stopped.")

        self.logger.info("Sending poison pill to WekaProcesses")
        for i in range(len(self.processes)):
            self.input_queue.put(None) 

        self.logger.info("Waiting for weka processes to join")

        for wp in self.processes:
            wp.join()
            self.logger.info("{} has joined".format(wp))

        self.logger.info("All weka processes have been joined")

def run_experiment(data_dir="./data", results_dir="./results", committee_sizes=[11,21], overwrite=False,
                   experiment_logger:ExperimentLogger=None,n_splits=10, n_repeats=1,
                   n_jobs = -1, max_heap_size=None, reset_counter=10,
                   memory_percent_threshold=70):

    logger = experiment_logger.get_logger()
    logging_queue = experiment_logger.get_logging_queue()

    exp_manager = ExperimentManager(comm_sizes=committee_sizes, loggin_queue=logging_queue)

    logger.info("Starting the experiment")
    os.makedirs(results_dir, exist_ok=True)
    
    exp_manager.generate_configs()

    n_base_classifiers = exp_manager.n_base_classifiers
    n_methods = exp_manager.n_methods
    
    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for data_file in filenames:

            data_file_basename = os.path.splitext(data_file)[0]
            logger.info("Processing set: {}".format(data_file_basename))
            data_file_path = os.path.join(dirpath, data_file)

            experiment_runner = ExperimentRunner(
                        logging_queue=logging_queue,
                        comm_sizes=committee_sizes,
                        data_file_path=data_file_path,
                        results_dir=results_dir,
                        overwrite=overwrite,
                        append=True,
                        n_splits=n_splits,
                        n_repeats = n_repeats,
                        n_jobs= n_jobs,
                        max_heap_size= max_heap_size,
                        reset_counter= reset_counter,
                        memory_percent_threshold= memory_percent_threshold,
            )
            experiment_runner.run_experiment()

    logger.info("Experiment ended!")
        
            

def analyze_results(results_dir="./results", alpha=0.05):

    logging.info("Starting the results analysis")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                
                if dump_file.endswith(".pickle"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    out_pdf_filename = os.path.join(results_dir, dump_file_basename+".pdf")
                    out_tex_directory = os.path.join(results_dir, dump_file_basename+"_tex")
                    os.makedirs(out_tex_directory,exist_ok=True)

                    logging.info("Processing: {}".format(dump_file_basename))

                    with open(dump_file_path,"rb") as fh:
                        results_storage = pickle.load(fh)
                    
                    metrics_names = results_storage[Dims.METRICS.value].values
                    base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
                    method_names = results_storage[Dims.METHODS.value].values
                    comm_sizes = results_storage[Dims.COMM_SIZES.value].values
                    # comm_sizes x base_classifiers x methods x metrics x splits
                    results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                            Dims.METHODS.value, Dims.METRICS.value,
                                             Dims.FOLDS.value ).to_numpy()
                    
                    #TODO should also be changed?
                    method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
                    results = results[:,:,method_indices]

                    with PdfPages(out_pdf_filename) as pdf:

                        for metric_id, metric_name in enumerate(metrics_names):
                            for base_classifier_id, base_classifier_name in enumerate(base_classifiers_names):

                                for method_id, method_name in enumerate( method_names ):
                                    # com_sizes x splits
                                    results_subset = results[:,base_classifier_id, method_id,metric_id,:]
                                    medians = np.median(results_subset,axis=1)
                                    q1 = np.quantile(results_subset,0.25, axis=1)
                                    q3 = np.quantile(results_subset,0.75, axis=1)
                                    plt.plot(comm_sizes, medians,label="{} -- median".format(method_name),marker="o")
                                    plt.fill_between(comm_sizes,q1,q3,alpha=0.2)
                                    plt.xlabel("Committee size")
                                    plt.ylabel("Metric value: {}".format(metric_name))
                                    plt.title("{}, {}".format(base_classifier_name, metric_name))
                                    plt.legend(loc='upper right')
                                pdf.savefig()
                                tex_graph_filename = os.path.join(out_tex_directory
                                                    , "{}_{}_{}.tex".format(dump_file_basename,metric_name, base_classifier_name))
                                tikzplotlib.save(tex_graph_filename,standalone=True)
                                plt.close()

def analyze_results_sizes(results_dir="./results", alpha=0.05):

    logging.info("Starting the results analysis")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                if dump_file.endswith(".pickle"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    
                    out_tex_directory = os.path.join(results_dir, dump_file_basename+"_tex")
                    os.makedirs(out_tex_directory,exist_ok=True)

                    logging.info("Processing: {}".format(dump_file_basename))
                    
                    with open(dump_file_path,"rb") as fh:
                        results_storage = pickle.load(fh)
                    
                    metrics_names = results_storage[Dims.METRICS.value].values
                    base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
                    method_names = results_storage[Dims.METHODS.value].values
                    comm_sizes = results_storage[Dims.COMM_SIZES.value].values
                    # comm_sizes x base_classifiers x methods x metrics x splits
                    results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                            Dims.METHODS.value, Dims.METRICS.value,
                                             Dims.FOLDS.value ).to_numpy()

                    method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
                    results = results[:,:,method_indices]

                    n_methods = len(method_names)

                    for comm_size_id, comm_size in enumerate(comm_sizes):
                        size_results_dir = os.path.join(results_dir, dump_file_basename,"{}".format(comm_size))
                        os.makedirs(size_results_dir, exist_ok=True)

                        out_pdf_filename = os.path.join(size_results_dir,"{}_{}.pdf".format(dump_file_basename, comm_size))
                        report_filename = os.path.join(size_results_dir, "{}_{}.md".format(dump_file_basename, comm_size))
                        report_file_handler = open(report_filename, "w")

                        with PdfPages(out_pdf_filename) as pdf:
                            for metric_id, metric_name in enumerate(metrics_names):
                                print("# {}".format(metric_name), file=report_file_handler)
                                for base_classifier_id, base_classifier_name in enumerate(base_classifiers_names):
                                    print("## {}".format(base_classifier_name), file=report_file_handler)
                                    
                                    # methods x splits
                                    sub_results = results[comm_size_id, base_classifier_id, :, metric_id, :]

                                    plt.boxplot(sub_results.transpose())
                                    plt.title( "{}, {}".format(metric_name, base_classifier_name) )
                                    plt.xticks( range(1,len(method_names)+1), method_names )
                                    pdf.savefig()                                    
                                    plt.close()

                                    fold_means = np.mean(sub_results, axis=1)
                                    fold_sdevs = np.std(sub_results, axis=1)

                                    for method_id, method_name in enumerate( method_names ):

                                        print("{} {} +- {}".format(method_name, fold_means[method_id], fold_sdevs[method_id]), file=report_file_handler)
                                        plt.errorbar([method_id], fold_means[method_id], fold_sdevs[method_id], linestyle='None', marker='^', label=method_name)
                                    
                                    plt.title("{}, {}".format(metric_name,base_classifier_name))
                                    plt.xlabel("classifier_id")
                                    plt.ylabel("Criterion value")
                                    plt.legend()
                                    pdf.savefig()
                                    plt.close()

                                    p_vals = np.zeros( (n_methods, n_methods) )
                                    values = sub_results.transpose()

                                    for i in range(n_methods):
                                        for j in range(n_methods):
                                            if i == j:
                                                continue

                                            values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                                            if values_squared_diff > 1E-4:
                                                with warnings.catch_warnings(): #Normal approximation
                                                    warnings.simplefilter("ignore")
                                                    p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                            else:
                                                p_vals[i,j] = 1.0
                                    
                                    p_val_vec  = p_val_matrix_to_vec(p_vals)

                                    p_val_vec_corrected = multipletests(p_val_vec, method='hommel')


                                    corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)

                                    p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
                                    print("PVD:\n", p_val_df,file=report_file_handler)

                         
                        report_file_handler.close()

def analyze_results_sizes_latex(results_dir="./results", alpha=0.05):

    logging.info("Starting the results analysis")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                if dump_file.endswith(".pickle"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    
                    out_tex_directory = os.path.join(results_dir, dump_file_basename+"_tex")
                    os.makedirs(out_tex_directory,exist_ok=True)

                    logging.info("Processing: {}".format(dump_file_basename))
                    
                    with open(dump_file_path,"rb") as fh:
                        results_storage = pickle.load(fh)
                    
                    metrics_names = results_storage[Dims.METRICS.value].values
                    base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
                    method_names = results_storage[Dims.METHODS.value].values
                    comm_sizes = results_storage[Dims.COMM_SIZES.value].values
                    # comm_sizes x base_classifiers x methods x metrics x splits
                    results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                            Dims.METHODS.value, Dims.METRICS.value,
                                             Dims.FOLDS.value ).to_numpy()

                    method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
                    results = results[:,:,method_indices]

                    n_methods = len(method_names)

                    for comm_size_id, comm_size in enumerate(comm_sizes):
                        size_results_dir = os.path.join(results_dir, dump_file_basename,"{}".format(comm_size))
                        os.makedirs(size_results_dir, exist_ok=True)

                        report_filename = os.path.join(size_results_dir, "{}_{}.tex".format(dump_file_basename, comm_size))
                        report_file_handler = open(report_filename, "w")


                        for metric_id, metric_name in enumerate(metrics_names):
                            print("# {}".format(metric_name), file=report_file_handler)
                            for base_classifier_id, base_classifier_name in enumerate(base_classifiers_names):
                                print("## {}".format(base_classifier_name), file=report_file_handler)
                                
                                # methods x splits
                                sub_results = results[comm_size_id, base_classifier_id, :, metric_id, :]

                                fold_means = np.mean(sub_results, axis=1)
                                fold_sdevs = np.std(sub_results, axis=1)

                                vals = [ [m,"$\pm$",s] for m, s in zip(fold_means, fold_sdevs) ]

                                mi = pd.MultiIndex.from_arrays( [ ["{}, {}".format(metric_name,base_classifier_name) for _ in range(3)], ["" for _ in range(3)] ] )
                                
                                df = pd.DataFrame(vals, index=["{}".format(n) for n in method_names], columns=mi)

                                df.style.format(precision=3,na_rep="")\
                                    .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                                    .to_latex(report_file_handler, multicol_align="c")
                                
                                print("", file=report_file_handler)

                                p_vals = np.zeros( (n_methods, n_methods) )
                                values = sub_results.transpose()

                                for i in range(n_methods):
                                    for j in range(n_methods):
                                        if i == j:
                                            continue

                                        values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                                        if values_squared_diff > 1E-4:
                                            with warnings.catch_warnings(): #Normal approximation
                                                warnings.simplefilter("ignore")
                                                p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                                        else:
                                            p_vals[i,j] = 1.0
                                
                                p_val_vec  = p_val_matrix_to_vec(p_vals)

                                p_val_vec_corrected = multipletests(p_val_vec, method='hommel')


                                corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)

                                p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
                                mic = pd.MultiIndex.from_arrays(  [
                                ["{}, {}".format(metric_name, base_classifier_name) for _ in range( n_methods -1)],
                                ["{}".format(s) for s in method_names[1:]]
                                ] )

                                corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
                                corr_p_val_matrix[np.tril_indices(corr_p_val_matrix.shape[0],k=-1)] = np.nan

                                print("P-Vals: ", file=report_file_handler)

                                p_val_df = pd.DataFrame(corr_p_val_matrix[:-1,1:], columns=mic,index=method_names[:-1] )
                                p_val_df.style.format(precision=3,na_rep="",escape="latex").highlight_between(left=0, right=0.05,  props='textbf:--rwrap;').\
                                    format_index(escape="latex",axis=0).format_index(escape="latex", axis=1).\
                                    to_latex(report_file_handler, multicol_align="c", hrules=False)

                         
                        report_file_handler.close()

def analyze_results_sizes_ranks(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()
        
        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

        overall_results.append( np.mean(results,axis=-1) )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics

    out_pdf_filename = os.path.join(results_dir, "Ranks.pdf")
    tex_out_dir = os.path.join(results_dir,"Ranks")
    os.makedirs(tex_out_dir, exist_ok=True)

    rank_report_path = os.path.join(results_dir, "Rank.tex")
    rank_report_handler = open(rank_report_path,"w")
    

    with PdfPages(out_pdf_filename) as pdf:

        for metric_id, metric_name in enumerate(metrics_names):
            print("# {}".format(metric_name), file=rank_report_handler)            
            for classifier_id, classifier_name in enumerate(base_classifiers_names):
                print("## {}".format(classifier_name), file=rank_report_handler)

                sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods
                sub_results_ranks = rankdata(sub_results, axis=2)
                sub_results_avg_ranks = np.mean(sub_results_ranks,axis=0)# comm_sizes x methods

                mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
                av_rnk_df = pd.DataFrame(sub_results_avg_ranks,columns=mi , index=["Avg Rnk {}:{}".format(a,si) for si,a in zip( comm_sizes,string.ascii_letters)])

                for comm_size_id, (comm_size,comm_letter) in enumerate( zip( comm_sizes,string.ascii_letters)):
                    p_vals = np.zeros( (n_methods, n_methods) )
                    values = sub_results[:,comm_size_id,:] # sets  x methods

                    for i in range(n_methods):
                        for j in range(n_methods):
                            if i == j:
                                continue

                            values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                            if values_squared_diff > 1E-4:
                                with warnings.catch_warnings(): #Normal approximation
                                    warnings.simplefilter("ignore")
                                    p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                            else:
                                p_vals[i,j] = 1.0
                    
                    p_val_vec  = p_val_matrix_to_vec(p_vals)

                    p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                    corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                    s_test_outcome = []
                    for i in range(n_methods):
                        out_a = []
                        for j in range(n_methods):
                            if sub_results_avg_ranks[comm_size_id,i] > sub_results_avg_ranks[comm_size_id,j] and corr_p_val_matrix[i,j]<alpha:
                                out_a.append(j+1)
                        if len(out_a) == 0:
                            s_test_outcome.append("--")
                        else:
                            s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                    av_rnk_df.loc["{} {}:{}_T".format("Avg Rnk",comm_letter,comm_size)] = s_test_outcome
                    av_rnk_df.sort_index(inplace=True)

                av_rnk_df.style.format(precision=3,na_rep="")\
                .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                .to_latex(rank_report_handler, multicol_align="c")

                for method_id, method_name in enumerate( method_names ):
                    plt.plot(comm_sizes, sub_results_avg_ranks[:,method_id], label=method_name, marker="o")
                    plt.title("{}, {} -- Average ranks".format(metric_name, classifier_name))
                    plt.xlabel("Committee size")
                    plt.ylabel("Criterion value ranks")
                    plt.ylim((1,n_methods))
                    plt.legend(loc='lower right')
                pdf.savefig()
                tex_graph_filepath = os.path.join(tex_out_dir,"{}_{}.tex".format(metric_name,classifier_name))
                tikzplotlib.save(tex_graph_filepath,standalone=True)
                plt.close()

    out_pdf_filename_all = os.path.join(results_dir, "Ranks_all.pdf")
    tex_out_dir_all = os.path.join(results_dir,"Ranks_all")
    os.makedirs(tex_out_dir_all, exist_ok=True)

    rank_report_all_path = os.path.join(results_dir, "Rank_all.tex")
    rank_report_all_handler = open(rank_report_all_path,"w")


    with PdfPages(out_pdf_filename_all) as pdf:

        for metric_id, metric_name in enumerate(metrics_names):
            print("# {}".format(metric_name),file=rank_report_all_handler)

            sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods
            sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods
            # comm_sizes x (sets * classifiers)  x methods
            sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

            sub_results_ranks = rankdata(sub_results_concatenate, axis=2)
            sub_results_avg_ranks = np.mean(sub_results_ranks,axis=1)# comm_sizes x methods

            mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
            av_rnk_df = pd.DataFrame(sub_results_avg_ranks,columns=mi , index=["Avg Rnk {}:{}".format(a,si) for si,a in zip( comm_sizes, string.ascii_letters)])

            for comm_size_id, (comm_size, comm_letter) in enumerate( zip(comm_sizes,string.ascii_letters)):
                p_vals = np.zeros( (n_methods, n_methods) )
                values = sub_results_concatenate[comm_size_id] # (sets * classifiers) x methods

                for i in range(n_methods):
                    for j in range(n_methods):
                        if i == j:
                            continue

                        values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                        if values_squared_diff > 1E-4:
                            with warnings.catch_warnings(): #Normal approximation
                                warnings.simplefilter("ignore")
                                p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                        else:
                            p_vals[i,j] = 1.0
                
                p_val_vec  = p_val_matrix_to_vec(p_vals)

                p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                s_test_outcome = []
                for i in range(n_methods):
                    out_a = []
                    for j in range(n_methods):
                        if sub_results_avg_ranks[comm_size_id,i] > sub_results_avg_ranks[comm_size_id,j] and corr_p_val_matrix[i,j]<alpha:
                            out_a.append(j+1)
                    if len(out_a) == 0:
                        s_test_outcome.append("--")
                    else:
                        s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}")
                av_rnk_df.loc["{} {}:{}_T".format("Avg Rnk",comm_letter,comm_size)] = s_test_outcome
                av_rnk_df.sort_index(inplace=True)

            av_rnk_df.style.format(precision=3,na_rep="")\
            .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
            .to_latex(rank_report_all_handler, multicol_align="c")

            for method_id, method_name in enumerate( method_names ):
                plt.plot(comm_sizes, sub_results_avg_ranks[:,method_id], label=method_name, marker="o")
                plt.title("{}-- Average ranks".format(metric_name))
                plt.xlabel("Committee size")
                plt.ylabel("Criterion value ranks")
                plt.ylim((1,n_methods))
                plt.legend(loc='lower right')
            pdf.savefig()
            tex_graph_filepath = os.path.join(tex_out_dir_all,"{}.tex".format(metric_name))
            tikzplotlib.save(tex_graph_filepath,standalone=True)
            plt.close()

    rank_report_all_handler.close()

def analyze_results_sizes_ranks2(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    n_base_classifiers = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_comm_sizes = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()

        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)
        n_comm_sizes = len(comm_sizes)
        n_base_classifiers = len(base_classifiers_names)

        overall_results.append( results )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics x splits

    out_pdf_filename = os.path.join(results_dir, "Ranks_wra.pdf")
    tex_out_dir = os.path.join(results_dir,"Ranks_wra")
    os.makedirs(tex_out_dir, exist_ok=True)

    with PdfPages(out_pdf_filename) as pdf:

        for metric_id, metric_name in enumerate(metrics_names):
            for classifier_id, classifier_name in enumerate(base_classifiers_names):
  

                sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods x splits

                ranks = np.zeros( (n_comm_sizes, n_methods) )

                for comm_size_id, comm_size in enumerate(comm_sizes):
                    ranks[comm_size_id] = wra( sub_results[:,comm_size_id], 0.5 * np.ones(n_sets) )
                

                for method_id, method_name in enumerate( method_names ):
                    plt.plot(comm_sizes, ranks[:,method_id], label=method_name, marker="o")
                    plt.title("{}, {} -- Average ranks".format(metric_name, classifier_name))
                    plt.xlabel("Committee size")
                    plt.ylabel("Criterion value ranks")
                    plt.ylim((1,n_methods))
                    plt.legend(loc='lower right')
                pdf.savefig()
                tex_graph_filepath = os.path.join(tex_out_dir,"{}_{}.tex".format(metric_name,classifier_name))
                tikzplotlib.save(tex_graph_filepath,standalone=True)
                plt.close()

    out_pdf_filename_all = os.path.join(results_dir, "Ranks_all_wra.pdf")
    tex_out_dir_all = os.path.join(results_dir,"Ranks_all_wra")
    os.makedirs(tex_out_dir_all, exist_ok=True)

    


    with PdfPages(out_pdf_filename_all) as pdf:

        for metric_id, metric_name in enumerate(metrics_names):

            sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods x splits
            sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods x splits
            # comm_sizes x (sets * classifiers)  x methods x splits
            sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

            ranks = np.zeros( (n_comm_sizes, n_methods) )

            for comm_size_id, comm_size in enumerate(comm_sizes):
                ranks[comm_size_id] = wra( sub_results_concatenate[comm_size_id], 0.5 * np.ones(n_sets * n_base_classifiers) )
            
            for method_id, method_name in enumerate( method_names ):
                plt.plot(comm_sizes, ranks[:,method_id], label=method_name, marker="o")
                plt.title("{}-- Average ranks".format(metric_name))
                plt.xlabel("Committee size")
                plt.ylabel("Criterion value ranks")
                plt.ylim((1,n_methods))
                plt.legend(loc='lower right')
            pdf.savefig()
            tex_graph_filepath = os.path.join(tex_out_dir_all,"{}.tex".format(metric_name))
            tikzplotlib.save(tex_graph_filepath,standalone=True)
            plt.close()



def analyze_results_sizes_ranks_std(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()

        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

        overall_results.append( np.std(results,axis=-1) )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics

    out_pdf_filename = os.path.join(results_dir, "Ranks_std.pdf")
    tex_out_dir = os.path.join(results_dir,"Ranks_std")
    os.makedirs(tex_out_dir, exist_ok=True)

    rank_report_path = os.path.join(results_dir, "Rank_std.tex")
    rank_report_handler = open(rank_report_path,"w")
    

    with PdfPages(out_pdf_filename) as pdf:

        for metric_id, metric_name in enumerate(metrics_names):
            print("# {}".format(metric_name), file=rank_report_handler)            
            for classifier_id, classifier_name in enumerate(base_classifiers_names):
                print("## {}".format(classifier_name), file=rank_report_handler)

                sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods
                sub_results_ranks = rankdata(sub_results, axis=2)
                sub_results_avg_ranks = np.mean(sub_results_ranks,axis=0)# comm_sizes x methods

                mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
                av_rnk_df = pd.DataFrame(sub_results_avg_ranks,columns=mi , index=["Avg Rnk {}:{}".format(a,si) for si,a in zip( comm_sizes,string.ascii_letters)])

                for comm_size_id, (comm_size,comm_letter) in enumerate( zip( comm_sizes,string.ascii_letters)):
                    p_vals = np.zeros( (n_methods, n_methods) )
                    values = sub_results[:,comm_size_id,:] # sets  x methods

                    for i in range(n_methods):
                        for j in range(n_methods):
                            if i == j:
                                continue

                            values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                            if values_squared_diff > 1E-4:
                                with warnings.catch_warnings(): #Normal approximation
                                    warnings.simplefilter("ignore")
                                    p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                            else:
                                p_vals[i,j] = 1.0
                    
                    p_val_vec  = p_val_matrix_to_vec(p_vals)

                    p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                    corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                    s_test_outcome = []
                    for i in range(n_methods):
                        out_a = []
                        for j in range(n_methods):
                            if sub_results_avg_ranks[comm_size_id,i] < sub_results_avg_ranks[comm_size_id,j] and corr_p_val_matrix[i,j]<alpha:
                                out_a.append(j+1)
                        if len(out_a) == 0:
                            s_test_outcome.append("--")
                        else:
                            s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}") 
                    av_rnk_df.loc["{} {}:{}_T".format("Avg Rnk",comm_letter,comm_size)] = s_test_outcome
                    av_rnk_df.sort_index(inplace=True)

                av_rnk_df.style.format(precision=3,na_rep="")\
                .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
                .to_latex(rank_report_handler, multicol_align="c")

                for method_id, method_name in enumerate( method_names ):
                    plt.plot(comm_sizes, sub_results_avg_ranks[:,method_id], label=method_name, marker="o")
                    plt.title("{}, {} -- Average ranks".format(metric_name, classifier_name))
                    plt.xlabel("Committee size")
                    plt.ylabel("Criterion value ranks")
                    plt.ylim((1,n_methods))
                    plt.legend(loc='lower right')
                pdf.savefig()
                tex_graph_filepath = os.path.join(tex_out_dir,"{}_{}.tex".format(metric_name,classifier_name))
                tikzplotlib.save(tex_graph_filepath,standalone=True)
                plt.close()

    out_pdf_filename_all = os.path.join(results_dir, "Ranks_all_std.pdf")
    tex_out_dir_all = os.path.join(results_dir,"Ranks_all_std")
    os.makedirs(tex_out_dir_all, exist_ok=True)

    rank_report_all_path = os.path.join(results_dir, "Rank_all_std.tex")
    rank_report_all_handler = open(rank_report_all_path,"w")


    with PdfPages(out_pdf_filename_all) as pdf:

        for metric_id, metric_name in enumerate(metrics_names):
            print("# {}".format(metric_name),file=rank_report_all_handler)

            sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods
            sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods
            # comm_sizes x (sets * classifiers)  x methods
            sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

            sub_results_ranks = rankdata(sub_results_concatenate, axis=2)
            sub_results_avg_ranks = np.mean(sub_results_ranks,axis=1)# comm_sizes x methods

            mi = pd.MultiIndex.from_arrays( [ ["{}".format(metric_name) for _ in range(n_methods)], [m for m in method_names] ] )
            av_rnk_df = pd.DataFrame(sub_results_avg_ranks,columns=mi , index=["Avg Rnk {}:{}".format(a,si) for si,a in zip( comm_sizes, string.ascii_letters)])

            for comm_size_id, (comm_size, comm_letter) in enumerate( zip(comm_sizes,string.ascii_letters)):
                p_vals = np.zeros( (n_methods, n_methods) )
                values = sub_results_concatenate[comm_size_id] # (sets * classifiers) x methods

                for i in range(n_methods):
                    for j in range(n_methods):
                        if i == j:
                            continue

                        values_squared_diff = np.sqrt (np.sum( (values[:,i] - values[:,j])**2 ) )
                        if values_squared_diff > 1E-4:
                            with warnings.catch_warnings(): #Normal approximation
                                warnings.simplefilter("ignore")
                                p_vals[i,j]  = wilcoxon(values[:,i], values[:,j]).pvalue #mannwhitneyu(values[:,i], values[:,j]).pvalue
                        else:
                            p_vals[i,j] = 1.0
                
                p_val_vec  = p_val_matrix_to_vec(p_vals)

                p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
                corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods,symmetrize=True)

                s_test_outcome = []
                for i in range(n_methods):
                    out_a = []
                    for j in range(n_methods):
                        if sub_results_avg_ranks[comm_size_id,i] < sub_results_avg_ranks[comm_size_id,j] and corr_p_val_matrix[i,j]<alpha:
                            out_a.append(j+1)
                    if len(out_a) == 0:
                        s_test_outcome.append("--")
                    else:
                        s_test_outcome.append("{\\scriptsize " + ",".join([ str(x) for x in out_a]) + "}")
                av_rnk_df.loc["{} {}:{}_T".format("Avg Rnk",comm_letter,comm_size)] = s_test_outcome
                av_rnk_df.sort_index(inplace=True)

            av_rnk_df.style.format(precision=3,na_rep="")\
            .format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)\
            .to_latex(rank_report_all_handler, multicol_align="c")

            for method_id, method_name in enumerate( method_names ):
                plt.plot(comm_sizes, sub_results_avg_ranks[:,method_id], label=method_name, marker="o")
                plt.title("{}-- Average ranks".format(metric_name))
                plt.xlabel("Committee size")
                plt.ylabel("Criterion value ranks")
                plt.ylim((1,n_methods))
                plt.legend(loc='lower right')
            pdf.savefig()
            tex_graph_filepath = os.path.join(tex_out_dir_all,"{}.tex".format(metric_name))
            tikzplotlib.save(tex_graph_filepath,standalone=True)
            plt.close()

    rank_report_all_handler.close()

def analyze_results_sizes_trends_means(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()
        
        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

        overall_results.append( np.mean(results,axis=-1) )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics

    
    report_path = os.path.join(results_dir, "Trend_means.md")
    report_handler = open(report_path,"w")
    

    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name), file=report_handler)            
        for classifier_id, classifier_name in enumerate(base_classifiers_names):
            print("## {}".format(classifier_name), file=report_handler)

            #method , no, up , down
            counts = np.zeros( (n_methods, 3) )
            sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods

            for set_id in range(n_sets):
                for method_id, method_name in enumerate( method_names ):
                    sub_sub_results = sub_results[set_id,:, method_id] # comm_sizes
                    mk_result = mk.original_test(sub_sub_results)
                    if mk_result.p <= alpha:
                        if mk_result.slope >0:
                            counts[method_id, 1]+=1
                        else:
                            counts[method_id, 2]+=1 
                    else:
                        counts[method_id,0]+=1

            counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
            counts_pd.to_markdown(report_handler)
            print("\n", file=report_handler)
            
            p_vals = np.zeros( (n_methods, n_methods))
            for i in range(n_methods):
                for j in range(n_methods):
                    if i == j:
                        continue

                    sub_counts = counts[[i,j],:]
                    _, p_val = chi_homogenity(sub_counts)
                    p_vals[i,j] = p_val

            p_val_vec  = p_val_matrix_to_vec(p_vals)
            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
            p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
            print("PVD:\n", p_val_df,file=report_handler)

    report_handler.close()
            
    report_all_path = os.path.join(results_dir, "Trend_means_all.md")
    report_all_handler = open(report_all_path,"w")



    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name),file=report_all_handler)

        counts = np.zeros( (n_methods, 3) )

        sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods
        sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods
        # comm_sizes x (sets * classifiers)  x methods
        sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

        for method_id, method_name in enumerate( method_names ):
            for set_id in range( sub_results_concatenate.shape[1]):
                t_data = sub_results_concatenate[:,set_id, method_id]
                mk_result = mk.original_test(t_data)
                if mk_result.p <= alpha:
                    if mk_result.slope >0:
                        counts[method_id, 1]+=1
                    else:
                        counts[method_id, 2]+=1 
                else:
                    counts[method_id,0]+=1

        counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
        counts_pd.to_markdown(report_all_handler)
        print("\n", file=report_all_handler)

        p_vals = np.zeros( (n_methods, n_methods))
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    continue

                sub_counts = counts[[i,j],:]
                _, p_val = chi_homogenity(sub_counts)
                p_vals[i,j] = p_val

        p_val_vec  = p_val_matrix_to_vec(p_vals)
        p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
        p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
        print("PVD:\n", p_val_df,file=report_all_handler)

    report_all_handler.close()

def analyze_results_sizes_trends_means_latex(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()

        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

        overall_results.append( np.mean(results,axis=-1) )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics

    
    report_path = os.path.join(results_dir, "Trend_means.tex")
    report_handler = open(report_path,"w")
    

    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name), file=report_handler)            
        for classifier_id, classifier_name in enumerate(base_classifiers_names):
            print("## {}".format(classifier_name), file=report_handler)

            #method , no, up , down
            counts = np.zeros( (n_methods, 3) )
            sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods

            for set_id in range(n_sets):
                for method_id, method_name in enumerate( method_names ):
                    sub_sub_results = sub_results[set_id,:, method_id] # comm_sizes
                    mk_result = mk.original_test(sub_sub_results)
                    if mk_result.p <= alpha:
                        if mk_result.slope >0:
                            counts[method_id, 1]+=1
                        else:
                            counts[method_id, 2]+=1 
                    else:
                        counts[method_id,0]+=1

            counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
            counts_pd.to_latex(report_handler)
            print("\n", file=report_handler)
            
            p_vals = np.zeros( (n_methods, n_methods))
            for i in range(n_methods):
                for j in range(n_methods):
                    if i == j:
                        continue

                    sub_counts = counts[[i,j],:]
                    _, p_val = chi_homogenity(sub_counts)
                    p_vals[i,j] = p_val

            p_val_vec  = p_val_matrix_to_vec(p_vals)
            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)

            p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
            mic = pd.MultiIndex.from_arrays(  [
                                ["{}, {}".format(metric_name, classifier_name) for _ in range( n_methods -1)],
                                ["{}".format(s) for s in method_names[1:]]
                                ] )

            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
            corr_p_val_matrix[np.tril_indices(corr_p_val_matrix.shape[0],k=-1)] = np.nan

            print("PVD:\n",file=report_handler)
            p_val_df = pd.DataFrame(corr_p_val_matrix[:-1,1:], columns=mic,index=method_names[:-1] )
            p_val_df.style.format(precision=3,na_rep="",escape="latex").highlight_between(left=0, right=0.05,  props='textbf:--rwrap;').\
                format_index(escape="latex",axis=0).format_index(escape="latex", axis=1).\
                to_latex(report_handler, multicol_align="c", hrules=False)

    report_handler.close()
            
    report_all_path = os.path.join(results_dir, "Trend_means_all.tex")
    report_all_handler = open(report_all_path,"w")



    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name),file=report_all_handler)

        counts = np.zeros( (n_methods, 3) )

        sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods
        sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods
        # comm_sizes x (sets * classifiers)  x methods
        sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

        for method_id, method_name in enumerate( method_names ):
            for set_id in range( sub_results_concatenate.shape[1]):
                t_data = sub_results_concatenate[:,set_id, method_id]
                mk_result = mk.original_test(t_data)
                if mk_result.p <= alpha:
                    if mk_result.slope >0:
                        counts[method_id, 1]+=1
                    else:
                        counts[method_id, 2]+=1 
                else:
                    counts[method_id,0]+=1

        counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
        counts_pd.to_latex(report_all_handler)
        print("\n", file=report_all_handler)

        p_vals = np.zeros( (n_methods, n_methods))
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    continue

                sub_counts = counts[[i,j],:]
                _, p_val = chi_homogenity(sub_counts)
                p_vals[i,j] = p_val

        p_val_vec  = p_val_matrix_to_vec(p_vals)
        p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
        p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )

        mic = pd.MultiIndex.from_arrays(  [
                ["{}".format(metric_name, classifier_name) for _ in range( n_methods -1)],
                ["{}".format(s) for s in method_names[1:]]
                ] )

        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
        corr_p_val_matrix[np.tril_indices(corr_p_val_matrix.shape[0],k=-1)] = np.nan

        print("P-Vals: ", file=report_all_handler)

        p_val_df = pd.DataFrame(corr_p_val_matrix[:-1,1:], columns=mic,index=method_names[:-1] )
        p_val_df.style.format(precision=3,na_rep="",escape="latex").highlight_between(left=0, right=0.05,  props='textbf:--rwrap;').\
            format_index(escape="latex",axis=0).format_index(escape="latex", axis=1).\
            to_latex(report_all_handler, multicol_align="c", hrules=False)

    report_all_handler.close()


def analyze_results_sizes_trends_stds(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()

        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

        overall_results.append( np.std(results,axis=-1) )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics

    
    report_path = os.path.join(results_dir, "Trend_stds.md")
    report_handler = open(report_path,"w")
    

    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name), file=report_handler)            
        for classifier_id, classifier_name in enumerate(base_classifiers_names):
            print("## {}".format(classifier_name), file=report_handler)

            #method , no, up , down
            counts = np.zeros( (n_methods, 3) )
            sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods

            for set_id in range(n_sets):
                for method_id, method_name in enumerate( method_names ):
                    sub_sub_results = sub_results[set_id,:, method_id] # comm_sizes
                    mk_result = mk.original_test(sub_sub_results)
                    if mk_result.p <= alpha:
                        if mk_result.slope >0:
                            counts[method_id, 1]+=1
                        else:
                            counts[method_id, 2]+=1 
                    else:
                        counts[method_id,0]+=1

            counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
            counts_pd.to_markdown(report_handler)
            print("\n", file=report_handler)
            p_vals = np.zeros( (n_methods, n_methods))
            for i in range(n_methods):
                for j in range(n_methods):
                    if i == j:
                        continue

                    sub_counts = counts[[i,j],:]
                    _, p_val = chi_homogenity(sub_counts)
                    p_vals[i,j] = p_val

            p_val_vec  = p_val_matrix_to_vec(p_vals)
            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
            p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
            print("PVD:\n", p_val_df,file=report_handler)

    report_handler.close()
            
    report_all_path = os.path.join(results_dir, "Trend_stds_all.md")
    report_all_handler = open(report_all_path,"w")



    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name),file=report_all_handler)

        counts = np.zeros( (n_methods, 3) )

        sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods
        sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods
        # comm_sizes x (sets * classifiers)  x methods
        sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

        for method_id, method_name in enumerate( method_names ):
            for set_id in range( sub_results_concatenate.shape[1]):
                t_data = sub_results_concatenate[:,set_id, method_id]
                mk_result = mk.original_test(t_data)
                if mk_result.p <= alpha:
                    if mk_result.slope >0:
                        counts[method_id, 1]+=1
                    else:
                        counts[method_id, 2]+=1 
                else:
                    counts[method_id,0]+=1

        counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
        counts_pd.to_markdown(report_all_handler)
        print("\n", file=report_all_handler)
        p_vals = np.zeros( (n_methods, n_methods))
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    continue

                sub_counts = counts[[i,j],:]
                _, p_val = chi_homogenity(sub_counts)
                p_vals[i,j] = p_val

        p_val_vec  = p_val_matrix_to_vec(p_vals)
        p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
        p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
        print("PVD:\n", p_val_df,file=report_all_handler)

    report_all_handler.close()

def analyze_results_sizes_trends_stds_latex(results_dir="./results", alpha=0.05):

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".pickle") ]

    logging.info("Starting the results analysis")

    
    metrics_names = None
    base_classifiers_names = None
    method_names = None
    comm_sizes = None
    n_methods = None
    n_sets = len(result_files)

    overall_results = []

    for set_id, dump_file in enumerate(result_files):
        dump_file_basename = os.path.splitext(dump_file)[0]
        dump_file_path = os.path.join(results_dir, dump_file)


        logging.info("Processing: {}".format(dump_file_basename))
                    
        with open(dump_file_path,"rb") as fh:
            results_storage = pickle.load(fh)
                    
        metrics_names = results_storage[Dims.METRICS.value].values
        base_classifiers_names = results_storage[Dims.BASE_CLASSIFIERS.value].values
        method_names = results_storage[Dims.METHODS.value].values
        comm_sizes = results_storage[Dims.COMM_SIZES.value].values
        # comm_sizes x base_classifiers x methods x metrics x splits
        results = results_storage.transpose( Dims.COMM_SIZES.value, Dims.BASE_CLASSIFIERS.value,
                                Dims.METHODS.value, Dims.METRICS.value,
                                    Dims.FOLDS.value ).to_numpy()

        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

        overall_results.append( np.std(results,axis=-1) )


    overall_results  = np.asanyarray(overall_results) # sets x comm_sizes x base_classifiers x methods x metrics

    
    report_path = os.path.join(results_dir, "Trend_stds.tex")
    report_handler = open(report_path,"w")
    

    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name), file=report_handler)            
        for classifier_id, classifier_name in enumerate(base_classifiers_names):
            print("## {}".format(classifier_name), file=report_handler)

            #method , no, up , down
            counts = np.zeros( (n_methods, 3) )
            sub_results = overall_results[:,:,classifier_id,:,metric_id] # sets x com_sizes x methods

            for set_id in range(n_sets):
                for method_id, method_name in enumerate( method_names ):
                    sub_sub_results = sub_results[set_id,:, method_id] # comm_sizes
                    mk_result = mk.original_test(sub_sub_results)
                    if mk_result.p <= alpha:
                        if mk_result.slope >0:
                            counts[method_id, 1]+=1
                        else:
                            counts[method_id, 2]+=1 
                    else:
                        counts[method_id,0]+=1

            counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
            counts_pd.to_latex(report_handler)
            print("\n", file=report_handler)
            p_vals = np.zeros( (n_methods, n_methods))
            for i in range(n_methods):
                for j in range(n_methods):
                    if i == j:
                        continue

                    sub_counts = counts[[i,j],:]
                    _, p_val = chi_homogenity(sub_counts)
                    p_vals[i,j] = p_val

            p_val_vec  = p_val_matrix_to_vec(p_vals)
            p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
            p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )
            mic = pd.MultiIndex.from_arrays(  [
                                ["{}, {}".format(metric_name, classifier_name) for _ in range( n_methods -1)],
                                ["{}".format(s) for s in method_names[1:]]
                                ] )

            corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
            corr_p_val_matrix[np.tril_indices(corr_p_val_matrix.shape[0],k=-1)] = np.nan

            print("PVD:\n",file=report_handler)
            p_val_df = pd.DataFrame(corr_p_val_matrix[:-1,1:], columns=mic,index=method_names[:-1] )
            p_val_df.style.format(precision=3,na_rep="",escape="latex").highlight_between(left=0, right=0.05,  props='textbf:--rwrap;').\
                format_index(escape="latex",axis=0).format_index(escape="latex", axis=1).\
                to_latex(report_handler, multicol_align="c", hrules=False)

    report_handler.close()
            
    report_all_path = os.path.join(results_dir, "Trend_stds_all.tex")
    report_all_handler = open(report_all_path,"w")



    for metric_id, metric_name in enumerate(metrics_names):
        print("# {}".format(metric_name),file=report_all_handler)

        counts = np.zeros( (n_methods, 3) )

        sub_results = overall_results[:,:,:,:,metric_id] # sets x com_sizes x classifiers x methods
        sub_results_reorder = np.swapaxes(sub_results,1,2) #  sets x classifiers x comm_sizes x methods
        # comm_sizes x (sets * classifiers)  x methods
        sub_results_concatenate = np.swapaxes( np.concatenate(sub_results_reorder,axis=0),0,1 )

        for method_id, method_name in enumerate( method_names ):
            for set_id in range( sub_results_concatenate.shape[1]):
                t_data = sub_results_concatenate[:,set_id, method_id]
                mk_result = mk.original_test(t_data)
                if mk_result.p <= alpha:
                    if mk_result.slope >0:
                        counts[method_id, 1]+=1
                    else:
                        counts[method_id, 2]+=1 
                else:
                    counts[method_id,0]+=1

        counts_pd = pd.DataFrame(data=counts, index=method_names, columns=["No", "Up", "Down"])
        counts_pd.to_latex(report_all_handler)
        print("\n", file=report_all_handler)
        p_vals = np.zeros( (n_methods, n_methods))
        for i in range(n_methods):
            for j in range(n_methods):
                if i == j:
                    continue

                sub_counts = counts[[i,j],:]
                _, p_val = chi_homogenity(sub_counts)
                p_vals[i,j] = p_val

        p_val_vec  = p_val_matrix_to_vec(p_vals)
        p_val_vec_corrected = multipletests(p_val_vec, method='hommel')
        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
        p_val_df = pd.DataFrame(corr_p_val_matrix, columns=method_names,index=method_names )

        mic = pd.MultiIndex.from_arrays(  [
                ["{}".format(metric_name, classifier_name) for _ in range( n_methods -1)],
                ["{}".format(s) for s in method_names[1:]]
                ] )

        corr_p_val_matrix = p_val_vec_to_matrix(p_val_vec_corrected[1],n_methods)
        corr_p_val_matrix[np.tril_indices(corr_p_val_matrix.shape[0],k=-1)] = np.nan

        print("P-Vals: ", file=report_all_handler)

        p_val_df = pd.DataFrame(corr_p_val_matrix[:-1,1:], columns=mic,index=method_names[:-1] )
        p_val_df.style.format(precision=3,na_rep="",escape="latex").highlight_between(left=0, right=0.05,  props='textbf:--rwrap;').\
            format_index(escape="latex",axis=0).format_index(escape="latex", axis=1).\
            to_latex(report_all_handler, multicol_align="c", hrules=False)

    report_all_handler.close()

def convert_from_old(results_dir="./results"):

    logging.info("Starting the results conversion")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                if dump_file.endswith(".npy"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    logging.info("Processing: {}".format(dump_file_basename))

                    new_dump_file_path = os.path.join(dirpath, "{}.pickle".format(dump_file_basename))
                    
                    raw_data = np.load(dump_file_path,allow_pickle=True).item()
                    metrics_names = raw_data["metrics"]
                    base_classifiers_names = raw_data["base_classifiers"]
                    method_names = raw_data["methods"]
                    comm_sizes = raw_data["comm_sizes"]
                    results = raw_data["results"]
                    fold_idxs = [k for k in range(results.shape[-1])] 
                    # comm_sizes x base_classifiers x methods x metrics x splits

                    coords={
                        Dims.COMM_SIZES.value: comm_sizes,
                        Dims.BASE_CLASSIFIERS.value: base_classifiers_names,
                        Dims.METHODS.value: method_names,
                        Dims.METRICS.value: metrics_names,
                        Dims.FOLDS.value: fold_idxs,
                    }
                    results_storage = ResultsStorage.init_coords(coords=coords)
                    results_storage[:] = results
                    results_storage.name="Results"

                    logging.info("Storing: {}".format(new_dump_file_path))

                    with open(new_dump_file_path,"wb") as fh:
                        pickle.dump(results_storage,file=fh)
    

    pass

if __name__ == '__main__':
    
    sizes = [i for i in range(1,11,2)]
    sizes += [10*i + 1 for i in range(1,4)]
    res_dir = os.path.join( settings.RESULSTPATH, "boosting" )
    os.makedirs(res_dir,exist_ok=True)

    data_dir = settings.DATAPATH

    experiment_logger = ExperimentLogger(logging_dir=res_dir, 
                                         logging_level=logging.INFO,
                                         log_file_stump="Experiment_Boosting"
                                         )
    logger_obj = experiment_logger.start_logging()

    # mp.set_start_method('fork')
   
    convert_from_old(results_dir=res_dir)

    run_experiment(committee_sizes=sizes,results_dir=res_dir, overwrite=True,
                   data_dir=data_dir,experiment_logger=experiment_logger,
                    n_jobs=-1, n_splits=8, n_repeats=1, max_heap_size="1g", reset_counter=1000,
                     memory_percent_threshold=70)
    
    analysis_functions = [
        analyze_results,
        analyze_results_sizes,
        analyze_results_sizes_latex,
        analyze_results_sizes_ranks,
        analyze_results_sizes_ranks2,
        analyze_results_sizes_ranks_std,
        analyze_results_sizes_trends_means,
        analyze_results_sizes_trends_means_latex,
        analyze_results_sizes_trends_stds,
        analyze_results_sizes_trends_stds_latex,
    ]
    
    ProgressParallel(backend="multiprocessing",n_jobs=-1, desc="Analysis", total=len(analysis_functions), leave=False)\
                            (delayed(fun)(res_dir) for fun in  analysis_functions )

    experiment_logger.stop_logging()