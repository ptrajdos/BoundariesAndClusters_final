from datetime import datetime
import os
from typing import Any
import warnings
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

CLASSIFIERS_CONFIG_FILE_NAME_STUMP="exp_Boosting_Classifiers_{}.json"
METHOD_NAMES=["BCs1", "BCs2", "BC", "Ta", "Si"]

def generate_metrics():
    metrics = {
        "BAC":(balanced_accuracy_score,{}),
        "Kappa":(cohen_kappa_score,{}),
        "F1": (f1_score,{"average":"micro"}),
        "PRC": (precision_score,{"average":"micro"}),
        "REC": (recall_score, {"average":"micro"})
    }
    return metrics


def generate_configs(comm_sizes, n_base_classifiers,n_methods):
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
    

class WekaProcess(mp.Process):

    def __init__(self, group: None = None, target = None,
                  name: str  = None, args = {}, kwargs  = {}, *,
                    daemon: bool  = None,
                    X = None, y=None, input_queue=None,output_queue = None,
                      classpath=None, metrics=None, logging_queue=None,
                      max_heap_size=None) -> None:
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
                config = self.input_queue.get(timeout=3)

                if config is None:
                    self.logger.debug("PID {} Has acquired poison pill. ".format( mp.current_process().pid))
                    self.output_queue.close()
                    self.output_queue.join_thread()
                    self.logger.debug("PID {} stopping JVM".format( mp.current_process().pid))
                    jvm.stop()
                    break
                else:
                    method_description, train_index, test_index, split_idx, b_class_idx, meth_idx, com_size_idx = config

                    self.logger.debug("PID {} Acquired another job".format( mp.current_process().pid))

                    X_train, X_test = self.X[train_index], self.X[test_index]
                    y_train, y_test = self.y[train_index], self.y[test_index]

                    method = WekaEstimator(classname=method_description["classname"], options=method_description["options"])
                    self.logger.debug("PID {} Fit ".format( mp.current_process().pid))
                    method.fit(X_train, y_train)
                    self.logger.debug("PID {} Predict ".format( mp.current_process().pid))
                    y_pred = method.predict(X_test)
        
                    self.logger.debug("PID {} Putting results to output queue".format( mp.current_process().pid))

                    for metric_id, metric_name in enumerate(self.metrics):
                        metric_fun, metric_params = self.metrics[metric_name]

                        metric_val = metric_fun(y_test, y_pred, **metric_params)

                        self.output_queue.put( (split_idx, metric_id, b_class_idx, meth_idx, com_size_idx,metric_val ) )

            except qu.Empty:
                self.logger.debug("PID {} empty queue sleeping".format( mp.current_process().pid))
                time.sleep(random.random())
            except Exception as ex:
                self.logger.error("PID {} Exception in main process loop: {}".format( mp.current_process().pid, ex))
                self.logger.error("PID {} Emergency JVM stop".format( mp.current_process().pid))
                jvm.stop()


        self.logger.debug("PID {}: Stopping".format( mp.current_process().pid))
        return None
    
class LoggerProcess(mp.Process):
    def __init__(self, group = None, target= None, name = None, args={}, kwargs={}, *, daemon= None,
                 output_dir=None, logging_queue=None) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

        self.output_dir = output_dir
        self.logging_queue = logging_queue

    
    def run(self) -> None:

        date_string = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        log_file = os.path.join(self.output_dir,"Experiment_Boosting_{}.log".format(date_string))
        open(log_file,"w+").close()

        log_format_str = "%(asctime)s;%(levelname)s;%(message)s"
        log_date_format = "%Y-%m-%d %H:%M:%S"
        
        logging.basicConfig(filename=log_file, level=logging.INFO,force=True,format=log_format_str, datefmt=log_date_format)
        logging.captureWarnings(True)

        logger = logging.getLogger()

        while True:
            try:
                record = self.logging_queue.get(timeout=3)
                if record is None:
                    break
                logger.handle(record)
            except qu.Empty:
                time.sleep(random.random())

        return None
    
class ResultsGatheringProcess(mp.Process):
    def __init__(self, group: None = None, name: str = None, args = {}, kwargs= {}, *, daemon: bool = None,
                 results_queue=None, logging_queue=None, metrics=None, b_class_map=None, method_map=None,
                 committee_sizes=None, data_file_basename=None,results_dir=None, n_splits=1) -> None:
        super().__init__(group, None, name, args, kwargs, daemon=daemon)

        self.results_queue = results_queue
        self.logging_queue = logging_queue
        self.metrics = metrics
        self.b_class_map = b_class_map
        self.method_map = method_map
        self.committee_sizes = committee_sizes
        self.data_file_basename = data_file_basename
        self.results_dir = results_dir
        self.n_splits = n_splits

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

        n_metrics = len(self.metrics)
        n_comm_sizes = len(self.committee_sizes)
        n_base_classifiers = len( self.b_class_map )
        n_methods = len(self.method_map)

        # comm_sizes x base_classifiers x methods x metrics x splits
        results_array = np.zeros( (n_comm_sizes, n_base_classifiers, n_methods, n_metrics, self.n_splits) )

        to_be_gathered = n_comm_sizes * n_base_classifiers * n_methods * n_metrics * self.n_splits
        gathered = 0


        self.logger.debug("PID {}: Result Gathering Process entering main loop.".format( mp.current_process().pid))

        start_time = timer()
        
        while True:
            try:
                self.logger.debug("PID: {} Result gathering Process -- get from queue.".format(mp.current_process().pid))
                record = self.results_queue.get(timeout=3)

                if record is None:
                    self.logger.debug("PID: {} Quitting gathering Process.".format(mp.current_process().pid))
                    
                    all_results_dict = {
                    "results":results_array, 
                    "metrics":[k for k in self.metrics],
                    "base_classifiers":[k for k in self.b_class_map],
                    "methods":[k for k in self.method_map],
                    "comm_sizes": [k for k in self.committee_sizes], 
                    }

                    out_dump_filename=os.path.join(self.results_dir, "{}.npy".format(self.data_file_basename))
                    np.save(out_dump_filename, all_results_dict)
                    self.logger.info("PID: {}; Numeric results for {} has been saved.".format(mp.current_process().pid,
                                                                                         self.data_file_basename))
                    break

                gathered += 1
                self.logger.debug("PID: {}; Result Gathering Process -- Collecting another result ({}/{}): {}".format(mp.current_process().pid,gathered,to_be_gathered,record))

                curr_time = timer()
                self._calc_time(start_time, curr_time, to_be_gathered, gathered)

                split_idx, metric_id, b_class_idx, meth_idx, com_size_idx,metric_val = record
                results_array[com_size_idx, b_class_idx, meth_idx, metric_id, split_idx] = metric_val
                
            except qu.Empty:
                self.logger.debug("PID: {}; Result quque empty".format(mp.current_process().pid))
                time.sleep(random.random())
            except Exception as ex:
                self.logger.debug("PID: {}; Result gathering: unknown exception: {}".format(mp.current_process().pid,ex))

        self.logger.debug("PID: {}; Result gathering at exit".format(mp.current_process().pid))
        return None

class ExperimentLogger:
    def __init__(self, logging_dir="./",logging_level=logging.INFO):
        self.logging_dir = logging_dir
        self.logging_level = logging_level
        self.logging_queue = None
        self.logger = None
        self.logger_process = None

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
                                        logging_queue=self.logging_queue
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

def run_experiment(data_dir="./data", results_dir="./results", committee_sizes=[11,21], overwrite=False,
                   experiment_logger:ExperimentLogger=None,n_splits=10, n_repeats=1,
                   n_jobs = -1, max_heap_size=None):

    logger = experiment_logger.get_logger()
    logging_queue = experiment_logger.get_logging_queue()

    logger.info("Starting the experiment")
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info("Generate configs!")
    n_base_classifiers_val = mp.Value('i')
    n_methods_val = mp.Value('i')
    numbers_gen_process = mp.Process(target=generate_configs, args=(committee_sizes,n_base_classifiers_val,n_methods_val,))
    numbers_gen_process.start()
    numbers_gen_process.join()

    n_base_classifiers = n_base_classifiers_val.value
    n_methods = n_methods_val.value
    
    logger.info("Configs generated.")

    for (dirpath, dirnames, filenames) in os.walk(data_dir):
        for data_file in filenames:

            data_file_basename = os.path.splitext(data_file)[0]
            logger.info("Processing set: {}".format(data_file_basename))
            data_file_path = os.path.join(dirpath, data_file)
            out_dump_filename = os.path.join(results_dir, data_file_basename+".npy")

            if os.path.exists(out_dump_filename) and not overwrite:
                logger.info("Results for {} are present. Skipping".format(data_file_basename))
                continue

            logger.info("Starting experiment for: {}".format(data_file_basename))
            
            X, y, meta = load_arff(data_file_path, class_index="last")
            y = np.asanyarray(to_nominal_labels(y)) 
            X,y = shuffle(X,y,random_state=0)
            
            n_comm_sizes = len(committee_sizes)
            metrics = generate_metrics()
            n_metrics = len(metrics)

            b_class_map = {}
            method_map = {}


            skf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=0)
            n_total_splits = skf.get_n_splits()

            weka_processes = []
            input_queue = mp.Queue()
            
            output_queue = mp.Queue()

            
            result_process  = None
            

            n_processes = mp.cpu_count() if n_jobs <=0 else n_jobs
            
            for i in range(n_processes):
                 wp = WekaProcess(input_queue=input_queue, output_queue=output_queue, X=X, y=y,
                                   classpath=settings.CLASSPATH, metrics=metrics, logging_queue=logging_queue,
                                   max_heap_size=max_heap_size)
                 logger.debug("Before Starting WekaProcess {} for {}".format(wp,data_file_basename))
                 wp.start()
                 logger.debug("Started WekaProcess {} for {}".format(wp,data_file_basename))
                 weka_processes.append(wp) 

            for com_size_idx, com_size in  tqdm( enumerate( committee_sizes),desc="Committee sizes"):
                logger.info("Committee size for {}: {}".format(data_file_basename,com_size))
                
                config_file_path = os.path.join(settings.CONFIGPATH, CLASSIFIERS_CONFIG_FILE_NAME_STUMP.format(com_size))
                file_handler = open(config_file_path,"r")
                json_repr = json.load(file_handler)
                file_handler.close()
                
                methods = [ (r["name"], r) for r in json_repr ]
                if not b_class_map:
                    bc_idx = 0
                    meth_idx = 0
                    for m_name , _ in methods:
                        bc_name, meth_name  = m_name.split("_")

                        if bc_name not in b_class_map:
                            b_class_map[bc_name] = bc_idx
                            bc_idx += 1

                        if meth_name not in method_map:
                            method_map[meth_name] = meth_idx
                            meth_idx += 1

                if result_process is None:
                    result_process = ResultsGatheringProcess(
                        results_queue=output_queue,
                        logging_queue=logging_queue,
                        metrics=metrics,
                        b_class_map=b_class_map,
                        method_map=method_map,
                        committee_sizes=committee_sizes,
                        data_file_basename=data_file_basename,
                        results_dir=results_dir,
                        n_splits=n_total_splits,
                    )
                    logger.debug("Result gathering process is about to start, Set {} ".format(data_file_basename))
                    result_process.start()
                    logger.debug("Result gathering thread was started: {}".format(result_process))
                    
                    
                for met_name_full, method_description in tqdm(methods,leave=False,desc="Committees"):
                    b_class_name, meth_name = met_name_full.split("_")
                    b_class_idx, meth_idx = b_class_map[b_class_name], method_map[meth_name]

                    logger.info("Estimator: {} queued for {}; comm size {}".format(met_name_full, data_file_basename, com_size))

                    for split_idx, (train_index, test_index) in enumerate( skf.split(X, y)):
                        input_queue.put( (method_description,train_index, test_index, split_idx,
                                          b_class_idx, meth_idx, com_size_idx) )


            logger.debug("Set: {}, Sending poison pills for Weka Processes".format(data_file_basename))

            for _ in range(n_processes):
                input_queue.put(None)

            input_queue.close()

            logger.debug("{}: Waitign for processes to join".format(data_file_basename))
            
            for wp in weka_processes:
                wp.join()
                logger.debug("{} has joined".format(wp))

            logger.debug("{}: Sending poision pill to collecting process.".format(data_file_basename))

            output_queue.put(None)
            output_queue.close()
            
            logger.debug("{}: Waiting for result proces to join".format(data_file_basename))

            result_process.join()

            logger.debug("Result gathering process has been joined: {}".format(result_process))

    logger.info("Experiment ended!")
        
            

def analyze_results(results_dir="./results"):

    logging.info("Starting the results analysis")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                if dump_file.endswith(".npy"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    out_pdf_filename = os.path.join(results_dir, dump_file_basename+".pdf")
                    out_tex_directory = os.path.join(results_dir, dump_file_basename+"_tex")
                    os.makedirs(out_tex_directory,exist_ok=True)

                    logging.info("Processing: {}".format(dump_file_basename))
                    
                    raw_data = np.load(dump_file_path,allow_pickle=True).item()
                    metrics_names = raw_data["metrics"]
                    base_classifiers_names = raw_data["base_classifiers"]
                    method_names = raw_data["methods"]
                    comm_sizes = raw_data["comm_sizes"]
                    results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

                    method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
                    results = results[:,:,method_indices]

                    with PdfPages(out_pdf_filename) as pdf:

                        for metric_id, metric_name in enumerate(metrics_names):
                            for base_classifier_id, base_classifier_name in enumerate(base_classifiers_names):
                                for method_id, method_name in enumerate( method_names):
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

def analyze_results_sizes(results_dir="./results"):

    logging.info("Starting the results analysis")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                if dump_file.endswith(".npy"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    
                    out_tex_directory = os.path.join(results_dir, dump_file_basename+"_tex")
                    os.makedirs(out_tex_directory,exist_ok=True)

                    logging.info("Processing: {}".format(dump_file_basename))
                    
                    raw_data = np.load(dump_file_path,allow_pickle=True).item()
                    metrics_names = raw_data["metrics"]
                    base_classifiers_names = raw_data["base_classifiers"]
                    method_names = raw_data["methods"]
                    comm_sizes = raw_data["comm_sizes"]
                    results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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

def analyze_results_sizes_latex(results_dir="./results"):

    logging.info("Starting the results analysis")
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for dump_file in filenames:
                if dump_file.endswith(".npy"):
                    dump_file_basename = os.path.splitext(dump_file)[0]
                    dump_file_path = os.path.join(dirpath, dump_file)

                    
                    out_tex_directory = os.path.join(results_dir, dump_file_basename+"_tex")
                    os.makedirs(out_tex_directory,exist_ok=True)

                    logging.info("Processing: {}".format(dump_file_basename))
                    
                    raw_data = np.load(dump_file_path,allow_pickle=True).item()
                    metrics_names = raw_data["metrics"]
                    base_classifiers_names = raw_data["base_classifiers"]
                    method_names = raw_data["methods"]
                    comm_sizes = raw_data["comm_sizes"]
                    results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        n_base_classifiers = len(base_classifiers_names)
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        n_comm_sizes = len(comm_sizes)
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

        method_indices, method_names = reorder_names(method_names, METHOD_NAMES)
        results = results[:,:,method_indices]

        n_methods = len(method_names)

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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits


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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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

    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npy") ]

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
                    
        raw_data = np.load(dump_file_path,allow_pickle=True).item()
        metrics_names = raw_data["metrics"]
        base_classifiers_names = raw_data["base_classifiers"]
        method_names = raw_data["methods"]
        comm_sizes = raw_data["comm_sizes"]
        results = raw_data["results"] # comm_sizes x base_classifiers x methods x metrics x splits

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



if __name__ == '__main__':
    
    sizes = [i for i in range(1,11,2)]
    sizes += [10*i + 1 for i in range(1,4)]
    res_dir = os.path.join( settings.RESULSTPATH, "boosting" )
    os.makedirs(res_dir,exist_ok=True)

    data_dir = settings.DATAPATH

    experiment_logger = ExperimentLogger(logging_dir=res_dir, logging_level=logging.INFO)
    logger_obj = experiment_logger.start_logging()
   

    run_experiment(committee_sizes=sizes,results_dir=res_dir, overwrite=True,
                   data_dir=data_dir,experiment_logger=experiment_logger,
                    n_jobs=-1, n_splits=8, n_repeats=1, max_heap_size="1g" )
    analyze_results(results_dir=res_dir)
    analyze_results_sizes(results_dir=res_dir)
    analyze_results_sizes_latex(results_dir=res_dir)
    analyze_results_sizes_ranks(results_dir=res_dir,alpha=0.05)
    analyze_results_sizes_ranks2(results_dir=res_dir,alpha=0.05)
    analyze_results_sizes_ranks_std(results_dir=res_dir,alpha=0.05)
    analyze_results_sizes_trends_means(results_dir=res_dir,alpha=0.05)
    analyze_results_sizes_trends_means_latex(results_dir=res_dir,alpha=0.05)
    analyze_results_sizes_trends_stds(results_dir=res_dir,alpha=0.05)
    analyze_results_sizes_trends_stds_latex(results_dir=res_dir,alpha=0.05)

    experiment_logger.stop_logging()