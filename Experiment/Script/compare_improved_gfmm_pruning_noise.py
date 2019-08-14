# Run 10 times with different input

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.improvedonlinegfmm import ImprovedOnlineGFMM
from GFMM.faster_onlinegfmm import OnlineGFMM
import time
import random
import numpy as np
from functionhelper.preprocessinghelper import loadDataset


if __name__ == '__main__':
    
    save_imponline_gfmm_prob_result_folder_path = root_path + '/Experiment/modified_online_gfmm/noise/improved_online_gfmm/15_percent/teta_0_7/'
    save_online_gfmm_result_folder_path = root_path + '/Experiment/modified_online_gfmm/noise/original_online_gfmm/15_percent/teta_0_7/'
    save_online_gfmm_manhattan_result_folder_path = root_path + '/Experiment/modified_online_gfmm/noise/online_gfmm_manhattan/15_percent/teta_0_7/'
    
    dataset_path = root_path + '/Dataset/train_test/dps/'
    
    dataset_names = ['blood_transfusion_dps', 'BreastCancerCoimbra_dps', 'haberman_dps', 'heart_dps', 'page_blocks_dps', 'landsat_satellite_dps', 'waveform_dps', 'yeast_dps']
	
    teta = 0.7
    percent_noise = 15
    
    fold_index = np.array([1, 2, 3, 4])
    
    for dt in range(len(dataset_names)):
        #try:
        print('Current dataset: ', dataset_names[dt])
        fold1File = dataset_path + dataset_names[dt] + '_1.dat'
        fold2File = dataset_path + dataset_names[dt] + '_2.dat'
        fold3File = dataset_path + dataset_names[dt] + '_3.dat'
        fold4File = dataset_path + dataset_names[dt] + '_4.dat'
        
        # Read data file
        fold1Data, _, fold1Label, _ = loadDataset(fold1File, 1, False)
        fold2Data, _, fold2Label, _ = loadDataset(fold2File, 1, False)
        fold3Data, _, fold3Label, _ = loadDataset(fold3File, 1, False)
        fold4Data, _, fold4Label, _ = loadDataset(fold4File, 1, False)
        
        numhyperbox_imponline_gfmm_prob_save = np.array([])
        numhyperbox_before_prun_imponline_gfmm_prob_save = np.array([])
        training_time_imponline_gfmm_prob_save = np.array([])
        testing_error_imponline_gfmm_prob_save = np.array([])
        testing_error_before_prun_imponline_gfmm_prob_save = np.array([])
        
        numhyperbox_online_gfmm_save = np.array([])
        numhyperbox_online_gfmm_manhattan_save = np.array([])
        numhyperbox_before_prun_online_gfmm_save = np.array([])
        training_time_online_gfmm_save = np.array([])
        training_time_online_gfmm_manhattan_save = np.array([])
        testing_error_online_gfmm_save = np.array([])
        testing_error_before_prun_online_gfmm_save = np.array([])
        testing_error_online_gfmm_manhattan_save = np.array([])
        testing_error_before_prun_online_gfmm_manhattan_save = np.array([])
        
        label_set = np.unique(fold1Label)
        # loop through 4 folds
        for fo in range(4):
            if fo == 0:
                # fold 1 is testing set
                trainingData = np.vstack((fold2Data, fold3Data))
                testingData = fold1Data
                validationData = fold4Data
                
                f2lb = fold2Label.copy()
                f3lb = fold3Label.copy()
                f4lb = fold4Label.copy()
                    
                testingLabel = fold1Label
                numTestSample = len(testingLabel)
            elif fo == 1:
                # fold 2 is testing set
                trainingData = np.vstack((fold3Data, fold4Data))
                testingData = fold2Data
                validationData = fold1Data
                
                f2lb = fold3Label.copy()
                f3lb = fold4Label.copy()
                f4lb = fold1Label.copy()
                
                testingLabel = fold2Label
                numTestSample = len(testingLabel)
            elif fo == 2:
                # fold 3 is testing set
                trainingData = np.vstack((fold4Data, fold1Data))
                testingData = fold3Data
                validationData = fold2Data
                
                f2lb = fold4Label.copy()
                f3lb = fold1Label.copy()
                f4lb = fold2Label.copy()
                
                testingLabel = fold3Label
                numTestSample = len(testingLabel)
            else:
                # fold 4 is testing set
                trainingData = np.vstack((fold1Data, fold2Data))
                testingData = fold4Data
                validationData = fold3Data
                
                f2lb = fold1Label.copy()
                f3lb = fold2Label.copy()
                f4lb = fold3Label.copy()
                
                testingLabel = fold4Label
                numTestSample = len(testingLabel)  
                
            if percent_noise > 0:
                num_noise_lb2 = int(percent_noise * len(f2lb) / 100)
                num_noise_lb3 = int(percent_noise * len(f3lb) / 100)
                num_noise_lb4 = int(percent_noise * len(f4lb) / 100)
                selected_id_lb2 = random.sample(range(len(f2lb)), num_noise_lb2)
                selected_id_lb3 = random.sample(range(len(f3lb)), num_noise_lb3)
                selected_id_lb4 = random.sample(range(len(f4lb)), num_noise_lb4)
                
                for id_mut in selected_id_lb2:
                    f2lb[id_mut] = random.choice(label_set[label_set != f2lb[id_mut]])
                    
                for id_mut in selected_id_lb3:
                    f3lb[id_mut] = random.choice(label_set[label_set != f3lb[id_mut]])
                    
                for id_mut in selected_id_lb4:
                    f4lb[id_mut] = random.choice(label_set[label_set != f4lb[id_mut]])
            
            trainingLabel = np.hstack((f2lb, f3lb))
            validationLabel = f4lb
            
            # improved online GFMM
            impOnlnClassifier = ImprovedOnlineGFMM(gamma = 1, teta = teta, sigma = 1 - teta, isDraw = False, oper = 'min', isNorm = False)
            impOnlnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            numhyperbox_before_prun_imponline_gfmm_prob_save = np.append(numhyperbox_before_prun_imponline_gfmm_prob_save, len(impOnlnClassifier.classId)) 
            
            result_prob = impOnlnClassifier.predict(testingData, testingData, testingLabel, True)
            if result_prob != None:
                err = np.round(result_prob.summis / numTestSample * 100, 3)
                testing_error_before_prun_imponline_gfmm_prob_save = np.append(testing_error_before_prun_imponline_gfmm_prob_save, err)   
                
            start_t = time.perf_counter()
            impOnlnClassifier.pruning_val(validationData, validationData, validationLabel, True)
            end_t = time.perf_counter()
            
            training_time_imponline_gfmm_prob_save = np.append(training_time_imponline_gfmm_prob_save, impOnlnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_imponline_gfmm_prob_save = np.append(numhyperbox_imponline_gfmm_prob_save, len(impOnlnClassifier.classId))  
            
            result_prob = impOnlnClassifier.predict(testingData, testingData, testingLabel, True)
            if result_prob != None:
                err = np.round(result_prob.summis / numTestSample * 100, 3)
                testing_error_imponline_gfmm_prob_save = np.append(testing_error_imponline_gfmm_prob_save, err)
            
            # online GFMM
            onlnClassifier = OnlineGFMM(gamma = 1, teta = teta, tMin = teta, isDraw = False, oper = 'min', isNorm = False)
            onlnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            numhyperbox_before_prun_online_gfmm_save = np.append(numhyperbox_before_prun_online_gfmm_save, len(onlnClassifier.classId)) 
            result_no_manhat = onlnClassifier.predict(testingData, testingData, testingLabel, False)
            if result_no_manhat != None:
                err = np.round(result_no_manhat.summis / numTestSample * 100, 3)
                testing_error_before_prun_online_gfmm_save = np.append(testing_error_before_prun_online_gfmm_save, err)            
            
            start_t = time.perf_counter()
            onlnClassifier.pruning_val(validationData, validationData, validationLabel, False)
            end_t = time.perf_counter()
            
            training_time_online_gfmm_save = np.append(training_time_online_gfmm_save, onlnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_online_gfmm_save = np.append(numhyperbox_online_gfmm_save, len(onlnClassifier.classId))            
                
            result_no_manhat = onlnClassifier.predict(testingData, testingData, testingLabel, False)
            if result_no_manhat != None:
                err = np.round(result_no_manhat.summis / numTestSample * 100, 3)
                testing_error_online_gfmm_save = np.append(testing_error_online_gfmm_save, err)
            
            # using manhattan
            onlnClassifier = OnlineGFMM(gamma = 1, teta = teta, tMin = teta, isDraw = False, oper = 'min', isNorm = False)
            onlnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            result_manhat = onlnClassifier.predict(testingData, testingData, testingLabel, True)
            if result_manhat != None:
                err = np.round(result_manhat.summis / numTestSample * 100, 3)
                testing_error_before_prun_online_gfmm_manhattan_save = np.append(testing_error_before_prun_online_gfmm_manhattan_save, err)
                
            start_t = time.perf_counter()
            onlnClassifier.pruning_val(validationData, validationData, validationLabel, True)
            end_t = time.perf_counter()
            
            training_time_online_gfmm_manhattan_save = np.append(training_time_online_gfmm_manhattan_save, onlnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_online_gfmm_manhattan_save = np.append(numhyperbox_online_gfmm_manhattan_save, len(onlnClassifier.classId))            
                
            result_manhat = onlnClassifier.predict(testingData, testingData, testingLabel, True)
            if result_manhat != None:
                err = np.round(result_manhat.summis / numTestSample * 100, 3)
                testing_error_online_gfmm_manhattan_save = np.append(testing_error_online_gfmm_manhattan_save, err)
                
#        # save improved online gfmm result to file
        data_imponline_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_imponline_gfmm_prob_save.reshape(-1, 1), numhyperbox_imponline_gfmm_prob_save.reshape(-1, 1), training_time_imponline_gfmm_prob_save.reshape(-1, 1), testing_error_before_prun_imponline_gfmm_prob_save.reshape(-1, 1), testing_error_imponline_gfmm_prob_save.reshape(-1, 1)))
        filename_imponline = save_imponline_gfmm_prob_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_imponline, 'w').close() # make existing file empty
        
        with open(filename_imponline,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_imponline_save, fmt='%s', delimiter=', ')
        
        # save online gfmm
        data_online_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_online_gfmm_save.reshape(-1, 1), numhyperbox_online_gfmm_save.reshape(-1, 1), training_time_online_gfmm_save.reshape(-1, 1), testing_error_before_prun_online_gfmm_save.reshape(-1, 1), testing_error_online_gfmm_save.reshape(-1, 1)))
        filename_online = save_online_gfmm_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_online_save, fmt='%s', delimiter=', ')
        
        data_online_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_online_gfmm_save.reshape(-1, 1), numhyperbox_online_gfmm_manhattan_save.reshape(-1, 1), training_time_online_gfmm_manhattan_save.reshape(-1, 1), testing_error_before_prun_online_gfmm_manhattan_save.reshape(-1, 1), testing_error_online_gfmm_manhattan_save.reshape(-1, 1)))
        filename_online = save_online_gfmm_manhattan_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_online_save, fmt='%s', delimiter=', ')
        
        
#        except:
#            pass
        
    print('---Finish---')