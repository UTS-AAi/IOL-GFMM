# Run 10 times with different input

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.improvedonlinegfmm import ImprovedOnlineGFMM
from GFMM.faster_onlinegfmm import OnlineGFMM
import numpy as np
from functionhelper.preprocessinghelper import loadDataset


if __name__ == '__main__':
    
    save_imponline_gfmm_result_folder_path = root_path + '/Experiment/modified_online_gfmm/no_pruning/improved_online_gfmm/teta_0_7/'
    save_online_gfmm_result_folder_path = root_path + '/Experiment/modified_online_gfmm/no_pruning/original_online_gfmm/teta_0_7/'
    
    save_imponline_gfmm_prob_result_folder_path = root_path + '/Experiment/modified_online_gfmm/no_pruning/improved_online_gfmm_probability/teta_0_7/'
    save_online_gfmm_manhattan_result_folder_path = root_path + '/Experiment/modified_online_gfmm/no_pruning/original_online_gfmm_manhattan/teta_0_7/'
   
    dataset_path = root_path + '/Dataset/train_test/dps/'
    
    dataset_names = ['blood_transfusion_dps', 'BreastCancerCoimbra_dps', 'haberman_dps', 'heart_dps', 'page_blocks_dps', 'landsat_satellite_dps', 'waveform_dps', 'yeast_dps']
	
    teta = 0.7
    
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
        
        numhyperbox_imponline_gfmm_save = np.array([])
        training_time_imponline_gfmm_save = np.array([])
        testing_error_imponline_gfmm_save = np.array([])
        testing_error_imponline_gfmm_prob_save = np.array([])
        num_sample_in_boundary_imponline_gfmm_prob_save = np.array([])
        
        numhyperbox_online_gfmm_save = np.array([])
        training_time_online_gfmm_save = np.array([])
        testing_error_online_gfmm_save = np.array([])
        testing_error_online_gfmm_manhattan_save = np.array([])
        num_sample_in_boundary_online_gfmm_manhattan_save = np.array([])
        
        # loop through 4 folds
        for fo in range(4):
            if fo == 0:
                # fold 1 is testing set
                trainingData = np.vstack((fold2Data, fold3Data, fold4Data))
                testingData = fold1Data
                trainingLabel = np.hstack((fold2Label, fold3Label, fold4Label))
                testingLabel = fold1Label
                numTestSample = len(testingLabel)
            elif fo == 1:
                # fold 2 is testing set
                trainingData = np.vstack((fold3Data, fold4Data, fold1Data))
                testingData = fold2Data
                trainingLabel = np.hstack((fold3Label, fold4Label, fold1Label))
                testingLabel = fold2Label
                numTestSample = len(testingLabel)
            elif fo == 2:
                # fold 3 is testing set
                trainingData = np.vstack((fold4Data, fold1Data, fold2Data))
                testingData = fold3Data
                trainingLabel = np.hstack((fold4Label, fold1Label, fold2Label))
                testingLabel = fold3Label
                numTestSample = len(testingLabel)
            else:
                # fold 4 is testing set
                trainingData = np.vstack((fold1Data, fold2Data, fold3Data))
                testingData = fold4Data
                trainingLabel = np.hstack((fold1Label, fold2Label, fold3Label))
                testingLabel = fold4Label
                numTestSample = len(testingLabel)                
        
            # improved online GFMM
            impOnlnClassifier = ImprovedOnlineGFMM(gamma = 1, teta = teta, sigma = 1 - teta, isDraw = False, oper = 'min', isNorm = False)
            impOnlnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            training_time_imponline_gfmm_save = np.append(training_time_imponline_gfmm_save, impOnlnClassifier.elapsed_training_time)
            numhyperbox_imponline_gfmm_save = np.append(numhyperbox_imponline_gfmm_save, len(impOnlnClassifier.classId))            
                
            result_no_prob = impOnlnClassifier.predict(testingData, testingData, testingLabel, False)
            if result_no_prob != None:
                err = np.round(result_no_prob.summis / numTestSample * 100, 3)
                testing_error_imponline_gfmm_save = np.append(testing_error_imponline_gfmm_save, err)
            
            result_prob = impOnlnClassifier.predict(testingData, testingData, testingLabel, True)
            if result_prob != None:
                err = np.round(result_prob.summis / numTestSample * 100, 3)
                testing_error_imponline_gfmm_prob_save = np.append(testing_error_imponline_gfmm_prob_save, err)
                num_sample_in_boundary_imponline_gfmm_prob_save = np.append(num_sample_in_boundary_imponline_gfmm_prob_save, result_prob.numSampleInBoundary) 
            
            # online GFMM
            onlnClassifier = OnlineGFMM(gamma = 1, teta = teta, tMin = teta, isDraw = False, oper = 'min', isNorm = False)
            onlnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            training_time_online_gfmm_save = np.append(training_time_online_gfmm_save, onlnClassifier.elapsed_training_time)
            numhyperbox_online_gfmm_save = np.append(numhyperbox_online_gfmm_save, len(onlnClassifier.classId))            
                
            result_no_manhat = onlnClassifier.predict(testingData, testingData, testingLabel, False)
            if result_no_manhat != None:
                err = np.round(result_no_manhat.summis / numTestSample * 100, 3)
                testing_error_online_gfmm_save = np.append(testing_error_online_gfmm_save, err)
            
            # using manhattan
            result_manhat = onlnClassifier.predict(testingData, testingData, testingLabel, True)
            if result_manhat != None:
                err = np.round(result_manhat.summis / numTestSample * 100, 3)
                testing_error_online_gfmm_manhattan_save = np.append(testing_error_online_gfmm_manhattan_save, err)
                num_sample_in_boundary_online_gfmm_manhattan_save = np.append(num_sample_in_boundary_online_gfmm_manhattan_save, result_manhat.numSampleInBoundary)
                
#        # save improved online gfmm result to file
        data_imponline_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_imponline_gfmm_save.reshape(-1, 1), training_time_imponline_gfmm_save.reshape(-1, 1), testing_error_imponline_gfmm_save.reshape(-1, 1)))
        filename_imponline = save_imponline_gfmm_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_imponline, 'w').close() # make existing file empty
        
        with open(filename_imponline,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_imponline_save, fmt='%s', delimiter=', ')
        
        data_imponline_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_imponline_gfmm_save.reshape(-1, 1), training_time_imponline_gfmm_save.reshape(-1, 1), testing_error_imponline_gfmm_prob_save.reshape(-1, 1), num_sample_in_boundary_imponline_gfmm_prob_save.reshape(-1, 1)))
        filename_imponline = save_imponline_gfmm_prob_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_imponline, 'w').close() # make existing file empty
        
        with open(filename_imponline,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, No samples in boundary\n')
            np.savetxt(f_handle, data_imponline_save, fmt='%s', delimiter=', ')
        
        # save online gfmm
        data_online_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_online_gfmm_save.reshape(-1, 1), training_time_online_gfmm_save.reshape(-1, 1), testing_error_online_gfmm_save.reshape(-1, 1)))
        filename_online = save_online_gfmm_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error\n')
            np.savetxt(f_handle, data_online_save, fmt='%s', delimiter=', ')
        
        data_online_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_online_gfmm_save.reshape(-1, 1), training_time_online_gfmm_save.reshape(-1, 1), testing_error_online_gfmm_manhattan_save.reshape(-1, 1), num_sample_in_boundary_online_gfmm_manhattan_save.reshape(-1, 1)))
        filename_online = save_online_gfmm_manhattan_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, No samples in boundary\n')
            np.savetxt(f_handle, data_online_save, fmt='%s', delimiter=', ')
        
        
#        except:
#            pass
        
    print('---Finish---')