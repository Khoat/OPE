#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

@author: doanphongtung

"""
import sys, os, shutil
import Streaming_FW

sys.path.insert(0, '../common')
import utilities
    
def main():
    # Check input
    if len(sys.argv) != 5:
        print"usage: python run_Streaming_FW.py [train file] [setting file] [model folder] [divided data folder]"
        exit()
    # Get environment variables
    train_file = sys.argv[1]
    setting_file = sys.argv[2]
    model_folder = sys.argv[3]
    divided_data_folder = sys.argv[4]
    tops = 10#int(sys.argv[5])    
    # Create model folder if it doesn't exist
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.makedirs(model_folder)
    # Read settings
    print'reading setting ...'
    ddict = utilities.read_setting(setting_file)
    print'write setting ...'
    file_name = '%s/setting.txt'%(model_folder)
    utilities.write_setting(ddict, file_name)
    # Read data for computing perplexities
    print'read data for computing perplexities ...'
    (corpusids_part1, corpuscts_part1, corpusids_part2, corpuscts_part2) = \
    utilities.read_data_for_perpl(divided_data_folder)
    # Initialize the algorithm
    print'initialize the algorithm ...'
    streaming_fw = Streaming_FW.StreamingFW(ddict['num_terms'], ddict['num_topics'], ddict['eta'], ddict['iter_infer'])
    # Start
    print'start!!!'
    i = 0
    while i < ddict['iter_train']:
        i += 1
        print'\n***iter_train:%d***\n'%(i)
        datafp = open(train_file, 'r')
        j = 0
        while True:
            j += 1
            (wordids, wordcts) = utilities.read_minibatch_list_frequencies(datafp, ddict['batch_size'])
            # Stop condition            
            if len(wordids) == 0:
                break
            #
            print'---num_minibatch:%d---'%(j)
            (time_e, time_m, theta) = streaming_fw.static_online(ddict['batch_size'], wordids, wordcts)
            # Compute sparsity
            sparsity = utilities.compute_sparsity(theta, theta.shape[0], theta.shape[1], 't')
            # Compute perplexities
            (LD2, ld2_list) = utilities.compute_perplexities_fw(streaming_fw._lambda, ddict['iter_infer'], \
                                         corpusids_part1, corpuscts_part1, corpusids_part2, corpuscts_part2)   
            # Search top words of each topics
            list_tops = utilities.list_top(streaming_fw._lambda, tops)    
            # Write files
            utilities.write_file(i, j, streaming_fw._lambda, time_e, time_m, theta, sparsity, LD2, ld2_list, list_tops, tops, model_folder)
        datafp.close()
    # Write final model to file
    file_name = '%s/lambda_final.dat'%(model_folder)
    utilities.write_topics(streaming_fw._lambda, file_name)
    # Finish
    print'done!!!'        
if __name__ == '__main__':
    main()