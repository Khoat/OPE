import sys
import string
import numpy as np
import per_vb
import per_fw

"""
    Read all documents in the file and stores terms and counts in lists.
"""
def read_data(filename):
    wordids = list()
    wordcts = list()
    fp = open(filename, 'r')
    while True:
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            break
        terms = string.split(line)
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype = np.int32)
        cts = np.zeros(doc_length, dtype = np.int32)
        for j in range(1, doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    fp.close()
    return(wordids, wordcts)
    
"""
    Read data for computing perplexities.
"""
def read_data_for_perpl(divided_data_folder):
    corpusids_part1 = list()
    corpuscts_part1 = list()
    corpusids_part2 = list()
    corpuscts_part2 = list()
    for i in range(5):
        filename_part1 = '%s/data_test_%d_part_1.txt'%(divided_data_folder, i+1)
        filename_part2 = '%s/data_test_%d_part_2.txt'%(divided_data_folder, i+1)
        (wordids_1, wordcts_1) = read_data(filename_part1)
        (wordids_2, wordcts_2) = read_data(filename_part2)
        corpusids_part1.append(wordids_1)
        corpuscts_part1.append(wordcts_1)
        corpusids_part2.append(wordids_2)
        corpuscts_part2.append(wordcts_2)
    return(corpusids_part1, corpuscts_part1, corpusids_part2, corpuscts_part2)

"""
    Read mini-batch and stores terms and counts in lists. 
"""
def read_minibatch_list_frequencies(fp, batch_size):
    wordids = list()
    wordcts = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        terms = string.split(line)
        doc_length = int(terms[0])
        ids = np.zeros(doc_length, dtype = np.int32)
        cts = np.zeros(doc_length, dtype = np.int32)
        for j in range(1,doc_length + 1):
            term_count = terms[j].split(':')
            ids[j - 1] = int(term_count[0])
            cts[j - 1] = int(term_count[1])
        wordids.append(ids)
        wordcts.append(cts)
    return(wordids, wordcts)
    
"""
    Read mini-batch and stores each document as a sequence of tokens (wordtks: token1 token2 ...). 
"""    
def read_minibatch_list_sequences(fp, batch_size):
    wordtks = list()
    lengths = list()
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            break
        tks = list()
        tokens = string.split(line)
        counts = int(tokens[0]) + 1
        for j in range(1, counts):
            token_count = tokens[j].split(':')
            token_count = map(int, token_count)
            for k in range(token_count[1]):
                tks.append(token_count[0])
        wordtks.append(tks)
        lengths.append(len(tks))
    return(wordtks, lengths)    
    
"""
    Read mini-batch and stores in dictionary (train_cts: (term:frequency)).
"""    
def read_minibatch_dict(fp, batch_size):
    train_cts = list()
    stop = 0
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 5:
            stop = 1
            break
        ids = list()
        cts = list()
        terms = string.split(line)
        for j in range(1,int(terms[0]) + 1):
            term_count = terms[j].split(':')
            ids.append(int(term_count[0]))
            cts.append(int(term_count[1]))
        ddict = dict(zip(ids, cts))
        train_cts.append(ddict)
    return(train_cts, stop)
    
"""
    Read setting file.
"""    
def read_setting(file_name):
    f = open(file_name, 'r')
    settings = f.readlines()
    f.close()
    sets = list()
    vals = list()
    for i in range(len(settings)):
        #print'%s\n'%(settings[i])
        if settings[i][0] == '#':
            continue
        set_val = settings[i].split(':')
        sets.append(set_val[0])
        vals.append(float(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['num_terms'] = int(ddict['num_terms'])
    ddict['num_topics'] = int(ddict['num_topics'])
    ddict['iter_train'] = int(ddict['iter_train'])
    ddict['iter_infer'] = int(ddict['iter_infer'])
    ddict['batch_size'] = int(ddict['batch_size'])
    return(ddict)
    
"""
    Compute perplexities, employing Variational Bayes.
"""
def compute_perplexities_vb(beta, alpha, eta, max_iter, corpusids_part1, 
                           corpuscts_part1, corpusids_part2, corpuscts_part2):
    vb = per_vb.VB(beta, alpha, eta, max_iter)
    LD2 = 0.
    ld2_list = list()
    for k in range(5):
        ld2 = vb.compute_perplexity(corpusids_part1[k], corpuscts_part1[k], corpusids_part2[k], corpuscts_part2[k])
        LD2 += ld2
        ld2_list.append(ld2)
    return(LD2 / 5, ld2_list)
    
"""
    Compute perplexities, employing Frank-Wolfe.
"""
def compute_perplexities_fw(beta, max_iter, corpusids_part1, corpuscts_part1, 
                            corpusids_part2, corpuscts_part2):
    fw = per_fw.FW(beta, max_iter)
    LD2 = 0.
    ld2_list = list()
    for k in range(5):
        ld2 = fw.compute_perplexity(corpusids_part1[k], corpuscts_part1[k], corpusids_part2[k], corpuscts_part2[k])
        LD2 += ld2
        ld2_list.append(ld2)
    return(LD2 / 5, ld2_list)

"""
    Compute document sparsity.
"""
def compute_sparsity(doc_tp, batch_size, num_topics, _type):
    sparsity = np.zeros(batch_size, dtype = np.float)
    if _type == 'z':
        for d in range(batch_size):
            N_z = np.zeros(num_topics, dtype = np.int)
            N = len(doc_tp[d])
            for i in xrange(N):
                N_z[doc_tp[d][i]] += 1.
            sparsity[d] = len(np.where(N_z != 0)[0])
    else:
        for d in range(batch_size):
            sparsity[d] = len(np.where(doc_tp[d] > 1e-10)[0])
    sparsity /= num_topics
    return(np.mean(sparsity))

"""
    Create list of top words of topics.
"""
def list_top(beta, tops):
    min_float = -sys.float_info.max
    num_tops = beta.shape[0]
    list_tops = list()
    for k in range(num_tops):
        top = list() 
        arr = np.array(beta[k,:], copy = True)
        for t in range(tops):
            index = arr.argmax()
            top.append(index)
            arr[index] = min_float
        list_tops.append(top)
    return(list_tops)

"""
-------------------------------------------------------------------------------
"""   
                
def write_topics(beta, file_name):
    num_terms = beta.shape[1]
    num_topics = beta.shape[0]
    f = open(file_name, 'w')
    for k in range(num_topics):
        for i in range(num_terms - 1):
            f.write('%.10f '%(beta[k][i]))
        f.write('%.10f\n'%(beta[k][num_terms - 1]))
    f.close()
    
def write_topic_mixtures(theta, file_name):
    batch_size = theta.shape[0]
    num_topics = theta.shape[1]
    f = open(file_name, 'a')
    for d in range(batch_size):
        for k in range(num_topics - 1):
            f.write('%.5f '%(theta[d][k]))
        f.write('%.5f\n'%(theta[d][num_topics - 1]))
    f.close()
    
def write_perplexities(LD2, ld2_list, model_folder):
    per_file_name = '%s/perplexities.csv'%(model_folder)
    f = open(per_file_name, 'a')
    f.writelines('%f,'%(LD2))
    f.close()
    per_pairs_file_name = '%s/perplexities_pairs.csv'%(model_folder)
    f = open(per_pairs_file_name, 'a')
    f.writelines('%f,' % ld2 for ld2 in ld2_list)
    f.writelines('\n')
    f.close()

def write_topic_top(list_tops, file_name):
    num_topics = len(list_tops)
    tops = len(list_tops[0])
    f = open(file_name, 'w')
    for k in range(num_topics):
        for j in range(tops - 1):
            f.write('%d '%(list_tops[k][j]))
        f.write('%d\n'%(list_tops[k][tops - 1]))
    f.close()
    
def write_sparsity(sparsity, file_name):
    f = open(file_name, 'a')
    f.write('%.10f,' % (sparsity))
    f.close()
    
def write_time(i, j, time_e, time_m, file_name):
    f = open(file_name, 'a')
    f.write('tloop_%d_iloop_%d, %f, %f, %f,\n'%(i, j, time_e, time_m, time_e + time_m))
    f.close()
    
def write_loop(i, j, file_name):
    f = open(file_name, 'w')
    f.write('%d, %d'%(i,j))
    f.close()
    
def write_setting(ddict, file_name):
    keys = ddict.keys()
    vals = ddict.values()
    f = open(file_name, 'w')
    for i in range(len(keys)):
        f.write('%s: %f\n'%(keys[i], vals[i]))
    f.close()
    
def write_file(i, j, beta, time_e, time_m, theta, sparsity, LD2, ld2_list, list_tops, tops, model_folder):
    beta_file_name = '%s/beta_%d_%d.dat'%(model_folder, i, j)
    theta_file_name = '%s/theta_%d.dat'%(model_folder, i)
    top_file_name = '%s/top%d_%d_%d.dat'%(model_folder, tops, i, j)
    spar_file_name = '%s/sparsity_%d.csv'%(model_folder, i)
    time_file_name = '%s/time_%d.csv'%(model_folder, i)
    loop_file_name = '%s/loops.csv'%(model_folder)
    """    
    # write beta    
    if j % 10 == 1:
        write_topics(beta, beta_file_name)
    # write theta
    write_topic_mixtures(theta, theta_file_name)
    """
    # write perplexities
    write_perplexities(LD2, ld2_list, model_folder)
    # write list top
    write_topic_top(list_tops, top_file_name)
    # write sparsity
    write_sparsity(sparsity, spar_file_name)
    # write time
    write_time(i, j, time_e, time_m, time_file_name)
    # write loop
    write_loop(i, j, loop_file_name)