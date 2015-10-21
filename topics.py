#! /usr/bin/python

# usage: python topics.py [beta-file] [vocab-file] [num words] [result-file]
#
# [beta-file] is output from the dln-c code
# [vocab-file] is a list of words, one per line
# [num words] is the number of words to print from each topic
# [result-file] is file to write top [num words] words 

import sys
import math

def print_topics(beta_file, vocab_file, nwords, result_file):

    # get the vocabulary

    vocab = file(vocab_file, 'r').readlines()
    # vocab = map(lambda x: x.split()[0], vocab)
    vocab = map(lambda x: x.strip(), vocab)

    # for each line in the mu file

    indices = range(len(vocab))
    
    # open file to write
    
    fp = open(result_file, 'w')
    topic_no = 0
    for topic in file(beta_file, 'r'):
        fp.write('topic %03d\n' % (topic_no))
        topic = map(float, topic.split())
        indices.sort(lambda x,y: -cmp(topic[x], topic[y]))
        for i in range(nwords):
            fp.write ('   %s \t\t %f\n' % (vocab[indices[i]], topic[indices[i]]))
        topic_no = topic_no + 1
        fp.write( '\n')
    fp.close()

if (__name__ == '__main__'):

    if (len(sys.argv) != 5):
       print 'usage: python topics.py [beta-file] [vocab-file] [num words] [result-file]\n'
       sys.exit(1)

    beta_file = sys.argv[1]
    vocab_file = sys.argv[2]
    nwords = int(sys.argv[3])
    result_file = sys.argv[4]
    print_topics(beta_file, vocab_file, nwords, result_file)
