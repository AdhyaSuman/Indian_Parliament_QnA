import numpy as np
from scipy.io import savemat
import gensim.corpora as corpora
import gensim
import os
import random
import pickle
import collections
from octis.dataset.dataset import Dataset

def pickling(data, path):
    location = open(path,'wb')
    pickle.dump(data, location)
    location.close()

def for_detm(data_load_dir, save_dir, unprep_path, for_sg_path, min_count=5, threshold=20, verbose=True, seed=1):
    random.seed(seed)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_load_dir)

    # Build the bigram model
    unprep = [line.strip() for line in open(unprep_path, 'r').readlines()]
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))# deacc=True removes punctuations

    unprep_corpus = list(sent_to_words(unprep))
    bigram = gensim.models.Phrases(unprep_corpus, min_count=min_count, threshold=threshold) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    bigram_corpus = make_bigrams(dataset.get_corpus())
    #get the sorted copus and the corresponding timestamps:   
    indexes = [int(line.strip()) for line in open(data_load_dir + 'indexes.txt', 'r').readlines()]
    corpus = [bigram_corpus[indexes.index(i)] for i in sorted(indexes)]
    tstamp = [dataset.get_labels()[indexes.index(i)] for i in sorted(indexes)]
    tslice = list(collections.Counter(tstamp).values())
    tslice_cum = list(np.cumsum(tslice))

    train_corpus = []
    test_corpus = []
    validation_corpus = []
    tstamp_tr = []
    tstamp_ts = []
    tstamp_va = []
    for s,e in zip([0]+tslice_cum, tslice_cum):
        values = random.sample(range(s, e), e-s)
        train_ind, test_ind, validation_ind = np.split(values, [int(.8*len(values)), int(.9*len(values))])

        train_corpus = train_corpus + ([corpus[ind] for ind in train_ind])
        tstamp_tr = tstamp_tr + [tstamp[ind] for ind in train_ind]

        test_corpus = test_corpus + ([corpus[ind] for ind in test_ind])
        tstamp_ts = tstamp_ts + [tstamp[ind] for ind in test_ind]

        validation_corpus = validation_corpus + ([corpus[ind] for ind in validation_ind])
        tstamp_va = tstamp_va + [tstamp[ind] for ind in validation_ind]
    
    #removing the words that are not in train_corpus
    id2word = corpora.Dictionary(train_corpus)
    vocab = list(id2word.token2id.keys())
    if verbose:
        print('vocabulary after removing words not in train: {}'.format(len(vocab)))
    test_corpus = [[w for w in test_corpus[ind] if w in vocab] for ind in range(len(test_corpus))]
    validation_corpus = [[w for w in validation_corpus[ind] if w in vocab] for ind in range(len(validation_corpus))]
    
    with open(for_sg_path + 'for_sg.txt', 'w') as f:
        for text in [' '.join(i) for i in corpus]:
            f.write("%s\n" % text)

    #pickling of tstamps:
    pickling(tstamp, save_dir + 'timestamps.pkl')

    with open(save_dir + 'timestamps.txt', "w") as f:
        for item in tstamp:
            f.write("%s\n" % item)

    savemat(save_dir + 'bow_tr_timestamps.mat', {'timestamps': tstamp_tr}, do_compression=True)
    savemat(save_dir + 'bow_ts_timestamps.mat', {'timestamps': tstamp_ts}, do_compression=True)
    savemat(save_dir + 'bow_va_timestamps.mat', {'timestamps': tstamp_va}, do_compression=True)
    
    # Split test set in 2 halves
    if verbose:
        print('splitting test documents in 2 halves...')
    test_corpus_h1 = [[w for i,w in enumerate(doc) if i<=len(doc)/2.0-1] for doc in test_corpus]
    test_corpus_h2 = [[w for i,w in enumerate(doc) if i>len(doc)/2.0-1] for doc in test_corpus]

    def split_bow(corpus):
        tokens = []
        counts = []
        for text in corpus:
            token = []
            count = []
            for token_id, cnt in id2word.doc2bow(text):
                token.append(token_id)
                count.append(cnt)
            tokens.append(token)
            counts.append(count)
        return tokens, counts
    
    tokens_tr, counts_tr = split_bow(train_corpus)
    tokens_ts, counts_ts = split_bow(test_corpus)
    tokens_va, counts_va = split_bow(validation_corpus)
    tokens_ts_h1, counts_ts_h1 = split_bow(test_corpus_h1)
    tokens_ts_h2, counts_ts_h2 = split_bow(test_corpus_h2)

    savemat(save_dir + 'bow_tr_tokens.mat', {'tokens': tokens_tr}, do_compression=True)
    savemat(save_dir + 'bow_tr_counts.mat', {'counts': counts_tr}, do_compression=True)

    savemat(save_dir + 'bow_ts_tokens.mat', {'tokens': tokens_ts}, do_compression=True)
    savemat(save_dir + 'bow_ts_counts.mat', {'counts': counts_ts}, do_compression=True)

    savemat(save_dir + 'bow_ts_h1_tokens.mat', {'tokens': tokens_ts_h1}, do_compression=True)
    savemat(save_dir + 'bow_ts_h1_counts.mat', {'counts': counts_ts_h1}, do_compression=True)

    savemat(save_dir + 'bow_ts_h2_tokens.mat', {'tokens': tokens_ts_h2}, do_compression=True)
    savemat(save_dir + 'bow_ts_h2_counts.mat', {'counts': counts_ts_h2}, do_compression=True)

    savemat(save_dir + 'bow_va_tokens.mat', {'tokens': tokens_va}, do_compression=True)
    savemat(save_dir + 'bow_va_counts.mat', {'counts': counts_va}, do_compression=True)
    
    pickling(vocab, save_dir + 'vocab.pkl') 
    pickling(train_corpus, save_dir+'train_corpus.pkl')
    pickling(test_corpus, save_dir+'test_corpus.pkl')
    with open(save_dir + 'vocab.txt', 'w') as outfile:
        outfile.write("\n".join(vocab))


def for_dlda(data_load_dir, save_dir, unprep_path, verbose=True, min_count=5,
             threshold=20, add_val=False, add_test=False, seed = 2021):
    """
    Saves the following files:
    ----------
    train_bow, test_bow : {iterable of list of (int, float), scipy.sparse.csc}, optional
        Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`).
        If not given, the model is left untrained (presumably because you want to call
        :meth:`~gensim.models.ldamodel.LdaSeqModel.update` manually).
    time_slice : list of int, optional
        Number of documents in each time-slice. Each time slice could for example represent a year's published
        papers, in case the corpus comes from a journal publishing over multiple years.
        It is assumed that `sum(time_slice) == num_documents`.
    id2word : dict of (int, str), optional
        Mapping from word IDs to words. It is used to determine the vocabulary size, as well as for
        debugging and topic printing.
    lda_model : :class:`~gensim.models.ldamodel.LdaModel`
        Model whose sufficient statistics will be used to initialize the current object if `initialize == 'gensim'`.
    
    Parameters
    ----------
    data_load_dir : {str} path to load the dataset
    save_dir : {str} path to save the files
    verbose : {bool} If True some steps will be printed
    min_count : {int}
    threshold : {int}
    add_val : {bool}
    add_test : {bool}
    """
    random.seed(seed)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(data_load_dir)
    # Build the bigram model
    unprep = [line.strip() for line in open(unprep_path, 'r').readlines()]
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

    unprep_corpus = list(sent_to_words(unprep))
    bigram = gensim.models.Phrases(unprep_corpus, min_count=min_count, threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]
    bigram_corpus = make_bigrams(dataset.get_corpus())
    #get the sorted copus and the corresponding timestamps:   
    indexes = [int(line.strip()) for line in open(data_load_dir+'indexes.txt', 'r').readlines()]
    corpus = [bigram_corpus[indexes.index(i)] for i in sorted(indexes)]
    tstamp = [dataset.get_labels()[indexes.index(i)] for i in sorted(indexes)]
    tslice = list(collections.Counter(tstamp).values())
    tslice_cum = list(np.cumsum(tslice))

    if add_val and add_test:
        train_corpus = corpus
        tslice_tr = tslice
        id2word = corpora.Dictionary(train_corpus)
        train_bow = [id2word.doc2bow(text) for text in train_corpus]
        vocab = list(id2word.token2id.keys())
        
        pickling(train_corpus, save_dir+'train_corpus.pkl')
        pickling(train_bow, save_dir+'train_bow.pkl')
        pickling(id2word, save_dir+'id2word.pkl')
        pickling(vocab, save_dir+'vocab.pkl')
        pickling(tslice_tr, save_dir+'tslice_tr.pkl')
        with open(save_dir + 'vocab.txt', 'w') as outfile:
            outfile.write("\n".join(vocab))
    elif not(add_test):
        train_corpus = []
        test_corpus = []
        tstamp_tr = []
        tstamp_ts = []
        for s,e in zip([0]+tslice_cum, tslice_cum):
            values = random.sample(range(s, e), e-s)
            train_ind, test_ind, val_ind = np.split(values, [int(.8*len(values)), int(.9*len(values))])
            
            if add_val:
                train_ind = sorted(train_ind+val_ind) #add validation set to the train data
            else:
                train_ind = sorted(train_ind)
            test_ind = sorted(test_ind)

            train_corpus = train_corpus + ([corpus[ind] for ind in train_ind])
            tstamp_tr = tstamp_tr + [tstamp[ind] for ind in train_ind]
            tslice_tr = list(collections.Counter(tstamp_tr).values())

            test_corpus = test_corpus + ([corpus[ind] for ind in test_ind])
            tstamp_ts = tstamp_ts + [tstamp[ind] for ind in test_ind]
            tslice_ts = list(collections.Counter(tstamp_ts).values())

        #removing the words that are not in train_corpus
        id2word = corpora.Dictionary(train_corpus)
        vocab = list(id2word.token2id.keys())
        if verbose:
            print('vocabulary after removing words not in train: {}'.format(len(vocab)))
        test_corpus = [[w for w in test_corpus[ind] if w in vocab] for ind in range(len(test_corpus))]
        #Create Bow:
        train_bow = [id2word.doc2bow(text) for text in train_corpus]
        test_bow = [id2word.doc2bow(text) for text in test_corpus]

        pickling(train_corpus, save_dir+'train_corpus.pkl')
        pickling(test_corpus, save_dir+'test_corpus.pkl')
        pickling(train_bow, save_dir+'train_bow.pkl')
        pickling(test_bow, save_dir+'test_bow.pkl')
        pickling(id2word, save_dir+'id2word.pkl')
        pickling(vocab, save_dir+'vocab.pkl')
        pickling(tslice_tr, save_dir+'tslice_tr.pkl')
        pickling(tslice_ts, save_dir+'tslice_ts.pkl')
        with open(save_dir + 'vocab.txt', 'w') as outfile:
            outfile.write("\n".join(vocab))