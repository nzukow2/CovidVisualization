
LESSON_ID = 'dmap:mlu:word2vec'
NET_ID = 'jacanty2'

def install_ide(*args):
    from unittest.mock import Mock

    class MyMock(Mock):

        def __repr__(self):
            return ''
    return MyMock()
ide = install_ide(NET_ID, LESSON_ID)
pass #print(ide.welcome())
pass #ide.reader.view_section(1)
import gzip
import gensim
import pandas as pd
import warnings
pass #warnings.simplefilter(action='ignore', category=FutureWarning)
pass #warnings.simplefilter(action='ignore', category=UserWarning)
import LessonUtil as Util

def build_dataset_raw():
    filename = Util.path_for_data('cars.csv.gz')
    file = gzip.open(filename, 'rb')
    return [gensim.utils.simple_preprocess(line) for line in file]

def test_raw():
    document = build_dataset_raw()
    pass #print(document[0])
    pass #print(document[10])
pass #test_raw()
pass #ide.reader.view_section(3)
import LessonUtil as Util

def build_dataset():
    filename = Util.path_for_data('cars.csv.gz')
    df = pd.read_csv(filename, compression='gzip')
    df['Make_Model'] = ((df['Make'] + ' ') + df['Model'])
    features = ['Market Category', 'Make_Model', 'Vehicle Size', 'Vehicle Style', 'Engine Fuel Type', 'Transmission Type', 'Driven_Wheels']
    df = df[features]
    doc = []
    for (index, row) in df.iterrows():
        line = [r for v in row.values for r in str(v).split(',')]
        doc.append(line)
    return (doc, df)

def test_pd_data():
    (document, df) = build_dataset()
    pass #print(len(set(df['Make_Model'])))
    pass #print(document[0][0:5])
pass #test_pd_data()
pass #ide.reader.view_section(5)

def build_model_v0(doc):
    model = gensim.models.Word2Vec(doc)
    return model

def test_v0():
    (document, df) = build_dataset()
    model = build_model_v0(document)
    pass #print(len(model.wv.vocab))
pass #test_v0()
pass #ide.reader.view_section(7)

def evaluate_model(model, df=None):
    output = ''
    if (df is not None):
        unique_set = df['Make_Model'].unique()
        missing = 0
        for mm in unique_set:
            if (mm not in model.wv.vocab):
                missing += 1
        output += '{:d} models are missing of {:d}\n'.format(missing, len(unique_set))
    try:
        t = 'Toyota Camry'
        other = ['Honda Accord', 'Nissan Van', 'Mercedes-Benz SLK-Class']
        for o in other:
            output += ((((t + '->') + o) + ' ') + '{:0.4f}\n'.format(model.wv.similarity(t, o)))
        tuples = model.wv.most_similar(positive='Honda Odyssey', topn=3)
        for (mm, v) in tuples:
            output += (mm + ', ')
        output = output.strip(', ')
    except KeyError as e:
        output += ('\nError:' + str(e))
    return output

def test_v0():
    (document, df) = build_dataset()
    model = build_model_v0(document)
    pass #print(evaluate_model(model, df))
pass #test_v0()
pass #ide.reader.view_section(9)

def build_model_v1(doc):
    model = gensim.models.Word2Vec(doc, min_count=1, workers=1, window=10)
    return model

def test_v1():
    (document, df) = build_dataset()
    model = build_model_v1(document)
    pass #print(evaluate_model(model, df))
pass #test_v1()
import multiprocessing
pass #print(multiprocessing.cpu_count())
pass #ide.reader.view_section(11)

def build_model_v2(doc, ndim=100):
    model = gensim.models.Word2Vec(doc, min_count=1, workers=1, window=10, iter=15, size=ndim)
    return model

def test_v2():
    (document, df) = build_dataset()
    lis = [25, 50, 75, 100, 150, 200]
    lise = []
    for i in lis:
        model = build_model_v2(document, i)
        pass #print(('ndim is: ' + str(i)), evaluate_model(model, df))
pass #test_v2()
pass #ide.reader.view_section(13)
pass #ide.reader.view_section(14)

def build_model_v3(doc):
    model = gensim.models.Word2Vec(doc, min_count=1, workers=1, window=10, iter=15, size=100, sg=1, negative=15)
    return model

def test_v3():
    (document, df) = build_dataset()
    model = build_model_v3(document)
    pass #print(evaluate_model(model, df))
    model.save('carmodel.skipgram')
pass #test_v3()
pass #ide.reader.view_section(16)

def test_load():
    md2 = gensim.models.Word2Vec.load('carmodel.skipgram')
    pass #print(evaluate_model(md2))
pass #test_load()
pass #ide.reader.view_section(17)

def install_build_glove():
    pass
    pass
    pass
pass #install_build_glove()
pass #ide.reader.view_section(19)
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

def run_glove():
    pass

def test_glove():
    w2v_info = glove2word2vec('glove/vectors.txt', 'vec.word2vec')
    pass #print('voc. size, vector size', w2v_info)
    hp_model = KeyedVectors.load_word2vec_format('vec.word2vec', binary=False)
    pass #print(type(hp_model))
    pass #print(hp_model.most_similar('Harry', topn=5))
pass #ide.reader.view_section(21)
import gensim.downloader as api
import time

def load_glove_model(resource):
    pass #print(api.info(resource))
    st = time.time()
    model = api.load(resource)
    pass #print('load time', (time.time() - st))
    return model
pass #ide.reader.view_section(23)
pass #ide.reader.view_section(24)
pass #ide.reader.view_section(25)
pass #ide.reader.view_section(26)
pass #ide.reader.view_section(27)
pass #print(ide.tester.test_notebook())
pass #print(ide.tester.test_notebook(verbose=True))
pass #ide.tester.download_solution()
