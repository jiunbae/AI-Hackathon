from gensim.models import doc2vec
from multiprocessing import cpu_count

model = doc2vec.Doc2Vec(vector_size=300, window=300, min_count=5, workers=cpu_count())

print('Preparing vocabulary...')
documents = doc2vec.TaggedLineDocument('./result.txt')
model.build_vocab(documents)

print('corpus size : {}'.format(model.corpus_count))

for epoch in range(10):
    print('epoch {} : alpha {}'.format(epoch, model.alpha))
    model.train(documents, total_examples=model.corpus_count, epochs=1)

model_name = './doc2vec_model/doc2vec_twitter_kowiki_300000_docs.model'
model.save(model_name)
model.save_word2vec_format(model_name + '.word2vec_format')
