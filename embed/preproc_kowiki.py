# 1. Download korean wiki dump from https://dumps.wikimedia.org/kowiki/20180320/
# 2. git clone https://github.com/attardi/wikiextractor
# 3. python WikiExtractor.py kowiki-20180320-pages-articles-multistream.xml.bz2
# 4. python preproc_kowiki.py
# => result.txt : 
#       preprocessed result file with twitter pos analyzer
#       each line represents a document tokenized by konlpy.tag.Twitter.pos()
#       it will be used as corpus of doc2vec TaggedLineDocument

import os
from konlpy import tag
from multiprocessing import Process, Queue, cpu_count

def parse_file(pid, input_queue, output_queue):
    twitter = tag.Twitter()
    def tagger(sent):
        return [word for word, tag in twitter.pos(sent) if tag not in ['Punctuation', 'Unknown']]

    while True:
        get = input_queue.get()
        if get is None:
            output_queue.put(pid)
            break
        
        with open(get) as f:
            lines = f.readlines()
        
        sent = []
        for line in lines:
            if line == '\n' or line.find('<doc') != -1:
                continue
            
            if line.find('doc>') != -1:
                output_queue.put(' '.join(sent))
                sent = []
                continue

            line = line.strip()
            try:
                sent.append(' '.join(tagger(line)))
            except Exception as e:
                print('Exception occured : "{}"'.format(line))
                print(e)
                output_queue.put(pid)
                break

root_dir = './text'
input_queue = Queue()
output_queue = Queue()

for i in range(cpu_count()):
    p = Process(target=parse_file, args=(i, input_queue, output_queue))
    p.daemon = True
    p.start()

for dir in os.listdir(root_dir):
    for file in os.listdir(os.path.join(root_dir, dir)):
        input_queue.put(os.path.join(root_dir, dir, file))

for _ in range(cpu_count()):
    input_queue.put(None)

pid_sum = sum(range(cpu_count()))
wait_sum = 0

doc_num = 0
corpus = []

while wait_sum < pid_sum:
    get = output_queue.get()
    if isinstance(get, str):
        corpus.append(get)

        doc_num += 1
        print('doc[{}] : {}...'.format(doc_num, get[:10]))

        if doc_num % 10000 == 0:
            with open('result.txt', 'at') as f:
                f.write('\n'.join(corpus) + '\n')
            
            corpus = []
    else:
        wait_sum += get

with open('result.txt', 'at') as f:
    f.write('\n'.join(corpus))