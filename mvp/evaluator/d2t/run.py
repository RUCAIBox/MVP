import sys
from nltk.tokenize import word_tokenize
import subprocess
import os

def tokenize(string):
    return ' '.join(word_tokenize(string))

def normalize(string, remove=False):
    if remove:
        string = string.replace('`', '')
        string = string.replace("'", '')
        string = string.replace('"', '')
        string = string.replace(';', '')
        string = string.replace(' - ', ' -- ')
    return string

assert len(sys.argv) == 3

reference = sys.argv[1]
prediction = sys.argv[2]

tmp = open('/tmp/ref.txt', 'w')
with open(reference, 'r') as f:
    for line in f:
        line = line.strip()
        if len(line) > 0:
            tmp.write(normalize(line) + '\n')
tmp.close()
print("Finished writing the reference file")

tmp = open('/tmp/pred.txt', 'w')
with open(prediction, 'r') as f:
    for line in f:
        line = normalize(tokenize(line.strip()))
        tmp.write(line + '\n')
print("Finished writing the prediction file")

#subprocess.call(["head", '-n', '5000', '/tmp/ref.txt', '>', '/tmp/ref.txt.1'])
#subprocess.call(["head", '-n', '5000', '/tmp/pred.txt', '>', '/tmp/pred.txt.1'])
#os.system('head -n 5000 /tmp/ref.txt > /tmp/ref.txt.1')
#os.system('head -n 5000 /tmp/pred.txt > /tmp/pred.txt.1')

tmp.close()

os.system("perl multi-bleu.perl /tmp/ref.txt < /tmp/pred.txt")
