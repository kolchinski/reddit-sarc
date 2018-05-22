mkdir -p SARC/2.0/pol
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/train-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/train-unbalanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/test-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/pol/test-unbalanced.csv.bz2
bzip2 -d comments.json.bz2
bzip2 -d train-balanced.csv.bz2
bzip2 -d train-unbalanced.csv.bz2
bzip2 -d test-balanced.csv.bz2
bzip2 -d test-unbalanced.csv.bz2
mv comments.json SARC/2.0/pol/
mv train-balanced.csv SARC/2.0/pol/
mv train-unbalanced.csv SARC/2.0/pol/
mv test-balanced.csv SARC/2.0/pol/
mv test-unbalanced.csv SARC/2.0/pol/

mkdir ../static
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip
rm wiki-news-300d-1M-subword.vec.zip
mv wiki-news-300d-1M-subword.vec ../static/

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip
mkdir ../static/glove
mv glove*.txt ../static/glove/


mkdir -p SARC/2.0/main
wget http://nlp.cs.princeton.edu/SARC/2.0/main/comments.json.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/train-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/test-balanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/test-unbalanced.csv.bz2
wget http://nlp.cs.princeton.edu/SARC/2.0/main/train-unbalanced.csv.bz2
bzip2 -d comments.json.bz2
bzip2 -d train-balanced.csv.bz2
bzip2 -d train-unbalanced.csv.bz2
bzip2 -d test-unbalanced.csv.bz2
bzip2 -d test-balanced.csv.bz2
mv comments.json SARC/2.0/main/
mv train-balanced.csv SARC/2.0/main/
mv test-balanced.csv SARC/2.0/main/
mv test-unbalanced.csv SARC/2.0/main/
mv train-unbalanced.csv SARC/2.0/main/

wget http://nlp.cs.princeton.edu/DisC/amazon_glove1600.txt.bz2
bzip2 -d amazon_glove1600.txt.bz2
mv amazon_glove1600.txt ../static
