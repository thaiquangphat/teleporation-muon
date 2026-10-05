# Language Modeling on Wikitext103

## Download Wikitext103
```bash
mkdir -p ~/data/
cd ~/data/
kaggle datasets download -d vadimkurochkin/wikitext-103 -p .
unzip wikitext-103.zip -d wikitext-103
cp wikitext-103/wikitext-103/wiki.train.tokens ~/data/wt103/train.txt
cp wikitext-103/wikitext-103/wiki.valid.tokens ~/data/wt103/valid.txt
cp wikitext-103/wikitext-103/wiki.test.tokens  ~/data/wt103/test.txt
```
## Train MemTransformer
```bash
bash scripts.sh
```
