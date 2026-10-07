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

## Train on Modal
Create a Modal secret named `kaggle` with the keys `username` and `key` for
Kaggle dataset access. From this directory, start the default 100,000-step
Muon run with:
```bash
modal run modal_app.py
```
To run a shorter smoke test, set a smaller step count:
```bash
modal run modal_app.py --max-step 10
```
The dataset is stored in the `wt103-data` Modal Volume, and training logs and
checkpoints are committed to `wt103-checkpoints` under `/checkpoints/wt103`.
