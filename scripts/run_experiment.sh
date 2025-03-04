# Write the python command with any hydra overrides. This ends up being a log of ways that the experiments
# have been run in the past, to make it easier to figure out which command to execute. This script should
# be executed from the root kaggle_nlp_getting_started repo folder


# python trav_nlp/pipeline.py
# python trav_nlp/pipeline.py experiment.submit_to_kaggle=true
# python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-twitter-25
# python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-twitter-50
# python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-twitter-100
# python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-twitter-200

python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=fasttext-wiki-news-subwords-300
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=conceptnet-numberbatch-17-06-300
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=word2vec-ruscorpora-300
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=word2vec-google-news-300
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-wiki-gigaword-50
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-wiki-gigaword-100
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-wiki-gigaword-200
python trav_nlp/pipeline.py embeddings=glove1 embeddings.name=glove-wiki-gigaword-300
