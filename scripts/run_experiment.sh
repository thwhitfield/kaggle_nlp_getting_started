# Write the python command with any hydra overrides. This ends up being a log of ways that the experiments
# have been run in the past, to make it easier to figure out which command to execute. This script should
# be executed from the root kaggle_nlp_getting_started repo folder


# python trav_nlp/pipeline.py
# python trav_nlp/pipeline.py experiment.submit_to_kaggle=true
python trav_nlp/pipeline.py embeddings=glove1 embeddings.gensim_embedding_name=glove-twitter-25
python trav_nlp/pipeline.py embeddings=glove1 embeddings.gensim_embedding_name=glove-twitter-50
python trav_nlp/pipeline.py embeddings=glove1 embeddings.gensim_embedding_name=glove-twitter-100
python trav_nlp/pipeline.py embeddings=glove1 embeddings.gensim_embedding_name=glove-twitter-200


