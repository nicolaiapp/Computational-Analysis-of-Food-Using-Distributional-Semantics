# Computational-Analysis-of-Food-Using-Distributional-Semantics

https://www.nicolairuhnau.com

---

Many thanks to my supervisor, [Hinrich Schütze](http://www.cis.uni-muenchen.de/schuetze/), head of fundamental research on deep learning & statistical natural language processing at the [Center for Information and Language Processing](https://www.cis.uni-muenchen.de/research/index.html), Ludwig Maximilian University of Munich. 


Many thanks also to [Ehsaneddin Asgari](https://llp.berkeley.edu/asgari/).

---
My master's thesis in the field of [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) at the [Ludwig Maximilian University of Munich](http://www.en.uni-muenchen.de/index.html) at the [Center for Information and Language Processing](https://www.cis.uni-muenchen.de/research/index.html) 
-  goal
    -  benchmarking and visualizing [word embeddings](https://en.wikipedia.org/wiki/Word_embedding) using [machine learning](https://en.wikipedia.org/wiki/Machine_learning) and [topic modeling](https://en.wikipedia.org/wiki/Topic_model)
-  algorithms applied (wikipedia links)
    -  [word2vec](https://en.wikipedia.org/wiki/Word2vec)
    -  [fastText](https://en.wikipedia.org/wiki/FastText)
    -  [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
    -  [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
-  Python libraries used
    -  PyTorch
    -  scikit-learn
    -  gensim
    -  [tmtoolkit](https://github.com/WZBSocialScienceCenter/tmtoolkit)
    -  pandas
    -  MulticoreTSNE
    -  matplotlib and Seaborn
-  implementation
    -  [ingredient and cuisine prediction (sklearn Python Jupyter Notebooks)](https://github.com/nicolaiapp/Computational-Analysis-of-Food-Using-Distributional-Semantics/tree/master/implementation/prediction_classifiers)
    -  [topic models (tmtoolkit Python Jupyter Notebooks)](https://github.com/nicolaiapp/Computational-Analysis-of-Food-Using-Distributional-Semantics/tree/master/implementation/topic%20modeling)
    -  [PyTorch im2recipe modifications as described in section 2](https://github.com/nicolaiapp/im2recipe-Pytorch/)
    -  data
        -  [1 Million Recipes im2recipe](http://im2recipe.csail.mit.edu/dataset/login/)
        -  [Flavor Network Ahn et al](https:/~/github.com/nicolaiapp/Flavor-Network/)
-  the most interesting publications examined
    -  [im2recipe: Learning Cross-modal Embeddings for Cooking Recipes and Food Images 2017](http://pic2recipe.csail.mit.edu/im2recipe.pdf)
    -  [Flavor network and the principles of food pairing 2011](https://www.nature.com/articles/srep00196)
    -  [A Neural Network System for Transformation of Regional Cuisine Style 2018](https://www.frontiersin.org/articles/10.3389/fict.2018.00014/full)
    -  [Multilingualization of Restaurant Menu by Analogical Description 2017](https://dl.acm.org/doi/10.1145/3106668.3106671)
    -  [You Are What You Eat: Exploring Rich Recipe Information for Cross-Region Food Analysis 2018](https://ieeexplore.ieee.org/document/8059846)
    -  for the embeddings and any questions or requests please [contact me](https://nicolairuhnau.com/#contact)

---
__abstract__


Often the evaluation of food text representations is still done by hand, in lack of good automated evaluation methodologies. Visualizations of distributional food semantics are often not used to their full potential. There are only a few works that use dense word embeddings and more complex food corpus model techniques in general, such as topic modeling.


This thesis gives an overview of the existing literature and helps define the rather new field of research of the computational analysis of food using distributional semantics.

The use of various food text representations is investigated, creating embeddings and successfully conducting new experimental benchmarks in order to evaluate them.


The methodology of these automated benchmarks is explained step-by-step, showcasing powerful Python libraries.

Latent topics in a food dataset are extensively explored, with various visualizations techniques for a more intuitive understanding.

It is shown that a smaller domain specific corpus can produce embeddings with similar and sometimes better food category prediction capabilities as embeddings based on very large corpora such as Wikipedia or Google News.

An interesting find is how beneficial subword-level embeddings are in the context of food and what the reasons for this are. Complicated embeddings that include food image and recipe instructions information did generally perform not better and sometimes much worse than a simpler baseline embedding based on the same corpus.

It is shown how visualizations can be used to very eff ectively to explain results and describe datasets in an intuitive way.

The presented detailed list of the available linguistic food resources helps others in rapidly applying the techniques mentioned in this thesis to other datasets.

---

__cite.bib__

To cite this work using a bibliographic reference manager use the [cite.bib](https://github.com/nicolaiapp/Computational-Analysis-of-Food-Using-Distributional-Semantics/blob/master/cite.bib) or the bibtex code below

```bibtex
@mastersthesis{ruhnau_computational-analysis_2019,
	address = {Munich, Germany},
	title = {{Computational} {Analysis} of {Food} Using {Distributional} {Semantics}},
	school = {Ludwig Maximilian University of Munich},
	author = {Ruhnau, Nicolai},
	month = jan,
	year = {2019},
}
```
---
__tags__

natural language processing, deep learning, machine learning, data science, word2vec, word embeddings, distributional semantics, topic model, food, fastText, im2recipe, PyTorch, tmtoolkit, pyLDAvis, gensim, scikit-learn, pandas, MulticoreTSNE, matplotlib, t-SNE

---
License

[__GNU General Public License v3.0__](https://github.com/nicolaiapp/Computational-Analysis-of-Food-Using-Distributional-Semantics/blob/master/license)
