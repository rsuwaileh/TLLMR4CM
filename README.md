# When a Disaster Happens, We Are Ready: Location Mention Recognition from Crisis Tweets

Towards addressing the problem of recognizing locations, i.e., Location Mention Recognition (LMR), within social media posts during such disasters, past studies mainly focused on proposing techniques that assume the availability of abundant training data at the disaster onset. In our work, we adopt the more realistic assumption that _no_ (i.e., zero-shot setting) or _as little as a few hundred_ examples (i.e., few-shot setting) from the just-occurred event is available for training. Specifically, we examine the effect of training a BERT-based LMR model on past events using different settings, datasets, languages, and geo-proximity. Extensive empirical analysis provides several insights for building an effective LMR model during disasters, including 
1. Twitter crisis-related and location-specific data from geographically-nearby disaster events is more useful than all other combinations of training datasets in the zero-shot monolingual setting, 
2. using as few as 263-356 training tweets from the target language (i.e., few-shot setting) remarkably boosts the performance in the cross- and multilingual settings, and 
3. labeling about 500 target event's tweets leads to an acceptable LMR performance, higher than F1 of 0.7, in the monolingual settings. Finally, we conduct an extensive error analysis and highlight issues related to the quality of the available datasets and weaknesses of the current model.

**This repository provides the steps to reproduce the experiments that we reported in the our publication.**

## The datasets

We adopted three types of datasets in our work: (1) _General-purpose NER dataset_, (2) _Twitter NER dataset_, and (3) _Crisis-related multilingual Twitter LOC dataset_. We show various statistics of all the datasets in the cited publication below. 

1. **General-purpose NER dataset**: A well-known candidate for this category is the [CoNLL-2003 NER](https://www.clips.uantwerpen.be/conll2003/ner/) dataset [1]. 
2. **Twitter NER dataset**: We use the [Broad Twitter Corpus (BTC)](https://github.com/GateNLP/broad_twitter_corpus) as our Twitter NER dataset [2].
3. **Crisis-related Twitter LMR dataset**: We use seven datasets in this category; 
- The [floods dataset](https://github.com/halolimat/LNEx) consists of 4,500 tweets from _Chennai floods 2015_, _Louisiana floods 2016_, and _Houston floods 2016_ [3]. The tweets in these datasets are tagged using several location-related tags. In this work, we only use inLOC and outLOC, which indicate if the location is within or outside the disaster-affected areas respectively. We further filter out all hashtags used to collect the datasets, thus limiting their effect towards biasing the model's training process.
- The [multilingual datasets](https://revealproject.eu/geoparse-benchmark-open-dataset/) in this category are adopted from [4]. This source contains 6,386 multilingual tweets in English, Italian, and Turkish languages from four disasters, namely _Hurricane Sandy 2012_, _Christchurch earthquake 2012_, _Milan blackout 2013_, and _Turkey earthquake 2013_. Hurricane Sandy and Christchurch earthquake are English language datasets, Milan blackout is in Italian language, and Turkey earthquake is in Turkish language.

```
[1]. Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition. In Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL (pp. 142–147).

[2]. Derczynski, L., Bontcheva, K., & Roberts, I. (2016). Broad Twitter corpus: A diverse named entity recognition resource. In Proceedings of the 26th International Conference on Computational Linguistics: Technical Papers (pp. 1169–1179).

[3]. Al-Olimat, H. S., Thirunarayan, K., Shalin, V., & Sheth, A. (2018). Location name extraction from targeted text streams using gazetteer-based statistical language models. In Proceedings of the 27th International Conference on Computational Linguistics (pp. 1986–1997).

[4]. Middleton, S. E., Middleton, L., & Modafferi, S. (2014). Real-time crisis mapping of natural disasters using social media. IEEE Intelligent Systems, 29, 9–17.
```

For reproducibility, you need to follow these steps:

1. **Download the datasets** from the repository: Unfortunately, the crisis-related Twitter LMR datasets are licensed and have to be requested from the corresponding authors [3-4]. 
2. **Preprocess the datasets**: You need to convert the datssets into BILOU format and partition the datasets. Both CoNLL-2003 and BTC datasets are preprocessed and ready for download in this repository. Unfortunately, the preparation steps involve manual curation conducted at our side on the data before running the LMR model. Thus, to make the results reproducible, once you confirm that you acquired the crisis-related Twitter LMR datasets, we can share with you the preprocessed and partitioned data that we have.


## The LMR model:
We employed the NER BERT-based model from [HuggingFace](https://huggingface.co/) library. To run this model, follow the steps in [this forked version](https://github.com/rsuwaileh/transformers/tree/master/examples/ner).

### Publications
```
@article{suwaileh2022tlLMR4disaster,
    title={When a Disaster Happens, We Are Ready: Location Mention Recognition from Crisis Tweets},
    author={Suwaileh, Reem and Elsayed, Tamer and Imran, Muhammad and Sajjad, Hassan},
    journal={International Journal of Disaster Risk Reduction},
    year={2022}
}

@inproceedings{suwaileh2020tlLMR4disaster,
  title={Are We Ready for this Disaster? Towards Location Mention Recognition from Crisis Tweets},
  author={Suwaileh, Reem and Imran, Muhammad and Elsayed, Tamer and Sajjad, Hassan},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6252--6263},
  year={2020}
}

 
```

### Contact
Please send email to `rs081123@qu.edu.qa` if you have any question or suggestions.
