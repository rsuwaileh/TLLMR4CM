# When a Disaster Happens, We Are Ready: Location Mention Recognition from Crisis Tweets

### The Datasets:
To prepare the datasets, you need to follow these steps:
1. Acquire the datasets from the corresponding authors:
- The floods dataset consists of 4,500 tweets from Chennai floods 2015, Louisiana floods 2016, and Houston floods 2016 [1]. The tweets in these datasets are tagged using several location-related tags. In this work, we only use inLOC and outLOC, which indicate if the location is within or outside the disaster affected areas respectively. We further filter out all hashtags used to collect the datasets, thus limiting their effect towards biasing the models training process.
- The multilingual datasets in this category are adopted from [2]. This source contains 6,386 multilingual tweets in English, Italian, and Turkish languages from four disasters, namely Hurricane Sandy 2012, Christchurch earthquake 2012, Milan blackout 2013 and Turkey earthquake 2013. Hurricane Sandy and Christchurch earthquake are English language datasets, Milan blackout is in Italian language, and Turkey earthquake is in Turkish language.

2. Unfortunately, the dataset preparation steps involve manual curation conducted at our side on the data before running the LMR model. Thus, to make the results reproducible, once you confirm that you acquired the dataset, we can share the partitions that we have with you. 

[1]. Al-Olimat, H. S., Thirunarayan, K., Shalin, V., & Sheth, A. (2018). Location name extraction from targeted text streams using gazetteer-based statistical language models. In Proceedings of the 27th International Conference on Computational Linguistics (pp. 1986–1997).

[2]. Middleton, S. E., Middleton, L., & Modafferi, S. (2014). Real-time crisis mapping of natural disasters using social media. IEEE Intelligent Systems, 29, 9–17.

### The LMR model:
We employed the NER BERT-based model from [HuggingFace](https://huggingface.co/) library. To run this model, follow the steps in [this forked version](https://github.com/rsuwaileh/transformers/tree/master/examples/ner).

### Publications
```
@inproceedings{suwaileh2020tlLMR4disaster,
  title={Are We Ready for this Disaster? Towards Location Mention Recognition from Crisis Tweets},
  author={Suwaileh, Reem and Imran, Muhammad and Elsayed, Tamer and Sajjad, Hassan},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6252--6263},
  year={2020}
}

@article{suwaileh2022tlLMR4disaster,
    title={When a Disaster Happens, We Are Ready: Location Mention Recognition from Crisis Tweets},
    author={Suwaileh, Reem and Elsayed, Tamer and Imran, Muhammad and Sajjad, Hassan},
    journal={International Journal of Disaster Risk Reduction},
    year={2022}
}
 
```

