# Global-and-Local-Information-Adjustment-for-Semantic-Similarity-Evaluation

Semantic Similarity Evaluation

  * [Global and Local Information Adjustment for Semantic Similarity Evaluation](https://doi.org/10.3390/app11052161)
  
    * `Tak-Sung Heo`, `Jong-Dae Kim`, `Chan-Young Park`, `Yu-Seop Kim`

-------------------------------------------------

## Dataset

  * Word2Vec
    * English - [Google News](https://code.google.com/archive/p/word2vec/)
    * Korean - [Kookmin Corpus](http://nlp.kookmin.ac.kr)
    
  * English dataset (A total of 100,000 data) - [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset)
  * Korean dataset (A total of 11,000 data) - [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) by Google translator, [Exobrain Korean paraphrase corpus](http://aiopen.etri.re.kr/service_dataset.php) (label 0 : 0-1, label 1 : 4-5), [German Translation Pair of Hankuk University of Foreign Studies](http://deutsch.hufs.ac.kr/), [Naver Question Pairs](http://kin.naver.com/)

-------------------------------------------------

## Model Structure

<p align="center">
	<img src="https://github.com/HeoTaksung/Global-and-Local-Information-Adjustment-for-Semantic-Similarity-Evaluation/blob/main/Image/Structure.png" alt="Model" width="50%" height="50%"/>
</p>

  * Word2Vec
  
  * Bidirectional Long Short-Term Memory
    * Global Features
  
  * [Self-Attention](https://pypi.org/project/keras-self-attention/) (Zheng, Guineng, et al. "Opentag: Open attribute value extraction from product profiles." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.)
  
  * [Capsule Networks](https://github.com/TeamLab/text-capsule-network) (Kim, Jaeyoung, et al. "Text classification using capsules." Neurocomputing 376 (2020): 214-221.)
    * Local Features
  
  * Manhattan Distance (α weight)
    * αGlobal_Mahattan + (1-α)Local_Manhattan

## Results

  * Result of α weight
    <p align="center">
      <img src="https://github.com/HeoTaksung/Global-and-Local-Information-Adjustment-for-Semantic-Similarity-Evaluation/blob/main/Image/Result.png" alt="Alpha weight" width="80%" height="80%"/>
    </p>
    
  * Accuracy Result
     <p align="center">
      <img src="https://github.com/HeoTaksung/Global-and-Local-Information-Adjustment-for-Semantic-Similarity-Evaluation/blob/main/Image/Model_result.png" alt="Model_Result" width="80%" height="80%"/>
    </p>
    
