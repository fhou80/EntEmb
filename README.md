# Fine-Grained Semantics to Entity Embeddings (FGS2EE)
This repository contains code and data for the ACL 2020 paper,
```
@inproceedings{hou_2020_,
  title={Improving Entity Linking through Semantic Reinforced Entity Embeddings},
  author={Feng Hou and Ruili Wang and Jun He and Yi Zhou},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
The entity embeddings are tested on the following two linking models:
* the [wnel](https://github.com/lephong/wnel) model
* the [mulrel-nel](https://github.com/lephong/mulrel-nel) model

You can download the [entity embeddings](https://drive.google.com/file/d/1LFW888ewwzVaWP7mCNSejySbo-Wu56Sm/view?usp=sharing) directly and test them on your model.

## Generic steps
if you want to reproduce the entity embeddings, please follow the following steps:
* Download dataset
* Generate the semantic entity embeddings
* Generate the aggregated entity embeddings
* Test the entity embeddings on the linking models

## Download dataset
### Download our data
Download our data from [Googledrive](https://drive.google.com/open?id=1OtLnrH4SpDzdNNcuca-DdXCMwsDPsG3B)
1) the `entities_with_types_wikitext.zip` contains `[entity-name, entity types, wikipedia article`] from wikipedia-dump, unzip this file to directory `entities_types_texts`.
2) the `type_dict.type` (not include som OOVs) is a dictionary for type_word embeddings `type_vec.npy`, which is extracted from Word2Vec.
3) `type_list.ndjson` is the originally selected type words, `type_list_OOVs_remap.ndjson` is remapping OOV words.

### Download ganea's entity embeddings data
Download from links in repository [wenl](https://github.com/lephong/wnel) and [mulrel-nel](https://github.com/lephong/), get the following two files:
1) the entity dictionary `(ganea)dict.entity`
2) the entity embeddings `(ganea)entity_embeddings.npy`

## Generate the semantic entity embeddings
run: `python generate_semantic_embeddings.py <path of type_dict.type> <path of type_vec.npy> <directory of entities_types_texts> <saving_path> <value of T>`
* `saving_path` is the directory for saving semantic entity embeddings

## Generate aggregated entity embeddings
run: `python generate_aggregated_embeddings.py <path of semantic_dict.entity> <path of semantic_entity_vec.npy> <path of ganea_dict.entity> <path of ganea_entity_vec.npy> <saving path> <value of alpha>`

