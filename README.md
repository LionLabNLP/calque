
# I play of the cello: Evaluating Constructional Bleeding in Multilingual LLMs with CALQUE

<img width="3898" height="1453" alt="page1_toshiki (1)" src="https://github.com/user-attachments/assets/f3c960c2-78e3-4221-ac37-9d5abd18abf9" />


CALQUE is a multilingual dataset designed to evaluate constructional bleeding in multilingual large language models (LLMs) inspired from the minimal-pair paradigm.
The dataset covers seven languages (English, German, Swedish, French, Portuguese, Mandarin Chinese, and Japanese) and three construction types (body-part/everyday-activity/sensory experence).
This repository contains the dataset and code used to evaluate LLMs with CALQUE.

## Dataset

The dataset files are located in the `data/` directory.

- **`CALQUE.xlsx`**: The original CALQUE dataset.
- **`CALQUE_preprocessed.xlsx`**: A preprocessed version of CALQUE (e.g. removing periods).
- **`CALQUE_metadata.xlsx`**: The dataset enriched with metadata, including linguistic features used in the regression analyses and interlinear glosses following the [Leipzig Glossing Rules](https://www.eva.mpg.de/lingua/pdf/Glossing-Rules.pdf).

## Model Preference Calculation

To calculate model preferences, run the following under the `sentence_probability/` directory:

```bash
bash script/example_script.sh
```

The example script runs the preference calculation using [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b).

To evaluate a different model, modify the model identifier in `example_script.sh` accordingly.

Detailed results for each evaluated model are available in the `outputs2/` directory.

## Metadata Construction

To construct the metadata, run:

```bash
python src/metadata/build_metadata.py \
  --input ./data/CALQUE_preprocessed.xlsx \
  --output ./data/CALQUE_metadata.xlsx \
  --incomplete_preference_annotation \
  --align \
  --align-checkpoint awesome-without-co \
  --align-model ./models/awesome-align/model_without_co \
  --gloss all
```

This script constructs the metadata used in multivariable regression, including linguistic features and glosses (which require the OPENAI API key).

Some metadata features require additional resources, as described in the Requirements section.

## Additional Analyses

To reproduce the main plots, multivariable regression, spearman rank correlation and other statistical analyses, run:

```bash
python3 src/export_plots2.py \
  --outputs_dir ./outputs2 \
  --plots_dir ./plots2 \
  --metadata ./data/CALQUE_metadata.xlsx \
  --use-preference-annotation de fr pt sv zh ja \
  --only_main \
  --glotcc
```

The generated results are saved in the `plots2/` directory. `--glotcc` requires lang2size.json described below. If you want to run the code without pretraining language proportion data, you can simply omit that argument and execute the bash script.

## Interactive Results

Interactive visualizations of our results are available at:

https://konta3738.github.io/calque/

## Requirements

The Python environment can be reproduced using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate calque
```

Some scripts require additional resources.

### FLORES-200 Dataset

Download [FLORES-200](https://github.com/facebookresearch/flores/tree/main) and place the `dev` split at:

`./sentence_probability/data/flores200_dataset/dev/`

FLORES-200 is used to estimate unigram probabilities for SLOR calculation.



### GlotCC Language Statistics

The file `lang2size.json` should contain language-size statistics derived from `v1.0-stat-2.zip` in [GlotCC](https://github.com/cisnlp/GlotCC).

Extract the total content length field and place the file at:

`./sentence_probability/data/lang2size.json`

These statistics serve as a proxy for the relative representation of languages in pretraining data.


### Awesome-Align

Download the [Awesome-Align](https://github.com/neulab/awesome-align) checkpoint and place it at:

`./models/awesome-align/model_without_co/`

Awesome-Align is used to compute alignment-based features for the multivariable regression analyses.


## Citation

If you use CALQUE or the accompanying code in your research, please cite our paper:

*I play of the cello: Evaluating Constructional Bleeding in Multilingual LLMs with CALQUE.*

## License

The source code in this repository is licensed under the MIT License.

The CALQUE dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## References

[NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*](https://arxiv.org/abs/2207.04672).

[Kargaran et al. (2024), GlotCC: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages](https://www.proceedings.com/079017-0540.html).

[Dou and Neubig (2021), *Word Alignment by Fine-tuning Embeddings on Parallel Corpora*](https://aclanthology.org/2021.eacl-main.181/).

[OpenAI et al. (2025), *gpt-oss-120b & gpt-oss-20b Model Card*](https://arxiv.org/abs/2508.10925).
