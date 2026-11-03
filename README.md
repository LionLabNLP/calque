# CALQUE

This repository contains the dataset, code, and analysis results for **CALQUE**, a multilingual benchmark for evaluating constructional bleeding in large language models (LLMs).

## 1. Dataset

The dataset files are located in the `data/` directory.

- **`CALQUE.xlsx`**: The original CALQUE dataset.
- **`CALQUE_preprocessed.xlsx`**: A preprocessed version of CALQUE with periods removed.
- **`CALQUE_metadata.xlsx`**: The dataset enriched with metadata, including linguistic features used in the regression analyses and interlinear glosses following the [Leipzig Glossing Rules]([https://www.eva.mpg.de/lingua/resources/glossing-rules.php](https://www.eva.mpg.de/lingua/pdf/Glossing-Rules.pdf)).

## 2. Model Preference Calculation

To calculate model preferences, run:

```bash
bash script/example_script.sh
```

The example script runs the preference calculation using `openai/gpt-oss-120b`.

To evaluate a different model, modify the model identifier in `example_script.sh` accordingly.

Detailed results for each evaluated model are available in the `outputs2/` directory.

## 3. Metadata Construction

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

This script constructs the metadata used in subsequent analyses, including linguistic features and glosses.

Some metadata features require additional resources, as described in the Requirements section.

## 4. Additional Analyses

To reproduce the main plots and statistical analyses, run:

```bash
python3 src/export_plots2.py \
  --outputs_dir ./outputs2 \
  --plots_dir ./plots2 \
  --metadata ./data/CALQUE_metadata.xlsx \
  --use-preference-annotation de fr pt sv zh ja \
  --only_main
```

The generated results are saved in the `plots2/` directory.

## 5. Interactive Results

Interactive visualizations of our results are available at:

https://konta3738.github.io/calque/

These visualizations allow users to explore the results across models and languages.

## 6. Requirements

The Python environment can be reproduced using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate calque
```

Some analyses require additional resources.

### 6.1. FLORES-200 Dataset

Download [FLORES-200](https://github.com/facebookresearch/flores/tree/main) and place the `dev` split at:

`./sentence_probability/data/flores200_dataset/dev/`

FLORES-200 is used to estimate empirical unigram probabilities for SLOR calculation.

Reference: [NLLB Team et al. (2022), *No Language Left Behind: Scaling Human-Centered Machine Translation*](https://arxiv.org/abs/2207.04672).



### 6.2. GlotCC Language Statistics

The file `lang2size.json` contains language-size statistics derived from [GlotCC](https://github.com/cisnlp/GlotCC).

Place the file at:

`./sentence_probability/data/lang2size.json`

These statistics serve as a proxy for the relative representation of languages in pretraining data.

Reference: [Kargaran et al. (2024), GlotCC: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages](https://www.proceedings.com/079017-0540.html).

### 6.3. Awesome-Align

The Awesome-Align model should be placed under:

```text
./sentence_probability/models/awesome-align/
```

Awesome-Align is used to compute alignment-based features for the multivariable regression analyses.

This resource is optional and is only needed when computing the corresponding regression features.

https://github.com/neulab/awesome-align

### 6.3. Awesome-Align

Download the multilingually fine-tuned [Awesome-Align](https://github.com/neulab/awesome-align) checkpoint without `--train_co` and place it at:

`./models/awesome-align/model_without_co/`

Awesome-Align is used to compute alignment-based features for the multivariable regression analyses.

Reference: [Dou and Neubig (2021), *Word Alignment by Fine-tuning Embeddings on Parallel Corpora*](https://aclanthology.org/2021.eacl-main.181/).


## 7. Citation

If you use CALQUE or the accompanying code in your research, please cite our paper:

*I play of the cello: Evaluating Constructional Bleeding in Multilingual LLMs with CALQUE.*

## 8. License

License information for the code and dataset will be provided separately.
