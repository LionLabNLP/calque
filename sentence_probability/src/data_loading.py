import pandas as pd

CONSTRUCTION_REQUIRED_COLUMNS = [
    "Language",
    "Construction language",
    "Strategy",
    "Subject/Undergoer Person+Number",
    "Time",
    "Sentence",
    "English baseline",
    "Interlinear gloss",
    "Object language line",
]

PROMPT_REQUIRED_COLUMNS = ["Task language", "Prompt language", "Prompt length", "Prompt"]

KEY_COLUMNS = ["English baseline", "Subject/Undergoer Person+Number", "Time"]

TASK_LANGUAGE_BY_CODE = {
    "de": "German",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
    "pt": "Portuguese",
    "se": "Swedish",
    "sv": "Swedish",
    "en": "English",
}


def load_construction_data(path):
    df = pd.read_excel(path)

    missing = [c for c in CONSTRUCTION_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in construction data: {missing}")

    return df


def select_examples(df, lang, construction_lang, strategy):
    mask = (
        (df["Language"] == lang)
        & (df["Construction language"] == construction_lang)
        & (df["Strategy"] == strategy)
    )
    return df.loc[mask].reset_index(drop=True)


def match_positive_negative(pos_df, neg_df, key_columns=KEY_COLUMNS):
    return pos_df.merge(neg_df, on=key_columns, how="inner", suffixes=("_pos", "_neg"))


def find_translation_source_sentence(construction_df, source_lang, pair):
    baseline = pair["English baseline"]
    person_number = pair["Subject/Undergoer Person+Number"]
    time = pair["Time"]

    mask = (
        (construction_df["Strategy"] == "proper strategy")
        & (construction_df["Language"] == source_lang)
        & (construction_df["Construction language"] == source_lang)
        & (construction_df["English baseline"] == baseline)
        & (construction_df["Subject/Undergoer Person+Number"] == person_number)
        & (construction_df["Time"] == time)
    )
    matched = construction_df.loc[mask]

    key_desc = (
        f"source language={source_lang!r}, English baseline={baseline!r}, "
        f"Subject/Undergoer Person+Number={person_number!r}, Time={time!r}"
    )

    if matched.empty:
        raise ValueError(
            f"No translation source sentence found (proper strategy) for {key_desc}."
        )

    sentences = matched["Sentence"].unique()
    if len(sentences) > 1:
        raise ValueError(
            f"Ambiguous translation source sentence for {key_desc}: "
            f"found distinct Sentence values {list(sentences)!r}."
        )

    return sentences[0]


CONSTRUCTION2_REQUIRED_COLUMNS = [
    "Target language",
    "Proper English strategy",
    "English preference annotation",
    "Literal English translation of target language strategy",
    "Proper strategy in target language",
    "Target language preference annotation",
    "Literal translation of Proper English into target language",
    "Proper strategy in Japanese",
    "Japanese preference annotation",
    "Literal translation of Proper target language into Japanese",
    "Group",
    "Subgroup",
    "Comment",
    "Temporal interpretation",
]

CONSTRUCTION2_OPTIONAL_COLUMNS = ["Subject"]

LANGUAGE_CODE_ALIASES = {"se": "sv"}
TARGET_LANGUAGE_CANONICAL_COLUMN = "target_language_canonical"


def canonical_language_code(code):
    if code is None:
        return None
    try:
        if pd.isna(code):
            return code
    except (TypeError, ValueError):
        pass
    text = str(code).strip()
    return LANGUAGE_CODE_ALIASES.get(text, text)


def canonical_target_languages(df):
    if TARGET_LANGUAGE_CANONICAL_COLUMN in df.columns:
        return df[TARGET_LANGUAGE_CANONICAL_COLUMN]
    return df["Target language"].map(canonical_language_code)


def has_sentence(series):
    return series.map(lambda v: not _is_blank(v))


def _is_blank(value):
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


EXPERIMENT_STANDARD = "standard"
EXPERIMENT_JAPANESE_CONTROL = "japanese_control"
JAPANESE_CONTROL_COMPARISON = (
    "Proper strategy in Japanese",
    "Literal translation of Proper target language into Japanese",
)
JAPANESE_CONTROL_ANNOTATION_COLUMN = "Japanese preference annotation"


def comparison_experiment(pos_col, neg_col):
    return EXPERIMENT_JAPANESE_CONTROL if (pos_col, neg_col) == JAPANESE_CONTROL_COMPARISON else EXPERIMENT_STANDARD


def japanese_control_availability(df):
    pos_col, neg_col = JAPANESE_CONTROL_COMPARISON
    usable = has_sentence(df[pos_col]) & has_sentence(df[neg_col])
    work = pd.DataFrame({
        "Target language": df["Target language"],
        TARGET_LANGUAGE_CANONICAL_COLUMN: canonical_target_languages(df),
        "usable": usable.astype(int),
    })
    out = (
        work.groupby(["Target language", TARGET_LANGUAGE_CANONICAL_COLUMN], sort=True)["usable"]
        .agg(rows="size", usable="sum")
        .reset_index()
    )
    out["skipped"] = out["rows"] - out["usable"]
    total = {"Target language": "ALL", TARGET_LANGUAGE_CANONICAL_COLUMN: "ALL",
             "rows": int(out["rows"].sum()), "usable": int(out["usable"].sum()), "skipped": int(out["skipped"].sum())}
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)

PAIR_ID_COLUMN2 = "Proper English strategy"

SENTENCE_COLUMN_LANGUAGE_MAP = {
    "Proper English strategy": {
        "language": "en",
        "construction_language": "en",
    },
    "Literal English translation of target language strategy": {
        "language": "en",
        "construction_language": "target",
    },
    "Proper strategy in target language": {
        "language": "target",
        "construction_language": "target",
    },
    "Literal translation of Proper English into target language": {
        "language": "target",
        "construction_language": "en",
    },
    "Proper strategy in Japanese": {
        "language": "ja",
        "construction_language": "ja",
    },
    "Literal translation of Proper target language into Japanese": {
        "language": "ja",
        "construction_language": "target",
    },
}


def load_construction_data2(path):
    df = pd.read_excel(path)

    missing = [c for c in CONSTRUCTION2_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in construction data: {missing}")

    df[TARGET_LANGUAGE_CANONICAL_COLUMN] = df["Target language"].map(canonical_language_code)
    return df


def validate_sentence_columns2(df, pos_col, neg_col):
    for label, col in (("--pos", pos_col), ("--neg", neg_col)):
        if col not in df.columns:
            raise ValueError(f"{label} column not found in construction data: {col!r}")
        if col not in SENTENCE_COLUMN_LANGUAGE_MAP:
            raise ValueError(
                f"{label} column {col!r} is not a recognized sentence column. "
                f"Expected one of: {sorted(SENTENCE_COLUMN_LANGUAGE_MAP)}"
            )


def sentence_column_language(column, target_language):
    spec = SENTENCE_COLUMN_LANGUAGE_MAP[column]["language"]
    return target_language if spec == "target" else spec


def sentence_column_construction_language(column, target_language):
    spec = SENTENCE_COLUMN_LANGUAGE_MAP[column]["construction_language"]
    return target_language if spec == "target" else spec


def select_examples2(df, lang, pos_col, neg_col):
    validate_sentence_columns2(df, pos_col, neg_col)

    mask = (
        (canonical_target_languages(df) == canonical_language_code(lang))
        & has_sentence(df[pos_col])
        & has_sentence(df[neg_col])
    )

    return df.loc[mask].reset_index(drop=True)

def select_translation_rows2(df, target_language):
    mask = (
        (canonical_target_languages(df) == canonical_language_code(target_language))
        & has_sentence(df["Proper English strategy"])
    )
    return df.loc[mask].reset_index(drop=True)


def load_prompt_conditions(path, lang):
    df = pd.read_excel(path)

    missing = [c for c in PROMPT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) in prompt data: {missing}")

    if lang not in TASK_LANGUAGE_BY_CODE:
        raise ValueError(f"No Task language mapping for lang code: {lang!r}")

    task_language = TASK_LANGUAGE_BY_CODE[lang]
    df = df[df["Task language"] == task_language]

    conditions = [
        {
            "prompt_text": str(row["Prompt"]).strip(),
            "prompt_language": str(row["Prompt language"]).strip(),
            "prompt_length": str(row["Prompt length"]).strip(),
        }
        for _, row in df.iterrows()
    ]

    conditions.append({
        "prompt_text": None,
        "prompt_language": "raw",
        "prompt_length": "raw",
    })

    return conditions
