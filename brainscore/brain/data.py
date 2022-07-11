"""
Get brain data
"""


import json
import logging
import os
import re
from glob import glob
from itertools import groupby
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

from .. import constants, paths
from .exclude_scans import exclude_scan

logger = logging.getLogger(__name__)


class Dataset:
    """Task/subtask/group/subject iterator for simplicity"""

    def __init__(self, base_dir):

        # Get metadata for all subjects for a given task

        base_dir = Path(base_dir)
        with open(base_dir / "code" / "task_meta.json") as f:
            self.task_meta = json.load(f)

        # Skip 'schema' task for simplicity
        skip_tasks = ["notthefalllongscram", "notthefallshortscram", "schema"]

        for skip in skip_tasks:
            self.task_meta.pop(skip)

        # Skip 'notthefall' scramble and 'schema' tasks for simplicity
        self.task_id = 0
        self.subtask_id = 0
        self.group_id = 0
        self.subject_id = 0

    def __repr__(self):
        print("task_id %i " % self.task_id)
        print("subtask_id %i" % self.subtask_id)
        print("group_id %i" % self.group_id)

    def __iter__(self):
        return self

    def __next__(
        self,
    ):

        # task
        task_keys = list(self.task_meta.keys())
        task = task_keys[self.task_id]

        # subtask
        subtasks = ["slumlord", "reach"] if task == "slumlordreach" else [task]
        subtask = subtasks[self.subtask_id]

        # groups
        if task == "milkyway":
            groups = ["original", "vodka", "synonyms"]
        # elif task == 'prettymouth':
        #    groups = ['affair', 'paranoia']
        else:
            groups = [None]
        group = groups[self.group_id]

        stim_label = task + group if group else task

        # subjects
        subjects = sorted(self.task_meta[task].keys())
        subject = subjects[self.subject_id]

        # next iter

        def next():
            # update iterator
            self.subject_id += 1
            if self.subject_id == len(subjects):
                self.subject_id = 0
                self.group_id += 1
            if self.group_id == len(groups):
                self.group_id = 0
                self.subtask_id += 1
            if self.subtask_id == len(subtasks):
                self.subtask_id = 0
                self.task_id += 1
            if self.task_id == len(task_keys):
                raise StopIteration

        if group and group != self.task_meta[subtask][subject]["condition"]:
            next()
            return self.__next__()

        next()

        return task, subtask, group, subject, stim_label


def get_task_df(
    task_exclude=["notthefalllongscram", "notthefallshortscram", "schema"],
    keep_milk=False,
):
    # Matching between task and subjects
    df = pd.read_csv(paths.base_dir / "participants.tsv", sep="\t")
    df = df.astype("str")
    dataset = []
    for i, row in df.iterrows():
        for task, condition, comprehension in zip(
            row.task.split(","), row.condition.split(
                ","), row.comprehension.split(",")
        ):
            if task == "milkyway" and keep_milk:
                task = task + condition
            if task in task_exclude:
                continue
            if comprehension != "n/a":
                comprehension = float(comprehension)
                if "shapes" in task:
                    comprehension /= 10
            else:
                comprehension = np.nan
            dataset.append(
                {
                    "audio_task": task,
                    "subject": row.participant_id,
                    "comprehension": comprehension,
                }
            )
    dataset = pd.DataFrame(dataset)

    # Task info
    checked_tasks = get_checked_tasks()

    # Merge
    dataset = pd.merge(dataset, checked_tasks, on="audio_task", how="inner")

    return dataset


def get_checked_tasks():
    """
    Tasks checked by hand - report onset of the paper.
    """
    if paths.checked_gentle_path.exists():
        tasks = [
            p.parent.name
            for p in list(
                Path(paths.checked_gentle_path).glob("*/align.csv"))]
        tasks = {k: {} for k in tasks}

        for key in tasks.keys():
            if key in [
                "milkywayoriginal",
                "milkywaysynonyms",
                "milkywayvodka",
            ]:
                tasks[key]["bold_task"] = "milkyway"
            else:
                tasks[key]["bold_task"] = key

        # Set onsets for some tasks
        for key in [
            "21styear",
            "milkywayoriginal",
            "milkywaysynonyms",
            "milkywayvodka",
            "prettymouth",
            "pieman",
        ]:
            tasks[key]["onset"] = 0
        for key in ["piemanpni", "bronx", "black", "forgot"]:
            tasks[key]["onset"] = 8
        for key in [
            "slumlordreach",
            "shapessocial",
            "shapesphysical",
            "sherlock",
            "merlin",
            "notthefallintact",
        ]:
            tasks[key]["onset"] = 3
        for key in ["lucy"]:
            tasks[key]["onset"] = 2  # 1 in events.tsv, 2 in text
        for key in ["tunnel"]:
            tasks[key]["onset"] = 2

        checked_tasks = {k: v for k, v in tasks.items() if "onset" in v}
        checked_tasks = pd.DataFrame(checked_tasks).T.reset_index()
        checked_tasks = checked_tasks.rename(columns={"index": "audio_task"})
    else:
        create_checked_stimuli()
        checked_tasks = get_checked_tasks()
    return checked_tasks


def create_checked_stimuli():
    """Save to a new directory checked stimuli"""
    logger.info(f"Creating new checked stimuli in {paths.checked_gentle_path}")
    tasks_with_issues = ["notthefallintact", "prettymouth", "merlin"]
    new_starts = [[25.8], [21], [29, 29.15]]
    tasks = [p.parent.name for p in list(
        Path(paths.gentle_path).glob("*/align.csv"))]
    for task in tasks:
        save_folder = paths.checked_gentle_path / task
        save_folder.mkdir(exist_ok=True, parents=True)
        df = pd.read_csv(paths.gentle_path / task / "align.csv", header=None)
        (paths.checked_gentle_path / task).mkdir(exist_ok=True)
        df.to_csv(paths.checked_gentle_path / task /
                  "align.csv", header=None, index=False)
    for task, new_vals in zip(tasks_with_issues, new_starts):
        df = pd.read_csv(paths.gentle_path / task / "align.csv", header=None)
        for i, val in enumerate(new_vals):
            df.iloc[i, 2] = val
            df.iloc[i, 3] = val + 0.05
        df.to_csv(paths.checked_gentle_path / task /
                  "align.csv", header=None, index=False)
    return True


# add their own script to path for simplicity


def format_text(text, lower=True):
    text = text.replace("\n", " ")
    text = text.replace(" -- ", ". ")
    text = text.replace(" – ", ", ")
    text = text.replace("–", "-")
    text = text.replace(' "', ". ")
    text = text.replace(' "', ". ")
    text = text.replace('" ', ". ")
    text = text.replace('". ', ". ")
    text = text.replace('." ', ". ")
    text = text.replace("?. ", "? ")
    text = text.replace(",. ", ", ")
    text = text.replace("...", ". ")
    text = text.replace(".. ", ". ")
    text = text.replace(":", ". ")
    text = text.replace("…", ". ")
    text = text.replace("-", " ")
    text = text.replace("  ", " ")
    if lower:
        text = text.lower()
    return text


def replace_special_character_chains(text):
    text = text.replace("-", " ")
    text = text.replace('laughs:"You', 'laughs: "You')
    return text


def gentle_tokenizer(raw_sentence):
    seq = []
    for m in re.finditer(
            constants.REGEX_GENTLE_TOKENIZER, raw_sentence, re.UNICODE):
        start, end = m.span()
        word = m.group()
        seq.append((word, start, end))
    return seq


def split_with_index(s, c=" "):
    p = 0
    for k, g in groupby(s, lambda x: x == c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q  # or p, q-1 if you are really sure you want that
        p = q


def format_tokens(x, lower=False):
    x = np.array(x)
    fx = [format_text(" " + xi + " ", lower=lower).strip()
          for xi in x.reshape(-1)]
    fx = np.array(fx).reshape(x.shape)
    return fx


def space_tokenizer(text):
    return [(text[i:j], i, j) for i, j in split_with_index(text, c=" ")]


def match_transcript_tokens(transcript_tokens, gentle_tokens):
    transcript_line = np.array(
        [i[1] for i in transcript_tokens])  # begin of each word
    raw_words = []
    for word, start, end in gentle_tokens:
        middle = (start + end) / 2
        diff = (middle - transcript_line).copy()
        diff[diff < 0] = np.Inf
        matching_idx = np.argmin(diff).astype(int)
        raw_words.append(transcript_tokens[matching_idx])

    return raw_words


def preproc_stim(df, text_fname, lower=False):
    text = open(text_fname).read()

    text = format_text(text, lower=lower)
    transcript_tokens = space_tokenizer(text)
    gentle_tokens = gentle_tokenizer(text)
    assert len(gentle_tokens) == len(df)

    spans = match_transcript_tokens(transcript_tokens, gentle_tokens)
    assert len(spans) == len(gentle_tokens)

    tokens = [w[0] for w in spans]
    tokens = format_tokens(tokens, lower=lower)

    # word raw
    df["word_raw"] = tokens

    # is_final_word
    begin_of_sentences_marks = [".", "!", "?"]
    df["is_eos"] = [np.any([k in i for k in begin_of_sentences_marks])
                    for i in tokens]

    # is_bos
    df["is_bos"] = np.roll(df["is_eos"], 1)

    # seq_id
    df["sequ_index"] = df["is_bos"].cumsum() - 1

    # wordpos_in_seq
    df["wordpos_in_seq"] = df.groupby("sequ_index").cumcount()

    # wordpos_in_stim
    df["wordpos_in_stim"] = np.arange(len(tokens))

    # seq_len
    df["seq_len"] = df.groupby("sequ_index")["word_raw"].transform(len)

    # end of file
    df["is_eof"] = [False] * (len(df) - 1) + [True]
    df["is_bof"] = [True] + [False] * (len(df) - 1)

    df["word_raw"] = df["word_raw"].fillna("")
    df["word"] = df["word"].fillna("")


def get_timing(task, subtask):
    event_meta = json.load(open(paths.event_meta_path))
    onset = event_meta[task][subtask]["onset"]
    duration = event_meta[task][subtask]["duration"]
    return onset, duration


def get_stimulus(task, add_phones=False, add_pos=False, lower=True):
    stim_fname = paths.gentle_path / task / "align.csv"
    text_fname = paths.gentle_path / task / "transcript.txt"
    stimuli = pd.read_csv(
        stim_fname, names=["word", "word_low", "onset", "offset"])
    preproc_stim(stimuli, text_fname, lower=lower)
    if add_phones:
        json_name = paths.gentle_path / task / "align.json"
        dico = json.load(open(json_name, "r"))
        stimuli["phones"] = [
            [v2["phone"] for v2 in v["phones"]] if "phones" in v else []
            for v in dico["words"]
        ]

        stimuli["phones0"] = [
            ",".join([j.split("_")[0] for j in i]) for i in stimuli["phones"]
        ]
        stimuli["phones1"] = [
            ",".join([j.split("_")[1] for j in i]) for i in stimuli["phones"]
        ]
        stimuli["phones"] = [",".join(i) for i in stimuli["phones"]]

        stimuli["n_phones"] = [len(i.split(",")) for i in stimuli["phones"]]
        stimuli["n_words"] = 1

    if add_pos:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        pos = []
        for word in stimuli["word_low"]:
            if type(word) is float:
                pos.append("")
            else:
                tok = nlp(word)
                p = [w.tag_ for w in tok]
                pos.append("|".join(p))
        stimuli["postag"] = pos

    return stimuli


def add_pulses_to_stim(stimuli, n_pulse, TR=1.5, reset_onset=False):
    """
    Return a dataframe with volumes and stimulus (similar to MOUS)
    """
    events = stimuli.copy().interpolate()
    events["word_index"] = range(len(events))
    if reset_onset:
        min_time = events.iloc[0].onset
        if min_time >= 1:  # TO CHECK WHEN IT IS STORY VS MUSIC FILE
            events[["onset", "offset"]] -= events.iloc[0].onset
    events["condition"] = "word"
    events["type"] = "word"
    pulses = pd.DataFrame(
        {
            "condition": "Pulse",
            "type": "Pulse",
            "onset": np.arange(n_pulse) * TR,
            "volume": np.arange(n_pulse),
        }
    )
    pulses["offset"] = pulses["onset"]
    events = pd.concat([events, pulses], axis=0).sort_values(
        ["onset", "offset"])
    events["volume"] = events["volume"].fillna(method="ffill").astype(int)
    events["volume_delay"] = events.groupby("volume").cumcount(
    ) / events.groupby("volume")["volume"].transform("count")
    events["volume"] += events["volume_delay"]
    return events


def trim_pulses_and_stim(subj_data, events, offset, trim_init=6):
    """
    Cut extra pulses
    """
    subj_data = subj_data[trim_init:offset]
    events = events.query("volume<@offset and volume>=@trim_init")
    events["volume"] -= trim_init
    assert len(subj_data) == len(events.query("condition=='Pulse'"))
    return subj_data, events


def get_phone_dic(overwrite=False, kind="phones"):
    """
    If does not exists, generate and save the dictionnary
    {phone: id vector} for each possible phoneme of the
    tasks.
    """

    if kind == "phones":
        f = paths.phone_dic
    elif kind == "phones0":
        f = paths.phone0_dic
    elif kind == "phones1":
        f = paths.phone1_dic

    if f.is_file() and not overwrite:
        return np.load(f, allow_pickle=True).item()

    tasks = [p.parent.name for p in list(
        paths.gentle_path.glob("*/align.csv"))]
    phones = []
    for task in tasks:
        stimuli = get_stimulus(task, add_phones=True)
        phones.extend([f.split(",") for f in stimuli[kind]])
    phones = np.concatenate(phones)
    phones = [ph for ph in phones if len(ph) > 0]
    phones = np.unique(phones)

    phone_dic = {k: np.eye(len(phones))[i] for i, k in enumerate(phones)}
    phone_dic[""] = np.zeros(len(phones))

    paths.phone_dic.parent.mkdir(exist_ok=True, parents=True)
    np.save(paths.phone_dic, phone_dic)
    return phone_dic


def get_pos_dic(overwrite=False):
    if paths.pos_dic.is_file():
        return np.load(paths.pos_dic, allow_pickle=True).item()

    print("LOADING SPACY")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    labels = nlp.get_pipe("tagger").labels

    pos_dic = {k: np.eye(len(labels))[i] for i, k in enumerate(labels)}
    pos_dic[""] = np.zeros(len(labels))

    paths.pos_dic.parent.mkdir(exist_ok=True, parents=True)
    np.save(paths.pos_dic, pos_dic)
    return pos_dic


def read_gifti(gifti_fn):
    gii = nib.load(gifti_fn)
    data = np.vstack([da.data[np.newaxis, :] for da in gii.darrays])
    return data


def get_bold(gii_fname, subject, exclude=True, afni_dir=paths.afni_dir):

    # Load scans to exclude
    scan_exclude = json.load(open(paths.scan_exclude_path))

    # Load bold responses
    subj_data = None
    bold_fns = natsorted(glob(str(afni_dir / subject / "func" / gii_fname)))
    assert len(bold_fns), str(afni_dir / subject / "func" / gii_fname)
    for bold_fn in bold_fns:

        if exclude and exclude_scan(bold_fn, scan_exclude):
            print(f"Excluding {os.path.basename(bold_fn)}")
            continue
        subj_data = read_gifti(bold_fn)

    return subj_data


# MEAN BOLD


def compute_mean_bold(hemi, df_task, space="fsaverage6"):
    bold = {}
    counts = {}
    for i, row in tqdm(df_task.iterrows()):
        if i % 100 == 0:
            print("i", i)
        # Load bold responses
        gii_fname = f"{row.subject}_task-{row.bold_task}_*space"
        gii_fname += f"-{space}_hemi-{hemi}_desc-clean.func.gii"
        try:
            subj_data = get_bold(
                gii_fname, row.subject, exclude=True,
                afni_dir=paths.afni_dir_nosmooth)
        except AssertionError as e:
            print("Assertion error", e)
            continue
        if subj_data is None:
            continue
        subj_data = subj_data[row.onset:, :]
        if row.audio_task not in bold:
            bold[row.audio_task] = subj_data
            counts[row.audio_task] = 0
        else:
            nr, nc = np.stack(
                [subj_data.shape, bold[row.audio_task].shape]).min(0)
            bold[row.audio_task] = bold[row.audio_task][:nr, :nc]
            bold[row.audio_task] += subj_data[:nr, :nc]
            counts[row.audio_task] += 1
    mean_bolds = {k: bold[k] / counts[k] for k in bold.keys()}
    # np.save(save_file, mean_bolds)
    return mean_bolds


def get_mean_bold(hemi='L'):
    bold_file = str(paths.mean_bolds) % hemi
    if Path(bold_file).is_file():
        return np.load(bold_file, allow_pickle=True).item()
    else:
        print(f"Computing mean bolds to {bold_file}....")
        Path(bold_file).parent.mkdir(exist_ok=True, parents=True)
        df_task = get_task_df()
        bold = compute_mean_bold(hemi, df_task)
        np.save(bold_file, bold)
        print(f"Saved mean bolds to {bold_file}....")
        return bold
