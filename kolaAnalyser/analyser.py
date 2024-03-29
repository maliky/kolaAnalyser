# -*- coding: utf-8 -*-
"""
Un script qui analyse des motifs dans une base de données de prix.
Le but est de trouver des motifs fréquents ou intérssant dans divers sur des timesframe (window) par classique
ex. 7, 11, 13, 17, 19, 23, 31, 37, 41, 47, 49, 53, 59  minutes. (nombre entiers)
"""
# need to parallesise the code et do it in Haskell
import logging
import os
import re
import argparse
import pickle

from pathlib import Path
from typing import Sequence, List, Set
from multiprocessing import Pool  # , log_to_stderr
from numpy import ndarray
from pandas import (
    concat,
    DataFrame,
    Series,
    MultiIndex,
    date_range,
    Timedelta,
    Timestamp,
)

from mlkHelper.statistiques import (
    sample,
    prepare_data,
    add_rolling_essential,
    beast_boolean_mask,
    ts_extent,
)
from mlkHelper.statcons import (
    LHc,
    BEASTS,
)
from mlkHelper.timeUtils import (
    get_td_from_tsindex,
    get_td_shift_from_index_name,
)

from kolaAnalyser.market_proba import ordonne_motifs

# logger = log_to_stderr()  # logging.getLogger("INFO")  #log_to_stderr()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def data_f(df: DataFrame, frac: float, cols: Sequence = LHc):
    """Select the cols columns in the df DataFrame and return them."""
    return sample(df, cols, frac=frac)


def sdata_f(df: DataFrame, frac: float, by_, cols: Sequence = LHc):
    """
    Renvois un df sampled at x%
    and extened with rolling essential on by_ timeframe
    """
    return add_rolling_essential(data_f(df, frac, cols), by_=by_)


def add_beast(data: DataFrame, low_: str, high_: str):
    """Add the beast to the data using cols"""
    ndata = data.copy()
    _low = data.loc[:, low_]
    _high = data.loc[:, high_]

    _bmask = beast_boolean_mask(_low, _high)
    for _name in BEASTS:
        mask = _bmask[_name]
        ndata.loc[mask, "beasts"] = _name

    return ndata


def create_indexes_en_creneaux(data: DataFrame, ts_pas, freq_décalage="60s") -> Series:
    """
    Prend la base de donnée des prix, avec les rolling min and high
    et renvois un series Timestamp indexes, dont l'index est
    lui même un Timestamp Index.

    chaque champs est un index légèrement décalé par rapport aux autres.
    le décalage est de freq_décalage. et le pas des indexes est ts_pas
    """
    start_ts, END_TS = ts_extent(data)
    PAS = Timedelta(ts_pas)
    start_end_ts = start_ts + PAS - Timedelta("60s")
    start_ts_range = date_range(
        start=start_ts, end=start_end_ts, freq=Timedelta(freq_décalage)
    )

    def _make_index_with_PAS(start_ts_):
        """Fais un DatetimeIndex de pas voulu commençant à start_ts_"""
        return date_range(start=start_ts_, end=END_TS, freq=PAS)

    len_sequence = f"{len(_make_index_with_PAS(start_ts)):->9,d}".replace(",", " ")
    logger.info(
        f"Creating {len(start_ts_range)} '{ts_pas}' indexes"
        f" décalés de {freq_décalage} et de longueur {len_sequence}"
    )

    # pour chaque date de départ avec le shift spécifié
    # on créons un index DatetimeIndex avec le pas voulu.
    _indexes = {}
    for i, start_ts in enumerate(start_ts_range):
        print(f"{i+1:3d}/{len(start_ts_range)} :\t{start_ts}", end="\r")
        _indexes[start_ts] = _make_index_with_PAS(start_ts)

    indexes = Series(_indexes)
    # on homogénéise les indexes certain on un champ de plus que les autres
    # on enlève le dernier
    min_len = indexes.apply(len).min()
    max_mask = indexes.loc[indexes.apply(len) > min_len].index
    for idx in max_mask:
        indexes.loc[idx] = indexes.loc[idx][:-1]

    # on ajoute du métadata
    indexes.index.name = f"tsh_toute_les_{freq_décalage}"
    indexes.name = f"séquences_de_{ts_pas}"
    return indexes


####


def make_sequence_in_beasts(data, sequences_indexes_: Series, ts_pas: str) -> Series:
    """
    Renvois une Series avec un multiIndex dont le première niveau est un TimestampIndex en peigne
    et le second le nom des séries en créneaux
    ts_pas c'est le pas à l'intérieur de chaque série.
    """

    _td = get_td_from_tsindex(sequences_indexes_[0])
    logger.info(
        f"Creating {len(sequences_indexes_)} '{ts_pas} beasts' sequences "
        f" starting with a {Timedelta(_td*1e9)}s shift"
    )

    beast_sequences = Series(None)
    sequences_num = Series(None)
    # pour chaque indexes, liste de dates séparées par un ts_pas
    # on récupère les données prix, max_high low ect..
    for i, idx in enumerate(sequences_indexes_):
        print(f"{i:->4d}/{len(sequences_indexes_)}", end="\r")
        _prix = data.loc[idx].dropna()  # normalement n'enlève qu'un champs

        # on créér une série pour plus facilement la concaténé par la suite
        # l'index c'est les temps séparé d'un ts_pas
        # les valeurs sont les beasts
        _beasts = Series(
            add_beast(_prix, low_=f"min_low_r{ts_pas}", high_=f"max_high_r{ts_pas}")
            .dropna()
            .beasts,
            name=idx[0].strftime("%Y-%m-%dT%H:%M"),  # le nom c'est la date
        )
        beast_sequences = (
            _beasts if not len(beast_sequences) else concat([beast_sequences, _beasts])
        )
        _seq = Series(index=_beasts.index, data=[f"s{i+1}"] * len(_beasts), name="seq")
        sequences_num = (
            _seq if not len(sequences_num) else concat([sequences_num, _seq])
        )
    beast_sequences.name = "beast_seq"
    sequences_num.name = "seq"
    _df = DataFrame(beast_sequences).join(other=sequences_num)
    return Series(
        _df.sort_index().set_index("seq", append=True).beast_seq,
        name=f"{beast_sequences.name}_{ts_pas}",
    )


def get_last_letter(beast_sequence) -> str:
    """
    Transforme une sequence de beast en chaine de charactère avec seulement
    la dernière lettre du nom de la bête.
    """
    pattern_sequence = "".join(map(lambda b: b[-1], beast_sequence))
    return pattern_sequence


def build_beast_processsions(data: Series) -> DataFrame:
    """Constuire un df d'une colonne avec la processions des beasts"""
    # verifier la presence des times dans le nom de data
    assert isinstance(data, Series), (
        "{type(data)}:  data doit être une série.   Une liste des beasts avec"
        " un multi index permettant de séparer les séries temporelles"
        " qui s'intercallent."
    )
    assert isinstance(data.index, MultiIndex), (
        "L'index du data doit avoir deux niveaux, un avec des Timestamp"
        " l'autre une numérotation catégorisant les séries ex s1, s2, s3..."
    )
    assert isinstance(
        data.index[0][0], Timestamp
    ), "Le premier niveaux du multi-Index doit contenir des Timestamp"

    _patterns = {}
    gps = data.groupby(by=lambda x: x[1])

    logger.info(
        f"Regroupe les données selon le 2ème level de l'index (ie : {data.index.names[1]}), "
        f"len(gps)={len(gps)}"
    )
    len_gps = f"{len(gps):->9,d}".replace(",", " ")
    for i, (seq_name, _beast_seq) in enumerate(gps):
        len_i = f"{i+1:->9,}".replace(",", " ")
        print(f"{len_i}/{len_gps}", end="\r")
        start_ts = _beast_seq.index[0]
        _patterns[start_ts] = get_last_letter(_beast_seq.values)

    patterns = Series(_patterns).sort_index()
    patterns.index.names = (data.index.names[0] + "_start", data.index.names[1])

    df = DataFrame(patterns, columns=["processions"])
    df.columns.name = data.name
    return df


def get_mots_du_corpus_as_set(df_: DataFrame, taille_mot: int) -> set:
    """
    Prend un corpus sous forme de dataFrame
    et renvois tout les mots de taille_mot
    """
    docs = get_corpus_from_beast_processions(df_)
    len_range = len(docs[0]) - taille_mot + 1
    return set([doc[i : i + taille_mot] for doc in docs for i in range(len_range)])


def get_corpus_from_beast_processions(df_: DataFrame) -> ndarray:
    """
    Prend un DataFrame spécifique appelé beast_procession
    qui a une colonne appelé procession contenant une longue chaine de caractère
    représentant un document.
    Extrait ces chaines et renvois un tableau avec
    """
    assert "processions" in df_.columns, f"_df.columns={df_.columns}"
    values = df_.processions.values
    # on vérifie que tout les docs on la même longueur (il faudrait qqpart)
    return values


def get_last_word_of_corpus(df_: DataFrame, taille_mot: int) -> Set[str]:
    """Renvois l'ensemble des derniers mots de taille_mot de chaque documents du corpus"""
    if taille_mot == 1:
        return set()

    docs = get_corpus_from_beast_processions(df_)
    return set(map(lambda d: d[-taille_mot:], docs))


def create_dico_de_motifs_sets(longueurs_: List[int], corpus: DataFrame):
    """Créer un dictionnaire de motif de différentes longueurs à partir des symbols utilisé dans le corpus."""

    def _assert_mot_equal_len(mots_):
        len_mots = list(map(lambda m: len(m), mots_))
        message = f"Les mots ont des tailles différentes : {set(len_mots)}"
        assert sum([len_mot - max(len_mots) for len_mot in len_mots]) == 0, message

    _longueurs = sorted(longueurs_)

    taille_max_mot = _longueurs[-1]

    mots_à_la_taille = {
        taille_max_mot: get_mots_du_corpus_as_set(corpus, taille_max_mot)
    }

    # de la plus grande à la plus petite taille en excluant le max
    for taille_mot in _longueurs[::-1][1:]:
        if taille_mot + 1 in mots_à_la_taille.keys():
            mots_à_la_taille[taille_mot] = {
                m[:-1] for m in mots_à_la_taille[taille_mot + 1]
            } | get_last_word_of_corpus(corpus, taille_mot)
        else:
            mots_à_la_taille[taille_mot] = get_mots_du_corpus_as_set(corpus, taille_mot)
        try:
            _assert_mot_equal_len(mots_à_la_taille[taille_mot])
        except AssertionError as ae:
            import ipdb

            ipdb.set_trace()
            raise (ae)

    return mots_à_la_taille


def motifs_matches_inone(
    name: str, procession_, motifs_: Sequence, with_detail=False
) -> DataFrame:
    """
    Cherche les motifs de motifs_ dans la procession.
    motifs_ est une Sequence de sous-chaines de beasts de même taille.
    ex. ['aa', 'ab', 'ba', 'bb']

    # le dictionnaires motifs contient une succession de motifs
    # de taille identique identifié par leur taille
    # {1: ['a','b'], 2: ['aa', 'ab', 'ba', 'bb']}

    une procession est un object avec un index (DatetimeIndex)
    et un longue chaine de beasts (processions) résumant l'évolution du marché.
    la première beast à eu lieu à la date de l'index et s'est poursuive
    toutes les ... object name ?

    Renvois un DataFrame avec un MultiIndex comprenant autant de levels
    que la taille des motifs à chercher.
    Il y a trois colonnes.
    - nb_match: le nombre de fois que le mot a été trouvé dans la procession
    - freq_mot: la fréquence du mot par rapport au nombre de mot possible , la   Une avec l'objet botte (recoupant)
    """

    START_TS, SEQ_NUM = procession_.name
    PAS_TD = Timedelta(get_td_shift_from_index_name(procession_))

    def _trsf_botte_pos_in_ts(botte_position):
        """
        transform a botte_position in a timestamp

        get a position in the sequence (botte_position)
        and return the timestamp that it is refering too
        using start_ts given above and the timedelta of the sequence
        found in the index name

        un mot est une suite de lettre (de beast) dans la procession des beasts
        """
        return START_TS + botte_position * PAS_TD

    # def pad(num):
    #     return "\t"  * int(num)
    # on récupère la chaine de caractère dans l'objet
    _procession = procession_.values[0]

    # on construit un dictionnaire avec
    # check https://docs.python.org/3/library/os.html#os.cpu_count
    nb_processors = len(os.sched_getaffinity(0))
    logger.info(f"{name} hoping with {nb_processors} Thread / Node")

    # tout les motifs and sequence on la même taille
    NB_MOTIFS = f"{len(motifs_):->9,}".replace(",", " ")

    with Pool() as pool:
        D = {}
        for i, mot in enumerate(motifs_):
            matchP_name = f"matchP-{i:->8,}".replace(",", " ")
            if i % max(int(len(motifs_) / 50), 1) == 0:
                # printing only 50 of those
                logger.info(f"{matchP_name} sur {NB_MOTIFS}")

            kwargs = {
                "mot": mot,
                "procession": _procession,
            }

            D[tuple(mot)] = list(
                map(_trsf_botte_pos_in_ts, pool.apply(find_patternP, (), kwargs))
            )

    # on consitue une df avec le dictionnaire.
    # celle ci aura un multi-index

    LEN_MOTIF = len(motifs_[0])
    LEN_SEQ = len(_procession)

    _df = DataFrame(Series(D))
    _df.columns = ["mot_pos"]

    if with_detail:
        _df.loc[:, "fmot"] = _df.loc[:, "mot_pos"].apply(len) / len(_procession)
        _df.loc[:, "nb_match"] = _df.loc[:, "mot_pos"].apply(len)

        # la fréquence in sequence n'est pas un pourcentage.  mas une mesure de présence
        # S'il y a des overlap alors un mot peut être compter plusieurs fois en partie
        _df.loc[:, "fmot_in_seq"] = _df.nb_match / (LEN_SEQ / LEN_MOTIF) * 100

        # on réordonne les colonnes
        _df = _df.loc[:, ["fmot", "fmot_in_seq", "nb_match", "mot_pos"]]

    # on crée un colonne pour la mettre en index plus tard
    # préférable de sortindex
    # _df = _df.loc[_df.mot_pos.apply(len).sort_values().index]
    _df.loc[:, "seq_num"] = SEQ_NUM
    _df = _df.set_index("seq_num", append=True).sort_index()

    _df.columns.name = procession_.index.name
    _df.index.name = f"{SEQ_NUM}_mot_{LEN_MOTIF}"

    # on  retravail l'index pour y faire apparaitre le numéro de la série
    return _df


def find_pattern(pat: str, in_chaine) -> List:
    """
    Renvois les indices de début du pattern pat
    dans in_chaine
    le pattern doit être de la form p(?=rp) si
    l'on veut les objects qui se superposent
    """
    return [match.start() for match in re.finditer(re.compile(pat), in_chaine)]


def find_patternP(mot: str, procession) -> Series:
    """
    Find all mot in the procession of letter.
    Returns the results in the Q.
    Print some loging information with i the o
    """
    return find_pattern(f"{mot[0]}(?={''.join(mot[1:])})", procession)


def get_matches_stat(s_: Series) -> DataFrame:
    """
    Crée à partir de la serie de positions des statistiques descriptives
    nb_match : le nombres d'occurence du mots dans le corpus
    fmot : fréquence du mot dans le corpus
    fmot_in_seq : la fréquence du mot dans les séquences, ie le nombre
    d'occurence d'un mot rapporté à la longueur
    """
    _nb_matches = s_.apply(len)
    df = DataFrame({"nb_match": _nb_matches, "fmot": _nb_matches / len(s_) * 100})
    return df


def motifs_matches_inall(processions_, mots_: Sequence, with_detail=False) -> DataFrame:
    """
    Pour chacune des processions, trouve la date de départ d'un de mots_
    de la Séquences fournis.  Les mots doivent pouvoir être touvé dans l'index
    de la procession.
    Renvois un dataFrame avec un colonne contenant la séquence des dates
    et pour index les mot + le numéro de la séquence
    """

    def _flatten_mot_pos(s_: Series):
        """Concatène les liste 'valeur' de la Séries"""
        return [elt for list_ in s_.values for elt in list_]

    # la taille des mots,
    LEN_MOT = len(mots_[0])
    COL_NAME = processions_.columns.name
    # IDX_NAME = processions_.index.name

    # construit la dataframe avec toute les positions
    logger.info(f"Searching {LEN_MOT} letters words")
    _df = DataFrame(None)
    # with Pool() as pool:
    for i in range(len(processions_)):
        name = f"processions n{i:->4}"
        print(f"Starting {name} sur {len(processions_)}")
        kwargs = {
            "name": name,
            "procession_": processions_.iloc[i],
            "motifs_": mots_,
            "with_detail": False,
        }
        # print(f"type >>> {type(kwargs['procession_'])}")
        # _stat = pool.apply(motifs_matches_inoneP, kwargs)
        _stat = motifs_matches_inone(**kwargs)
        _df = _stat if not len(_df) else concat([_df, _stat])

    _df = _df.sort_index()  # to avoird lexsort depth performance warning

    # ## regroupe toute les positions des séquences en une seule
    gps_mot = _df.groupby(level=list(range(LEN_MOT)))

    # on créer une nouvelle df avec les listes des positions
    # on concatène les positions de départs des mots dans les différents
    # séquence en une seule séquence
    D = {}
    for i, mot in enumerate(mots_):
        print(f"{i+1:>9d}/{len(mots_)}", end="\r")
        gp_mot = gps_mot.get_group(tuple(mot))
        D[tuple(mot)] = _flatten_mot_pos(gp_mot.mot_pos)

    _mot_pos = Series(D, name="mot_pos")

    if with_detail:
        _df = concat([get_matches_stat(_mot_pos), _mot_pos], axis=1)
    else:
        _df = DataFrame(_mot_pos)

    _df.index.name = f"mot_{LEN_MOT}"
    _df.columns.name = COL_NAME
    return _df.sort_index()


def load_data(folder: Path, fname: str, action: str, action_baggage=None):
    """Charge les données en mémoire"""
    #  ça fait un gros fichier 500M, compressé
    _fname: Path = folder.joinpath(fname)
    logging.info(f"Action '{action}' {_fname} with baggage '{action_baggage}'")
    if action == "save":
        vedf = action_baggage["vedf"]
        fedf = action_baggage["fedf"]
        with open(_fname, "bw") as f:
            pickle.dump((vedf, fedf), f)
    elif action == "create":
        vedf, fedf = prepare_data(
            [2018, 2019, 2020, 2021],
            bins="1m",
            dirname=folder,
            bname="ADAUSD",
            source_dir="Kraken",
        )
        with open(_fname, "bw") as f:
            pickle.dump((vedf, fedf), f)
    elif action == "load":
        with open(_fname, "br") as f:
            (vedf, fedf) = pickle.load(f)
            vedf.head(3), f"{len(vedf)}"

    return (vedf, fedf)


def main(
    folder,
    action="load",
    ts_shift=None,
    pipe_td="1d",
    motifs_lens=[6],
):
    """
    Executing main élement of the programme
    loading the data,
    sequencing it, extraction of beasts and analysis of motifs
    """
    assert action in ["load", "save", "create"], f"action={action}"

    vedf, fedf = load_data(
        folder=folder,
        fname="vedf-fedf.pkl",
        action=action,
    )

    # data sampled with rolling
    data = sdata_f(fedf, frac=1, by_=pipe_td, cols=LHc)

    # à partir du jeu de donnée, on créer plusieurs
    # indexe de séries de pas 'BY' mais décalées.
    # elle nous servirons à calculers les bêtes pour
    # le pas "BY"
    _ts_shift = data.index.freqstr if ts_shift is None else ts_shift
    _sequences_indexes = create_indexes_en_creneaux(
        data, ts_pas=pipe_td, freq_décalage=_ts_shift
    )

    # Pour chaque série on calcule les bêtes
    # et renvoyons une serie avec multiIndex inclant la
    # séquence à laquellle appartient la beast
    sequence_in_beasts = make_sequence_in_beasts(
        data, sequences_indexes_=_sequences_indexes, ts_pas=pipe_td
    )

    # On regroupe les données selon la séquence.
    # pour chaque séquence on résume les bêtes avec une succesion de lettres
    # on appelle cela la procession
    beast_processions = build_beast_processsions(sequence_in_beasts)
    beast_processions.to_csv(f"beast_procession_{pipe_td}.csv")

    # on ignore les zébres extrèment rares
    # symbols = set(BEASTS) - set(("zebre","dragon", "sheep"))
    # on créer l'ensemble des mots que nous allons soumettre à l'évaluation
    # un mot est une petite séquence de beast calculer avec la fréquence "BY"
    # la taille d'un mot est limité par la puissance de calcule, la sa généricité.
    # compter ~ 6 lettres.  Changer la taille du "BY" pour couvrir
    # de plus grandes périodes de temps
    motifs_lens = sorted(motifs_lens)
    MAX_LEN_MOTIFS = motifs_lens[-1]

    MOTIFS = create_dico_de_motifs_sets(motifs_lens, beast_processions)
    # https://stackoverflow.com/questions/2846653/how-can-i-use-threading-in-python#28463266
    # check a multiprocess here and then a mutlithreading
    # sa permettrait peut-être d'utiliser les processeurs des différents noeud

    for len_mot in motifs_lens:
        logger.info(f"Searching motif of len {len_mot:>2}/{MAX_LEN_MOTIFS}")
        motifs_matches = motifs_matches_inall(
            processions_=beast_processions,
            mots_=list(MOTIFS[len_mot]),
            with_detail=True,
        )

        logger.info(f"Sorting and computing conditional proba {len_mot:>9}.")
        sorted_data = ordonne_motifs(motifs_matches, MOTIFS)

        bname = f"motifs_{sorted_data.columns.name}_mot{len_mot}"
        fname_csv = folder.joinpath(f"{bname}.csv")

        with open(fname_csv, "w") as f:
            logger.info(f"Saving to {fname_csv}")
            sorted_data.loc[
                :, ["freq_ll_s_mot", "fmot", "nb_match", "prod_freq"]
            ].to_csv(f)

        fname_pkl = folder.joinpath(f"{bname}.pkl")
        with open(fname_pkl, "bw") as f:
            logger.info(f"Saving to {fname_pkl}")
            pickle.dump(sorted_data, f)

    logger.info("End.")
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logLevel",
        "-L",
        default="INFO",
    )
    parser.add_argument(
        "--folder",
        "-f",
        help=(
            "In which folder store the data?  The folder_data containing data file "
            " will be a subfolder with the asset pair name"
        ),
        type=str,
        default="./Kraken",
    )
    parser.add_argument(
        "--action",
        "-A",
        help=("action to load, or recompute the data file"),
        type=str,
        default="load",
    )
    parser.add_argument(
        "--ts_shift",
        "-s",
        help=(
            "The data should have a resolution of 1m and when computing the rolling pipe, we can extend the data by shifting it by one minutes.  but this may be too much for test case, so ts_shift enable to set how to shift the data.  If not will use the data resolution"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pipe_td",
        "-p",
        help=(
            "Data for pipe is computed on rolling data using aggregation function of a time delta.  This could be 1d for 1 day pipe or 15m or 1h or any multiple of those unites"
        ),
        type=str,
        default="1d",
    )
    parser.add_argument(
        "--motifs_lens",
        "-m",
        nargs="+",
        default=[2],
        help=("La liste des longueurs de mots qui vont être cherchés dans le corpus."),
        type=int,
    )

    return parser.parse_args()


def main_prg():
    args = parse_args()

    logger.setLevel(args.logLevel)

    # execute
    main(
        folder=Path(args.folder),
        action=args.action,
        ts_shift=args.ts_shift,
        pipe_td=args.pipe_td,
        motifs_lens=args.motifs_lens,
    )


if __name__ == "__main__":
    main_prg()
