# -*- coding: utf-8 -*-
"""Functions to computer market probabilities or frequences"""
from typing import List, Dict
from pandas import DataFrame, Series, concat


def ordonne_motifs(df_: DataFrame, motifs: Dict) -> DataFrame:
    """
    Ordonne les motifs (mots) selon le produit de leur
    fréquence et de la fréquence d'avoir la dernière lettre
    connaissant le mot précédent.
    """

    freq_cond = freq_last_lettre_sachant_mots(df_, motifs)
    prod_freq = Series(df_.fmot * freq_cond, name="prod_freq")
    _df = concat([freq_cond, prod_freq, df_], axis=1)
    _df.columns.name = df_.columns.name
    return _df.sort_values(by="prod_freq")


def freq_l_sachant_mot(df_: DataFrame, last_lettre: str, mot_: str) -> Series:
    """
    Probabilité conditionnel de la dernière lettre sachant le mot
    """
    freq_mot = get_freq_mot(df_, mot_)
    freq_l_et_mot = get_freq_mot(df_, mot_ + last_lettre)
    freq = freq_l_et_mot / freq_mot
    return freq * 100


def freq_last_lettre_sachant_mots(df_: DataFrame, motifs) -> Series:
    """
    renvois une série avec les probabilités conditionnel de la dernière lettre
    du mot connaissant les 1ère lettres
    """
    LEN_MOT = len(df_.index[0])
    D = {}
    for i, (mot_nm1, last_letter) in enumerate(
        [(mot[:-1], mot[-1]) for mot in motifs[LEN_MOT]]
    ):
        print(f"{i+1}/{len(motifs[LEN_MOT])}", end="\r")
        D[(*mot_nm1, last_letter)] = freq_l_sachant_mot(df_, last_letter, mot_nm1)

    return Series(D, name="freq_ll_s_mot").fillna(0).sort_values()


def freq_mot_sachant_l(df_: DataFrame, last_lettre: str, mot_: str) -> Series:
    """
    Probabilité conditionnel d'un mot de n-1 connaissant la dernière lettre
    """
    freq_last_lettre = get_freq_last_letter(df_, last_lettre)
    freq_l_et_mot = get_freq_mot(df_, mot_ + last_lettre) / df_.nb_match.sum() * 100
    freq = freq_l_et_mot / freq_last_lettre
    return freq * 100


def get_freq_mot(df_: DataFrame, mot_: str = "llll") -> Series:
    """
    Renvois la fréquence du mot_ dans df_ en %
    C'est à dire le nombre d'occurence rapporter au nombre total d'occurences
    """
    assert len(mot_) <= len(
        df_.index[0]
    ), f"len(mot_)={len(mot_)}, mot_={mot_}, len(df_.index[0])={len(df_.index[0])}"

    matches = df_.nb_match
    freq = matches.loc[tuple(mot_)].sum() / matches.sum()
    return freq * 100


def get_freq_mot_droit(df_, mot_="l"):
    """
    Renvois la fréquence du mot en commençant par la droite
    at the end of the words of df_
    """
    _nb_levels = len(df_.index[0])

    assert (
        len(mot_) <= _nb_levels
    ), f"len(mot)={len(mot_)}, mot_={mot_}_nb_levels={_nb_levels}"

    # le truc c'est de créer un un index avec des slice none à gauche
    # pour compléter le mot que l'on passe
    _none = tuple([slice(None)] * (_nb_levels - len(mot_)))
    _matches = df_.nb_match
    _freq = _matches.loc[(*_none, *tuple(mot_))].sum() / _matches.sum()
    # ret = Series(_freq.sort_values(), name=f"freq_{mot_}_droit")
    return _freq * 100


def get_freq_last_letter(df_, last_letter="l"):
    """
    Renvois la fréquence de la lettre last_letter
    at the end of the words of df_
    """
    assert len(last_letter) == 1, f"len(last_letter)={len(last_letter)}"

    return get_freq_mot_droit(df_, last_letter)


def get_mot_n_moins_un(df: DataFrame) -> List:
    """
    Renvois l'ensemble des mots qui apparaissent
    dans le df et qui ont une longueur réduite
    de un.
    Les mots doivent être dans l'index du df.
    """
    mot_df_nm1 = set()
    for mot_df in df.index:
        mot_df_nm1 |= {mot_df[:-1]}
    return list(mot_df_nm1)
