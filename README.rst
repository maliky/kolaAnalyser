kolaAnalyser et un programme qui analyser les cours d'un marché donné
sous forme de prix holc .

Il transforme les données holc en chaines de charactères. Il y applique
des techniques de TAL pour découvrir de motifs (mot) plus ou moins
récurents et calculer des probabilités (fréquences) conditionnelles.

Notons :math:`h_n` et :math:`l_n` le maximun et minimun d'une pipe au
temps :math:`n` time\ :sub:`unit`. Alors selon le rapport avec les
:math:`l_{n-1}` et :math:`h_{n-1}` nous définissons les types de pipes
suivantes:

-  :math:`h_n` > :math:`h_{n-1}` ET :math:`l_n` >= :math:`l_{n-1}` ->
   bull (l), augmentation des prix ;
-  :math:`h_n` <= :math:`h_{n-1}` ET :math:`l_n` < :math:`l_{n-1}` ->
   bear (r), baisse des prix ;
-  :math:`h_n` < :math:`h_{n-1}` ET :math:`l_n` > :math:`l_{n-1}` ->
   sheep (p), contraction du prix ;
-  :math:`h_n` > :math:`h_{n-1}` ET :math:`l_n` < :math:`l_{n-1}` ->
   dragon (n), expansion des prix ;
-  :math:`h_n` == :math:`h_{n-1}` ET :math:`l_n` == :math:`l_{n-1}` ->
   dragon (n), contraction des prix ;

la \`time\ :sub:`unit`\ \` est l'unité d'aggrégation des données et de
calculs des pipes. Elle est définie par \`–pipe\ :sub:`td`\ \`.

Les données ayant une résolution d'une minutes et comme kolaAnalyser
utilise les fenêtre roulante pour le calcul des max et min, il est
possible de scinder les données initiale en définissant un décalage <
pipe\ :sub:`td` au départ de chaque série de prix. C'est le role de
\`ts\ :sub:`shift`\ \`.

Utilisation
===========

.. code:: bash

   usage: kolaAnalyse [-h] [--logLevel LOGLEVEL] [--folder FOLDER] [--action ACTION] [--ts_shift TS_SHIFT] [--pipe_td PIPE_TD] [--motif_max_len MOTIF_MAX_LEN]

   Un script qui analyse des motifs dans une base de données de prix. Le but est de trouver des motifs fréquents ou intérssant dans divers sur des timesframe (window) par
   classique ex. 7, 11, 13, 17, 19, 23, 31, 37, 41, 47, 49, 53, 59 minutes. (nombre entiers)

   optional arguments:
     -h, --help            show this help message and exit
     --logLevel LOGLEVEL, -L LOGLEVEL
     --folder FOLDER, -f FOLDER
                           In which folder store the data? The folder_data containing data file will be a subfolder with the asset pair name (default: ./Kraken)
     --action ACTION, -A ACTION
                           action to load, or recompute the data file (default: load)
     --ts_shift TS_SHIFT, -s TS_SHIFT
                           The data should have a resolution of 1m and when computing the rolling pipe, we can extend the data by shifting it by one minutes. but this may be too
                           much for test case, so ts_shift enable to set how to shift the data. If not will use the data resolution (default: None)
     --pipe_td PIPE_TD, -p PIPE_TD
                           Data for pipe is computed on rolling data using aggregation function of a time delta. This could be 1d for 1 day pipe or 15m or 1h or any multiple of
                           those unites (default: 1d)
     --motif_max_len MOTIF_MAX_LEN, -m MOTIF_MAX_LEN
                           We will be searching and saving statistiques for motifs from len 1 to motif_max_len. Set it here. Carrefull this can explode (default: 7)


