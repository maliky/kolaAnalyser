"""
Des fonctions qui pourraient avoir un utilité ou être source d'inspiration
"""
def sous_motif_se_répète(motif):
    """Renvois true si un sous motif se répète. sur tout le motif"""
    taille_max_sous_motif = len(motif) // 2
    for n in range(1, taille_max_sous_motif + 1):
        sous_motif = f"^({motif[:n]})+$"
        if re.fullmatch(re.compile(sous_motif), motif):
            return True
    return False



def motifs_sans_répétition(motifs):
    """
    Filtre la liste de motifs
    et renvoie une liste de motifs sans répétition"""
    res = []
    for i, motif in enumerate(motifs):
        print(f"{i:>9d}/{len(motifs)}", end="\r")
        if not sous_motif_se_répète(motif):
            res += motif
    return res


