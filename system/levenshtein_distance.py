
def minimim(i, j, k):
    """
    Returns the minimum value
    :param i:
    :param j:
    :param k:
    :return:
    """
    if i < j:
        if k < i:
            return k
        else:
            return i
    else:
        if k < j:
            return k
        else:
            return j


def levenshtein(chaine1, chaine2):
    """
    Computes the Levenshtein distance iteratively
    :param chaine1: first chain
    :param chaine2: second chain
    :return: Levenshtein distance between chain 1 and 2
    """
    size1 = len(chaine1)
    size2 = len(chaine2)

    distance = {}
    indice = (0, 0)
    distance[indice] = 0

    for i in range(0, size1+1):
        indice = (i, 0)
        distance[indice] = i

    for j in range(0, size2+1):
        indice = (0, j)
        distance[indice] = j

    for i in range(1, size1+1):
        for j in range(1, size2+1):
            indice = (i, j)
            prec1 = (i, j-1)
            prec2 = (i-1, j)
            prec3 = (i-1, j-1)
            ajout = 1
            if chaine1[i-1] == chaine2[j-1]:
                ajout = 0
            distance[indice] = minimim(distance[prec1]+1, distance[prec2]+1, distance[prec3]+ajout)

    indice = (size1, size2)
    return distance[indice]

