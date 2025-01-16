import numpy
import math
import matplotlib.pyplot as plt
import time #permet de tester le temps que met notre algorithme

""" Fonction permettant de centrer une matrice image à partir de x,y (qu'on tronque ici)"""
def matrice_centre(img, x_tronque, y_tronque):
    M_tronque = img[0:y_tronque, 0:x_tronque, 0:3]
    M_centre = M_tronque*255 - 128
    
    return M_centre

""" Fonction permettant de découper la matrice centré en bloc de 8"""
def matrice_centre_bloc(M_centre,x_tronque,y_tronque):
    M_centre_bloc = []
    for y in range(0, y_tronque, 8):
        for x in range(0, x_tronque, 8):
            M_centre_bloc.append(M_centre[y:y + 8, x:x + 8]) #attention: maintenant notre M_centre_bloc a 4 dimensions    
    return M_centre_bloc


"""Compression"""
Q=numpy.array([[16,11,10,16,24,40,51,61],
            [12,12,13,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]])

def funct_compression(M, x_tronque, y_tronque,P): #coder pour compresser une matrice à 3 canaux
    compression = []
    coef_nonNuls = 0
    
    for i in range(3): #on le fait pour chaque canal de couleur
        for sous_matrice in M:
            produit_intermediaire = numpy.matmul(P, sous_matrice[:,:,i])
            D = numpy.matmul(produit_intermediaire, numpy.transpose(P))
            M_compression1 = numpy.divide(D,Q)
            M_compression1 = numpy.trunc(M_compression1) #partie entiere
            compression.append(M_compression1)
            coef_nonNuls += numpy.count_nonzero(M_compression1[i])
    
    taux_compression = (1-coef_nonNuls/(y_tronque*x_tronque*3))*100 #en pourcentage
    print("Le taux de compression est de :", taux_compression, " %")
    
    return compression


def funct_decompression(compressionMat,P):
    decompression = []
    for sous_matrice in compressionMat:
        M_decompression_tmp = sous_matrice * Q
        M_decompression_tmp = numpy.matmul(numpy.transpose(P),M_decompression_tmp)
        M_decompression = numpy.matmul(M_decompression_tmp, P)
        decompression.append(M_decompression)
    
    return decompression


def reassemblerMatrice(decompression, x_tronque, y_tronque):    
    """ On recompose la matrice en extrayant chacune des couleurs (RGB)"""
    matrice_reassemble = numpy.zeros((y_tronque,x_tronque,3)) #Initialisation
    longueur = numpy.shape(decompression)[0] #le nombre de matrice 8x8
    
    """ On vectorise la matrice de decompression pour chacune des couleurs"""
    vect_red = numpy.ravel(decompression[0:longueur//3]) #On vectorise pour la matrice rouge
    vect_green = numpy.ravel(decompression[longueur//3:(2*longueur)//3]) #On vectorise pour la matrice verte
    vect_blue = numpy.ravel(decompression[(2*longueur)//3:longueur])#On vectorise pour la matrice bleu
    
    """ On reassemble avec la fonction reshape de la librairie numpy"""
    n = 0
    for y in range(0,y_tronque,8): #on fait des pas de 8
      for x in range(0,x_tronque,8): #on fait des pas de 8
        matrice_reassemble[y:y+8,x:x+8,0] = numpy.reshape(vect_red[64*n : 64*(n+1)],(8,8)) #premier canal = rouge
        matrice_reassemble[y:y+8,x:x+8,1] = numpy.reshape(vect_green[64*n : 64*(n+1)],(8,8)) #deuxieme canal = vert
        matrice_reassemble[y:y+8,x:x+8,2] = numpy.reshape(vect_blue[64*n : 64*(n+1)],(8,8)) #troisieme canal =bleu
        n+=1
    
    return matrice_reassemble


def pourcentage_erreur(matrice_image,matrice_recompose):
    erreur = 0
    for k in range(3):
        erreur += (numpy.linalg.norm(matrice_image[:,:,k] - matrice_recompose[:,:,k])) / numpy.linalg.norm(matrice_image[:,:,k])
    return "Le pourcentage d'erreur est de : " + str((erreur/3)*100) + " %"


if __name__ == '__main__':
    
    # Sélection de l'image à partir de la boîte de dialogue
    
    
    """ Ici on commence à mesurer le temps seulement après avoir choisit l'image"""
    tps1 = time.perf_counter()

    """Calcul de P """
    P = numpy.zeros((8, 8))  # initialisation de P
    c0 = 1 / (numpy.sqrt(2))  # c0 = 1/sqrt(2) d'après la consigne

    # Initialisation de la première ligne en dehors de la boucle
    for j in range(8):
        P[0, j] = 0.5 * c0

    # Boucle pour les autres lignes
    for i in range(1, 8):
        for j in range(8):
            P[i, j] = 0.5 * c0 * numpy.cos(((2 * j + 1) * i * math.pi) / 16)



    img = plt.imread('pns_original.png')
    y, x, rgb = numpy.shape(img)
    y_tronque = (y // 8) * 8
    x_tronque = (x // 8) * 8

    M_centre = matrice_centre(img, x_tronque, y_tronque)
    M_centre_bloc = matrice_centre_bloc(M_centre,x_tronque,y_tronque)

    compression = funct_compression(M_centre_bloc, x_tronque, y_tronque, P)
    decompression = funct_decompression(compression, P)
    recomposition = reassemblerMatrice(decompression, x_tronque, y_tronque)

    """On affiche l'image"""
    recomposition += 128
    recomposition /= 255
    recomposition = numpy.clip(recomposition, 0, 1)
    plt.imshow(recomposition)

    """On affiche l'erreur après avoir remis les éléments de la matrice de recomposition entre 0 et 1"""
    print(pourcentage_erreur(img, recomposition))
    tps2 = time.perf_counter()
    print("Le temps d'execution est de : ", tps2 - tps1, "secondes")
    
    plt.imsave("img_recompose.png",recomposition) # enregistrement de la nouvelle image