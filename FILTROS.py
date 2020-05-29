import numpy             as np
import matplotlib.pyplot as plt
import imageio

mascara_identidade       = np.array             ([[0, 0,  0],  [0, 1,   0],    [0, 0,  0]])
mascara_borda            = np.array             ([[0, 1,  0],  [1, -4,  1],    [0, 1,  0]])
mascara_afiado           = np.array             ([[0, -1, 0],  [-1, 5, -1],    [0, -1, 0]])
mascara_borrao           = (1/9) * np.array     ([[1, 1,  1],  [1, 1,   1],    [1, 1,  1]])
mascara_borrao_gaussiano = (1/16) * np.array    ([[1, 2,  1],  [2, 4,   2],    [1, 2,  1]])

def computar_histograma(imagem):
    assert len(imagem.shape) == 2
    histograma = {}
    for tom in imagem.flatten():
        assert tom >= 0 and tom <= 255
        if tom not in histograma:
            histograma[tom] = 0
        histograma[tom] += 1
    return histograma


def plotar_histograma(histograma):
    plt.bar(histograma.keys(), histograma.values(), color='black')
    plt.show()


def binarizar(imagem, k):
    assert len(imagem.shape) == 2
    largura, altura = imagem.shape
    nova_imagem = imagem.copy()
    for i in range(largura):
        for j in range(altura):
            if imagem[i, j] > k:
                nova_imagem[i, j] = 255
            else:
                nova_imagem[i, j] = 0
    return nova_imagem

def dividir_imagem_em_canais_de_cor(imagem):
    canais = np.dsplit(imagem, imagem.shape[-1])
    for canal in canais:
        yield canal[..., 0]


def mesclar_canais_de_cor_em_imagem(canais):
    return np.dstack(canais)

def convolucao(imagem, mascara):
    mascara = mascara[::-1, ::-1]
    return correlacao(imagem, mascara)

def correlacao(imagem, mascara):
    assert len(imagem.shape) == 2
    largura, altura = imagem.shape
    nova_imagem = imagem.copy()

    for i in range(1, largura - 1):
        for j in range(1, altura - 1):
            
            vizinhos = np.array(
                 [[imagem[i-1, j-1], imagem[i, j-1], imagem[i+1, j- 1]],
                 [imagem[i-1,    j], imagem[i,   j], imagem[i+1,    j]],
                 [imagem[i-1,  j+1], imagem[i, j+1], imagem[i+1, j+1]]]
            )

            if callable(mascara):
                pixel_com_mascara_aplicada = mascara(vizinhos)
            else:
                pixel_com_mascara_aplicada = np.sum(vizinhos * mascara)

            pixel_com_mascara_aplicada = np.clip(pixel_com_mascara_aplicada, 0, 255)
            nova_imagem[i, j] = pixel_com_mascara_aplicada

    return nova_imagem

def filtro_sobel(imagem):
    mascara_x = np.array([[-1, 0,   1], [-2, 0, 2], [-1, 0, 1]])
    mascara_y = np.array([[-1, -2, -1], [0, 0,  0], [1, 2,  1]])

    nova_imagem_x = convolucao(imagem, mascara_x)
    nova_imagem_y = convolucao(imagem, mascara_y)
    return np.clip(np.hypot(nova_imagem_x, nova_imagem_y), 0, 255).astype(np.uint8)


def filtro_prewitt(imagem):
    mascara_x = np.array([[-1, 0,   1], [-1, 0, 1], [-1, 0, 1]])
    mascara_y = np.array([[-1, -1, -1], [0, 0,  0], [1, 1,  1]])

    nova_imagem_x = convolucao(imagem, mascara_x)
    nova_imagem_y = convolucao(imagem, mascara_y)
    return np.clip(np.hypot(nova_imagem_x, nova_imagem_y), 0, 255).astype(np.uint8)

def filtro_mediana(vizinhos):
    vizinhos_ordenados = sorted(vizinhos.flatten())
    indice_meio        = len   (vizinhos_ordenados) // 2
    return vizinhos_ordenados[indice_meio]

def filtro_media(vizinhos):
    vizinhos_somados      = np.sum(vizinhos)
    return vizinhos_somados // len(vizinhos)

def quantizar(imagem, k):
    assert len(imagem.shape) == 2
    tons_por_k      = 256 // k
    largura, altura = imagem.shape
    nova_imagem     = imagem.copy()
    
    for i in range(largura):
        for j in range(altura):
            resto = nova_imagem[i, j] % tons_por_k
            nova_imagem[i, j] += tons_por_k - resto - 1
    return nova_imagem

def girar_imagem(imagem):
    assert len(imagem.shape) == 2 
    largura, altura = imagem.shape
    nova_imagem     = imagem.copy() 

    for i in range( largura ):
        for j in range(altura):
            nova_imagem[i, j] = imagem[j, i]

    return nova_imagem

def equalizar(imagem):
    assert len(imagem.shape) == 2
    histograma = computar_histograma(imagem)
    largura, altura = imagem.shape
    ideal = (largura * altura) // 256
    acumulador = 0
    histograma_equalizado = {}
    for tom in range(256):
        if tom in histograma:
            acumulador += histograma[tom]
        nova_intensidade = (acumulador // ideal) - 1
        if nova_intensidade < 0: 
            histograma_equalizado[tom] = 0
        else:
            histograma_equalizado[tom] = nova_intensidade
    nova_imagem = imagem.copy()
    for i in range(largura):
        for j in range(altura):
            nova_imagem[i, j] = histograma_equalizado[imagem[i, j]]
    return nova_imagem


def aplicar_funcao_em_canais(funcao, imagem, *args):
    def aplicar():
        for canal in dividir_imagem_em_canais_de_cor(imagem):
            yield funcao(canal, *args)

    return mesclar_canais_de_cor_em_imagem(list(aplicar()))


def run(imagem):

    girar = aplicar_funcao_em_canais(girar_imagem, imagem)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/girar.png',       girar             )    

    binarizado = aplicar_funcao_em_canais(binarizar, imagem, 127)                                                           #1
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/binarizado_cor.png',       binarizado      )
    
    identidade = aplicar_funcao_em_canais(convolucao, imagem, mascara_identidade)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/identidade_cor.png',       identidade      )   #2
    
    borda = aplicar_funcao_em_canais(convolucao, imagem, mascara_borda)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/borda_cor.png',            borda           )   #3
    
    afiado = aplicar_funcao_em_canais(convolucao, imagem, mascara_afiado)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/afiado_cor.png',           afiado          )   #4
    
    borrao = aplicar_funcao_em_canais(convolucao, imagem, mascara_borrao)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/borrao_cor.png',           borrao          )   #5
    
    borrao_gaussiano = aplicar_funcao_em_canais(convolucao, imagem, mascara_borrao_gaussiano)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/borrao_gaussiano_cor.png', borrao_gaussiano)   #6
    
    media = aplicar_funcao_em_canais(correlacao, imagem, filtro_media)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/media_cor.png',            media           )   #7
    
    mediana = aplicar_funcao_em_canais(correlacao, imagem, filtro_mediana)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/mediana_cor.png',          mediana         )   #8
    
    sobel = aplicar_funcao_em_canais(filtro_sobel, imagem)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/sobel_cor.png',            sobel           )   #9
    
    prewitt = aplicar_funcao_em_canais(filtro_prewitt, imagem)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/prewitt_cor.png',          prewitt         )   #10
       
    quantizado_32 = aplicar_funcao_em_canais(quantizar, imagem, 32)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/quantizado_32_cor.png',    quantizado_32   )   #11
    
    quantizado_8 = aplicar_funcao_em_canais(quantizar, imagem, 8)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/quantizado_8_cor.png',     quantizado_8    )   #...
    
    equalizado = aplicar_funcao_em_canais(equalizar, imagem)
    imageio.imwrite('/home/faculdade/git/digital_image_processing/exemplares/equalizado_cor.png',       equalizado      )   #12

lenna = imageio.imread ('lenna.jpg')
run                    (      lenna)
