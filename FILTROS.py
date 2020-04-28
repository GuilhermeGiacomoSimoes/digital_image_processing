import numpy as np
import matplotlib.pyplot as plt
import imageio

mascara_identidade = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
mascara_borda = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
mascara_afiado = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
mascara_borrao = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
mascara_borrao_gaussiano = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

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


def converter_para_cinza(imagem):
    assert len(imagem.shape) == 3
    largura, altura, *_ = imagem.shape
    imagem_cinza = np.zeros((altura, largura), dtype=imagem.dtype)
    for i, linha in enumerate(imagem):
        for j, (r, g, b) in enumerate(linha):
            tom_de_cinza = 0.2126 * r + 0.7152 * g + 0.0722 * b
            assert tom_de_cinza >= 0 and tom_de_cinza <= 255
            imagem_cinza[i, j] = tom_de_cinza
    return imagem_cinza


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


def correlacao(imagem, mascara):
    assert len(imagem.shape) == 2
    largura, altura = imagem.shape
    nova_imagem = imagem.copy()
    for i in range(1, largura - 1):
        for j in range(1, altura - 1):
            vizinhos = np.array(
                [[imagem[i-1, j-1], imagem[i, j-1], imagem[i+1, j-1]],
                 [imagem[i-1, j], imagem[i, j], imagem[i+1, j]],
                 [imagem[i-1, j+1], imagem[i, j+1], imagem[i+1, j+1]]]
            )
            if callable(mascara):
                pixel_com_mascara_aplicada = mascara(vizinhos)
            else:
                pixel_com_mascara_aplicada = np.sum(vizinhos * mascara)
            pixel_com_mascara_aplicada = np.clip(pixel_com_mascara_aplicada, 0, 255)
            nova_imagem[i, j] = pixel_com_mascara_aplicada
    return nova_imagem


def convolucao(imagem, mascara):
    mascara = mascara[::-1, ::-1]
    return correlacao(imagem, mascara)

def filtro_sobel(imagem):
    mascara_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mascara_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    nova_imagem_x = convolucao(imagem, mascara_x)
    nova_imagem_y = convolucao(imagem, mascara_y)
    return np.clip(np.hypot(nova_imagem_x, nova_imagem_y), 0, 255).astype(np.uint8)


def filtro_prewitt(imagem):
    mascara_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    mascara_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    nova_imagem_x = convolucao(imagem, mascara_x)
    nova_imagem_y = convolucao(imagem, mascara_y)
    return np.clip(np.hypot(nova_imagem_x, nova_imagem_y), 0, 255).astype(np.uint8)

def filtro_roberts(imagem):
    mascara_x = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    mascara_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    nova_imagem_x = convolucao(imagem, mascara_x)
    nova_imagem_y = convolucao(imagem, mascara_y)
    return np.clip(np.hypot(nova_imagem_x, nova_imagem_y), 0, 255).astype(np.uint8)

def filtro_mediana(vizinhos):
    vizinhos_ordenados = sorted(vizinhos.flatten())
    indice_meio = len(vizinhos_ordenados) // 2
    return vizinhos_ordenados[indice_meio]


def filtro_media(vizinhos):
    vizinhos_somados = np.sum(vizinhos)
    return vizinhos_somados // len(vizinhos)

def ordem_k(k):
    def ordem_com_k(vizinhos):
        vizinhos_ordenados = sorted(vizinhos.flatten())
        return vizinhos_ordenados[k]
    return ordem_com_k


def filtro_moda(vizinhos):
    histograma = computar_histograma(vizinhos)
    moda = np.argmax(histograma)
    assert moda >= 0 and moda <= 255
    return moda


def quantizar(imagem, k):
    assert len(imagem.shape) == 2
    tons_por_k = 256 // k
    largura, altura = imagem.shape
    nova_imagem = imagem.copy()
    for i in range(largura):
        for j in range(altura):
            resto = nova_imagem[i, j] % tons_por_k
            nova_imagem[i, j] += tons_por_k - resto - 1
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


def rodar_em_cor(imagem):
    binarizado = aplicar_funcao_em_canais(binarizar, imagem, 127)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/binarizado_cor.png', binarizado)
    identidade = aplicar_funcao_em_canais(convolucao, imagem, mascara_identidade)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/identidade_cor.png', identidade)
    borda = aplicar_funcao_em_canais(convolucao, imagem, mascara_borda)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/borda_cor.png', borda)
    afiado = aplicar_funcao_em_canais(convolucao, imagem, mascara_afiado)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/afiado_cor.png', afiado)
    borrao = aplicar_funcao_em_canais(convolucao, imagem, mascara_borrao)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/borrao_cor.png', borrao)
    borrao_gaussiano = aplicar_funcao_em_canais(convolucao, imagem, mascara_borrao_gaussiano)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/borrao_gaussiano_cor.png', borrao_gaussiano)
    moda = aplicar_funcao_em_canais(correlacao, imagem, filtro_moda)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/moda_cor.png', moda)
    media = aplicar_funcao_em_canais(correlacao, imagem, filtro_media)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/media_cor.png', media)
    mediana = aplicar_funcao_em_canais(correlacao, imagem, filtro_mediana)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/mediana_cor.png', mediana)
    sobel = aplicar_funcao_em_canais(filtro_sobel, imagem)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/sobel_cor.png', sobel)
    prewitt = aplicar_funcao_em_canais(filtro_prewitt, imagem)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/prewitt_cor.png', prewitt)
    roberts = aplicar_funcao_em_canais(filtro_roberts, imagem)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/roberts_cor.png', roberts)
    ordem_k_0 = aplicar_funcao_em_canais(ordem_k(0), imagem)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/ordem_k_0_cor.png', ordem_k_0)
    ordem_k_8 = aplicar_funcao_em_canais(ordem_k(8), imagem)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/ordem_k_8_cor.png', ordem_k_8)
    quantizado_32 = aplicar_funcao_em_canais(quantizar, imagem, 32)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/quantizado_32_cor.png', quantizado_32)
    quantizado_8 = aplicar_funcao_em_canais(quantizar, imagem, 8)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/quantizado_8_cor.png', quantizado_8)
    equalizado = aplicar_funcao_em_canais(equalizar, imagem)
    imageio.imwrite('/home/simoes/git-simoes/digital_image_processing/exemplares/equalizado_cor.png', equalizado)

def rodar_em_cinza(imagem):
    binarizado_cinza = binarizar(converter_para_cinza(lenna), 80)
    imageio.imwrite('exemplares/binarizado_cinza.png', binarizado_cinza)

    identidade_cinza = mascara_identidade(converter_para_cinza(convolucao, lenna))
    imageio.imwrite('exemplares/identidade_cinza.png', identidade_cinza)
    borda_cinza = mascara_borda(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/borda_cinza.png', borda_cinza)
    afiado_cinza = mascara_afiado(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/afiado_cinza.png', afiado_cinza)
    borrao_cinza = mascara_borrao(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/borrao_cinza.png', borrao_cinza)
    borrao_gaussiano_cinza = mascara_borrao_gaussiano(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/borrao_gaussiano_cinza.png', borrao_gaussiano_cinza)
    moda_cinza = filtro_moda(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/moda_cinza.png', moda_cinza)
    media_cinza = filtro_media(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/media_cinza.png', media_cinza)
    mediana_cinza = filtro_mediana(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/mediana_cinza.png', mediana_cinza)
    sobel_cinza = filtro_sobel(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/sobel_cinza.png', sobel_cinza)
    prewitt_cinza = filtro_prewitt(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/prewitt_cinza.png', prewitt_cinza)
    roberts_cinza = filtro_roberts(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/roberts_cinza.png', roberts_cinza)
    ordem_k_0_cinza = ordem_com_k(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/ordem_k_0_cinza.png', ordem_k_0_cinza)
    ordem_k_8_cinza = ordem_com_k(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/ordem_k_8_cinza.png', ordem_k_8_cinza)
    quantizado_32_cinza = quantizar(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/quantizado_32_cinza.png', quantizado_32_cinza)
    quantizado_8_cinza = quantizar(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/quantizado_8_cinza.png', quantizado_8_cinza)
    equalizado_cinza = equalizar(converter_para_cinza(lenna))
    imageio.imwrite('exemplares/equalizado_cinza.png', equalizado_cinza)