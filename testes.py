from FILTROS import *

lenna = imageio.imread ('lenna.jpg')


#
# TESTA SE A IMAGEM TEM LINHA E COLUNA
#
def test_imagem():
    assert len(lenna.shape) == 3



#
# A QUANTIDADE DE PIXELS DA COLUNA DA IMAGEM ORIGINAL TEM QUE SER IGUAL
# A QUANTIDADE DE PIXELS DAS LINAHS DA NOVA IMAGEM
#
def test_girar_imagem_largura_altura():
    largura_original, altura_original, bits = lenna.shape    
    nova_imagem = aplicar_funcao_em_canais(girar_imagem, lenna) 
    largura_nova, altura_nova, bits = nova_imagem.shape
   
    assert largura_original == altura_nova

#
# A QUANTIDADE DE PIXELS DA COLUNA DA IMAGEM ORIGINAL TEM QUE SER IGUAL
# A QUANTIDADE DE PIXELS DAS LINAHS DA NOVA IMAGEM
#
def test_girar_imagem_altura_largura():
    largura_original, altura_original, bits = lenna.shape    
    nova_imagem =  aplicar_funcao_em_canais(girar_imagem, lenna) 
    largura_nova, altura_nova, bits = nova_imagem.shape
   
    assert altura_original == largura_nova



#
# A IMAGEM BINARIZADA SO DEVE CONTER PIXELS NO VALOR DE 255 OU 0
#
def test_binarizacao():
    imagem_binarizada = aplicar_funcao_em_canais(binarizar, lenna, 127)
    largura, altura, bits= imagem_binarizada.shape
   
    nao_255_ou_0 = False
 
    for i in range(largura):
        for j in range(altura):
            if imagem_binarizada[i, j] is not  255 and imagem_binarizada[i, j] is not 0:
                nao_255_ou_0 = True
                break

    assert not nao_255_ou_0
            














 
