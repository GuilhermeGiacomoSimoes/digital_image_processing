from FILTROS import *

lenna = imageio.imread ('lenna.jpg')




def dividir_imagem_em_canais_de_cor_para_teste(imagem):
    canais = np.dsplit(imagem, imagem.shape[-1])
    for canal in canais:
        yield canal[..., 0]





#
# TESTA SE A IMAGEM TEM LINHA, COLUNA E NUMERO DE BITs
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

    nao_255_ou_0 = False 

    for canal in  dividir_imagem_em_canais_de_cor_para_teste(lenna):
        canal_binarizado = binarizar(canal, 127)
        print(canal_binarizado)
 
    assert (not nao_255_ou_0)







 
