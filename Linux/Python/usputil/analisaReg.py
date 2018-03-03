#autor: Roberto Kenji Hiramatsu
#Data: 2016-08-23
# le o arquivo contendo a tabela de analise no formato
# distancia, diferenca dos tres angulos dos pontos fiduciais,
# valor de dispersao de 1 e 2
# valor de posicao do rosto de 1 e 2
# valor indice dos individuos 1 e 2
# valor indicado que indice dos individuos sao os mesmos
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

arq = open('../../tabela2','r')

# le a primeira linha
print arq.readline()


matriz = []
for line in arq:
    #print "{:3} {}".format(conta,line)
    linha = []
    valores = line.replace(' ','').replace('\n','').split(';')
    #print valores
    for v in valores:
        try:
            vf = int(v)
            linha.append(vf)
        except ValueError:
            try:
                vf = float(v)
                linha.append(vf)
            except ValueError:
                if v == 'True':
                    linha.append(1.0)
                elif v == 'False':
                    linha.append(0.0)
    matriz.append(linha)

print 'Tem ',len(matriz),' linhas'

x = []
y = []

idxn = len(matriz[0])-1
print 'Ultimo indice da linha',idxn
#conta = 0
for linha  in matriz:
    inserir = True
    for idx in range(0,len(x)-1):
        if x[idx] == linha[0] and y[idx] == linha[idxn]:
            inserir = False
    if inserir:
        x.append(linha[0])
        y.append(linha[idxn])
    #print X[conta],y[conta]
    #conta += 1
print 'tem ',len(x),' x valores e ',len(y),' y valores'

X = np.zeros((len(x),1))
for idx in range(0,len(x)-1):
    X[idx,0] = x[idx]

print X
print y
# executa o classificador
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)
#
# # apresenta o grafico
plt.figure(1,figsize=(20,10))
plt.clf()
plt.scatter(X.ravel(),y,color='black',zorder=20)
X_test = np.linspace(0,4, len(matriz))
#
def model(x):
    return 1/(1+np.exp(-x))
#
loss = model(X_test*clf.coef_+clf.intercept_).ravel()
plt.plot(X_test,loss,color='blue',linewidth=3)
#
osl = linear_model.LinearRegression()
osl.fit(X, y)
#
plt.plot(X_test,osl.coef_*X_test+osl.intercept_,linewidth=1)
plt.axhline(.5,color='.5')
plt.ylabel('y')
plt.xlabel('x')
plt.xticks(())
plt.yticks(())
plt.ylim(-0.25,1.25)
plt.xlim(-1,4)
#
plt.show()
#===============================================================================
