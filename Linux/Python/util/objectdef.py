class Face:

    def __init__(self, rep, identity,arquivo = None,triangs = None, angh = 0,angv=0 ,bluridx=0.0):
        self.rep = rep
        self.identity = identity
        self.arquivo = arquivo
        self.triangs = triangs
        self.angh = angh
        self.angv = angv
        self.bluridx = bluridx

    def __repr__(self):
        return "{{id: {},arq: {},ah:{:4.2f},av:{:4.2f},rep[0:5]: {}, {},{:5.1f}}}".format(
            str(self.identity),self.arquivo,self.angh,self.angv,
            self.rep[0:5],self.triangs,self.bluridx
        )
    def compara(self,outro):
        dista = calcDistancia(self.rep, outro.rep)
        da1 =self.triangs.ang1-outro.triangs.ang1
        da2 =self.triangs.ang2-outro.triangs.ang2
        da3 =self.triangs.ang3-outro.triangs.ang3
        #return (dist,TriAng((a1-oa1),(a2-oa2),(a3-oa3)))
        print "{:6.3f} ; {:6.1f} ; {:6.1f} ; {:6.1f} ; {:6.1f} ; {:6.1f} ; {} ; {} ; {} ; {} ; {}".format(
              dista,da1,da2,da3,self.bluridx,outro.bluridx,
              (self.posf > 0 ),(outro.posf > 0),
              self.identity,outro.identity,(self.identity == outro.identity))

