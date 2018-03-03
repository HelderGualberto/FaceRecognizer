class A:
    def __init__(self):
        print "Iniciado A"
        
class B(A):
    #def __init__(self):
    #    print "Iniciado B"
     
    def opera(self):
        print "Opera de B"   
        
        
if __name__ == '__main__':
    a = B()