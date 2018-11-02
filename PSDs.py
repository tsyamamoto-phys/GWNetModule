

def DECIGO(f):

    '''
    Remya paper '15
    '''
    
    fp = 7.36
    s1 = 6.53*1e-49 * (1+ (f/fp)**2.0)
    s2 = 4.45*1e-51 * (f**(-4.0)) / (1+(f/fp)**2.0)
    s3 = 4.94*1e-52 * f**(-4.0)
    return s1+s2+s3
    
