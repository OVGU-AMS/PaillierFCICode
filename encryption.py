"""

"""

from phe import paillier
from encoding import encode_from_float, decode_to_float


class EncryptedEncoding:
    def __init__(self, pk, x, need_to_encrypt=True, mults=0):
        self.pk = pk
        if need_to_encrypt:
            self.x = pk.raw_encrypt(encode_from_float(x))
        else:
            self.x = x
        self.mults = mults
        return
    
    def __add__(self, other):
        if isinstance(other, EncryptedEncoding):
            if self.pk != other.pk:
                raise ValueError("Cannot add EncryptedEncoding objects encrypted under different public keys!")
            if self.mults != other.mults:
                print("Warning: Different encodings being added!")
            out = (self.x * other.x) % self.pk.nsquare
            return EncryptedEncoding(self.pk, out, need_to_encrypt=False, mults=self.mults)
        
        elif isinstance(other, int) or isinstance(other, float):
            out = (self.x * pow(self.pk.g, encode_from_float(other, mults=self.mults), self.pk.nsquare)) % self.pk.nsquare
            return EncryptedEncoding(self.pk, out, need_to_encrypt=False, mults=self.mults)
        
        else:
            raise TypeError("Cannot add types 'EncryptedEncoding' and '%s'!" % type(other))
    
    def __sub__(self, other):
        return self.__add__(other*-1)
    
    def __rsub__(self, other):
        return (self.__mul__(-1)).__add__(other)
    
    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            out = pow(self.x, encode_from_float(other), self.pk.nsquare)
            return EncryptedEncoding(self.pk, out, need_to_encrypt=False, mults=self.mults+1)
        else:
            raise TypeError("Cannot multiply types 'EncryptedEncoding' and '%s'!" % type(other))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def decrypt(self, sk):
        return decode_to_float(sk.raw_decrypt(self.x), mults=self.mults)


class DummyEncryptedEncoding:
    def __init__(self, pk, x):
        self.pk = pk
        self.x = x
        return

    def __add__(self, other):
        if isinstance(other, DummyEncryptedEncoding):
            return DummyEncryptedEncoding(self.pk, self.x+other.x)
        else:
            return DummyEncryptedEncoding(self.pk, self.x+other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, DummyEncryptedEncoding):
            raise ValueError("Cannot multipy encrypted numbers!")
        else:
            return DummyEncryptedEncoding(self.pk, self.x*other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def decrypt(self, sk):
        return self.x


def test():
    a = 233.423
    b = -34.345
    c = 555.5
    d = 2343.4
    e = 333.34

    pk, sk = paillier.generate_paillier_keypair(n_length=2048)

    enca = EncryptedEncoding(pk, a)
    encb = EncryptedEncoding(pk, b)
    encc = EncryptedEncoding(pk, c)
    encd = EncryptedEncoding(pk, d)

    print(a + b + c + d)
    print((enca + encb + encc + encd).decrypt(sk))

#test()