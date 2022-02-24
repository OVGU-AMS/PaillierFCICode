
# n+2m - mod bits
# m - frac bits
ENCODING_MOD_BITS = 160
ENCODING_FRAC_BITS = 32

def encode_from_float(x, mults=0, mod_bits=ENCODING_MOD_BITS, frac_bits=ENCODING_FRAC_BITS):
    sign = 0

    # Handle number sign
    if x<0:
        sign = 1
        x = -x

    max_enc = 2**mod_bits
    frac_factor = 2**((mults+1)*frac_bits)

    res = int(frac_factor*x)

    # Return sign to integer
    if sign:
        res = max_enc - res
    
    return res


def decode_to_float(z, mults=0, mod_bits=ENCODING_MOD_BITS, frac_bits=ENCODING_FRAC_BITS):
    sign = 0

    max_enc = 2**mod_bits
    max_enc_half = 2**(mod_bits-1)

    frac_factor = 2**((mults+1)*frac_bits)
    
    z = z % max_enc

    # Handle number sign
    if z > max_enc_half:
        sign = 1
        z = max_enc - z
    
    # Note large division here
    res = z / frac_factor

    # Return sign to float
    if sign:
        res = -res
    
    return res


def test():
    a = 23.42
    b = -44.23
    c = -23.43
    d = 34.2
    e = 44.23
    ea = encode_from_float(a)
    eb = encode_from_float(b)
    ec = encode_from_float(c)
    ed = encode_from_float(d, mults=1)
    ee = encode_from_float(e, mults=2)

    print(a+b)
    print(decode_to_float(ea+eb, mults=0))

    print(a*b + d)
    print(decode_to_float(ea*eb + ed, mults=1))

    print(a*b*c + e)
    print(decode_to_float(ea*eb*ec + ee, mults=2))

#test()