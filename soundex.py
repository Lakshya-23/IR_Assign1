def get_soundex_code(c):
    c = c.upper()
    if c in "BFPV": return "1"
    if c in "CGJKQSXZ": return "2"
    if c in "DT": return "3"
    if c in "L": return "4"
    if c in "MN": return "5"
    if c in "R": return "6"
    return "0"

def generate_soundex(token):
    if not token: return ""
    token = token.upper()
    first_letter = token[0]
    codes = [get_soundex_code(c) for c in token[1:]]
    prev_code = get_soundex_code(first_letter)
    encoded = [first_letter]
    for code in codes:
        if code != "0" and code != prev_code:
            encoded.append(code)
            prev_code = code
    encoded_str = "".join(encoded)
    encoded_str += "000"
    return encoded_str[:4]