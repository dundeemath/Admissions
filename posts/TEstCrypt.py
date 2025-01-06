


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sympy as sp
import pandas as pd
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import dh
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes



def encrypt_message(key, plaintext):
    iv=5
    iv=iv.to_bytes(16, 'big')
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return iv,ciphertext

def decrypt_message(key, iv, ciphertext):
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext
    
# list of strings
p=23
g=5
a=2
text='asef'
b=5
# Step 1: Generate Diffie-Hellman Parameters and Keys
parameters = dh.generate_parameters(generator=2, key_size=512)

# Generate private/public key pairs for two parties
private_key_a = parameters.generate_private_key()
private_key_b = parameters.generate_private_key()

# Generate public keys
public_key_a = private_key_a.public_key()
public_key_b = private_key_b.public_key()

# Step 2: Derive Shared Secret
shared_key_a = private_key_a.exchange(public_key_b)
shared_key_b = private_key_b.exchange(public_key_a)

# Validate shared keys are identical
assert shared_key_a == shared_key_b, "Shared keys are not equal!"

shared_secret=int(np.mod(g**(a*b),p))
shared_secret_bytes=shared_secret.to_bytes(16,'big')
#shared_secret=g**(a*b)
derived_key = HKDF(
    algorithm=hashes.SHA256(),
    length=32,
    salt=None,
    info=b"diffie-hellman-key-exchange",
).derive(shared_secret_bytes)

# Step 4: Encrypt a Message Using AES
message = text.encode(encoding="utf-8")

# Decrypt the message


iv=5
iv=iv.to_bytes(16, 'big')
encrypted='\x03\x18\xfe\xa3'
encrypted = encrypted.encode('ISO-8859-1')
decrypted_message2 = decrypt_message(derived_key, iv, encrypted)
print((decrypted_message2))



    








