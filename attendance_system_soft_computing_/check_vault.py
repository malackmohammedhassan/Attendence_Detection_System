import pickle
import os

VAULT_PATH = r"data\biometric_vault.pkl"

if os.path.exists(VAULT_PATH):
    with open(VAULT_PATH, 'rb') as f:
        vault = pickle.load(f)
    
    print(f"\n--- {len(vault)} Students Found in Vault ---")
    for i, (name, data) in enumerate(vault.items(), 1):
        # data[0] is the embedding, data[1] is the Reg No
        print(f"{i}. Name: {name:15} | Reg No: {data[1]}")
    print("---------------------------------------\n")
else:
    print("Vault file not found. Make sure the path is correct.")