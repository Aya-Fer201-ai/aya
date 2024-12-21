import tkinter as tk
from tkinter import messagebox

def rail_fence_encrypt(plaintext, k):
    # Créer un tableau pour contenir les caractères pour chaque rail
    rails = [''] * k
    direction = 1  # 1 signifie descendre, -1 signifie remonter
    rail_index = 0

    for char in plaintext:
        rails[rail_index] += char  # Placer le caractère dans le rail courant
        # Changer de direction si on atteint le haut ou le bas du rail
        if rail_index == 0:
            direction = 1
        elif rail_index == k - 1:
            direction = -1
        rail_index += direction  # Passer au rail suivant

    # Joindre tous les rails pour obtenir le message chiffré final
    return ''.join(rails)

def rail_fence_decrypt(ciphertext, k):
    rails = [''] * k
    direction = 1
    rail_index = 0
    rail_lengths = [0] * k

    # Première passe : déterminer combien de caractères vont dans chaque rail
    for char in ciphertext:
        rail_lengths[rail_index] += 1
        if rail_index == 0:
            direction = 1
        elif rail_index == k - 1:
            direction = -1
        rail_index += direction

    index = 0
    for i in range(k):
        rails[i] = ciphertext[index:index + rail_lengths[i]]
        index += rail_lengths[i]

    result = []
    rail_index = 0
    direction = 1

    for _ in range(len(ciphertext)):
        result.append(rails[rail_index][0])  # Prendre un caractère du rail courant
        rails[rail_index] = rails[rail_index][1:]  # Retirer ce caractère du rail
        
        if rail_index == 0:
            direction = 1
        elif rail_index == k - 1:
            direction = -1
        rail_index += direction

    return ''.join(result)

def encrypt_text():
    plaintext = entry_text.get()
    try:
        k = int(entry_k.get())
        if k < 2:
            raise ValueError("k doit être >= 2")
        
        ciphertext = rail_fence_encrypt(plaintext.replace(" ", "").upper(), k)
        result_var.set(f"Texte chiffré : {ciphertext}")
        
    except ValueError as e:
        messagebox.showerror("Erreur", str(e))

def decrypt_text():
    ciphertext = entry_text.get()
    try:
        k = int(entry_k.get())
        if k < 2:
            raise ValueError("k doit être >= 2")
        
        decrypted_text = rail_fence_decrypt(ciphertext, k)
        result_var.set(f"Texte déchiffré : {decrypted_text}")
        
    except ValueError as e:
        messagebox.showerror("Erreur", str(e))

# Création de la fenêtre principale
root = tk.Tk()
root.title("Chiffre Rail Fence")

# Création des éléments de l'interface graphique
label_k = tk.Label(root, text="Entrez le nombre de niveaux (k >= 2) :")
label_k.pack()

entry_k = tk.Entry(root)
entry_k.pack()

label_text = tk.Label(root, text="Entrez le texte :")
label_text.pack()

entry_text = tk.Entry(root)
entry_text.pack()

button_encrypt = tk.Button(root, text="Chiffrer", command=encrypt_text)
button_encrypt.pack()

button_decrypt = tk.Button(root, text="Déchiffrer", command=decrypt_text)
button_decrypt.pack()

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var)
result_label.pack()

# Lancer la boucle principale de l'interface graphique
root.mainloop()
