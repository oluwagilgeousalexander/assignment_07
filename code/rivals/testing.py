

text = "NEW!\n\nAvocado Toast\nV GF\n$8.50\nToast."
NOISE = {"NEW!", "NEW", "V", "GF"}
cleaned_list = []
cleaned = text.split("\n")
print(cleaned)
for line in cleaned:
    line = line.strip()
    if not line:
        continue
    if all(w in NOISE for w in line.split()):
        continue
    cleaned_list.append(line)
print(cleaned_list)
