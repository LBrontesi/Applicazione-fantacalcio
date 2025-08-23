tokens=["N'dicka"]
if "'" in tokens[0]:

    tokens = tokens[0].split("'", 1)
    print(f"{tokens[0].capitalize()}'{tokens[1].capitalize()}")