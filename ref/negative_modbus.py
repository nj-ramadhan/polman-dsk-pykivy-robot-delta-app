value = -16000

value = value + 2**16 if value & 2**15 else value

print(value)
