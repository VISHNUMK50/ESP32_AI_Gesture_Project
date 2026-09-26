# Run after training:
# python 06_convert_tflite_to_header.py
#
# This creates gesture_model.h for the Arduino sketch.

data = open("gesture_model.tflite", "rb").read()

with open("gesture_model.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("const unsigned char gesture_model_tflite[] = {\n")
    for i, b in enumerate(data):
        if i % 12 == 0:
            f.write("  ")
        f.write(f"0x{b:02x}")
        if i != len(data)-1:
            f.write(", ")
        if i % 12 == 11:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int gesture_model_tflite_len = {len(data)};\n")

print("Created gesture_model.h")
