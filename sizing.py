import json


# 2000 patients roughly with ECG I lead

if __name__ == "__main__":
    with open("filtered_ABP_I.txt", "r") as f:
        sum = 0
        a = json.load(f)
        for key, value in a.items():
            for segment in value:
                sum += segment["size"]
    print("num_entries", len(a))
    print("data size", sum)
