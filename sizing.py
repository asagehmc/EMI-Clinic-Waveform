import json


if __name__ == "__main__":
    with open("filtered_ABP_II.txt", "r") as f:
        sum = 0
        a = json.load(f)
        for key, value in a.items():
            for segment in value:
                sum += segment["size"]
    print(sum)
