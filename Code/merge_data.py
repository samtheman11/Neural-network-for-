import csv

def load_temperature(temp_file="la_temperature.csv"):
    temp_lookup = {}
    with open(temp_file) as f:
        lines = f.readlines()

    header_index = None
    for i, line in enumerate(lines):
        if line.startswith("time,"):
            header_index = i
            break
    if header_index is None:
        raise ValueError("Could not find a 'time,...' header row in the temperature file")

    header = lines[header_index].strip().split(",")
    temp_col_index = None
    for i, col in enumerate(header):
        if "temperature" in col.lower():
            temp_col_index = i
            break

    for line in lines[header_index + 1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        timestamp = parts[0]
        temperature_c = float(parts[temp_col_index])
        temp_lookup[timestamp] = temperature_c

    return temp_lookup


def merge_files(demand_file="real_data.csv", temp_file="la_temperature.csv", output_file="merged_data.csv"):
    temp_lookup = load_temperature(temp_file)

    matched = 0
    unmatched = 0

    with open(demand_file) as infile, open(output_file, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["hour", "day_of_week", "day_of_year", "temperature_c", "demand"])

        for row in reader:
            key = row["date_hour"]
            if key in temp_lookup:
                writer.writerow([
                    row["hour"],
                    row["day_of_week"],
                    row["day_of_year"],
                    temp_lookup[key],
                    row["demand"]
                ])
                matched += 1
            else:
                unmatched += 1

    print(f"Matched {matched} rows, {unmatched} rows had no temperature match")
    print(f"Saved merged data to {output_file}")

if __name__ == "__main__":
    merge_files()