import csv

def main():
    with open('submission.csv', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < 20:
                print(f"[{row['rank']}] {row['candidate_id']} | {row['reasoning']}")

if __name__ == "__main__":
    main()
