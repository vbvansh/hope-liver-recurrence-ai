#!/usr/bin/env python3
"""Create a new experiment folder from the template and add its row to the tracker.

What: makes experiments/EXP-<NNN>_<slug>/ containing README.md (the experiment
card, copied from TEMPLATE.md with ID, title, owner, date and parent filled in)
and an empty results/ folder, then appends a "planned" row to
experiments/tracker.csv.

Why: every experiment gets the same card and exactly one tracker row, so results
stay comparable and can be shared with teammates as one folder.

Usage (from the repo folder):
  python3 experiments/new_experiment.py "Reproduce nnU-Net baseline" --owner YOUR_NAME
  python3 experiments/new_experiment.py "Phase dropout 0.5" --owner YOUR_NAME --parent EXP-002 --slug phase-dropout

Outputs: the new folder and one new tracker line. An existing folder is never
overwritten. Works with Python 3.6+ (standard library only).
"""
import argparse
import csv
import datetime
import os
import re
import sys

ID_PATTERN = re.compile(r"^EXP-(\d{3,})")


def slugify(text, max_len=40):
    """Lower-case, letters and digits only, words joined by '-'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "experiment"


def next_id(root, tracker):
    """Next free EXP number, looking at both the folders and the tracker rows."""
    numbers = [0]
    for name in os.listdir(root):
        match = ID_PATTERN.match(name)
        if match:
            numbers.append(int(match.group(1)))
    if os.path.exists(tracker):
        with open(tracker, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                match = ID_PATTERN.match(row.get("exp_id", ""))
                if match:
                    numbers.append(int(match.group(1)))
    return "EXP-{:03d}".format(max(numbers) + 1)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Create a new experiment folder and tracker row.")
    ap.add_argument("title", help="short title, e.g. 'Reproduce nnU-Net baseline'")
    ap.add_argument("--owner", required=True, help="who runs it (first name)")
    ap.add_argument("--slug", help="folder name part (default: from the title)")
    ap.add_argument("--parent", default="none", help="parent experiment ID, e.g. EXP-002")
    ap.add_argument("--root", default=here, help="experiments folder (default: this script's folder)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    tracker = os.path.join(root, "tracker.csv")
    template = os.path.join(root, "TEMPLATE.md")
    if not os.path.exists(template):
        sys.exit("missing template: " + template)
    if not os.path.exists(tracker):
        sys.exit("missing tracker: " + tracker)

    exp_id = next_id(root, tracker)
    folder_name = "{}_{}".format(exp_id, slugify(args.slug or args.title))
    folder = os.path.join(root, folder_name)
    if os.path.exists(folder):
        sys.exit("already exists, not touching it: " + folder)

    today = datetime.date.today().isoformat()
    with open(template, encoding="utf-8") as f:
        card = f.read()
    for key, value in {"{EXP_ID}": exp_id, "{TITLE}": args.title, "{OWNER}": args.owner,
                       "{DATE}": today, "{PARENT}": args.parent, "{FOLDER}": folder_name}.items():
        card = card.replace(key, value)

    os.makedirs(os.path.join(folder, "results"))
    open(os.path.join(folder, "results", ".gitkeep"), "w").close()
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    with open(tracker, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    row = {name: "" for name in header}
    row.update({"exp_id": exp_id, "title": args.title, "owner": args.owner, "status": "planned",
                "date_started": today, "parent": args.parent, "data_release": "v1.1",
                "split_evaluated": "validation", "positive_class": "malignant + haemangioma (as released)",
                "card": folder_name + "/README.md"})
    with open(tracker, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header).writerow(row)

    print("created {} and added it to tracker.csv".format(folder_name))


if __name__ == "__main__":
    main()
