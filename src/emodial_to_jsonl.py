"""AI Hub 271 (Korean multi-turn emotional dialogues) xlsx -> jsonl of dialogues, stdlib only.

Column A marks a dialogue start with "S", B is the utterance, C the emotion label; speakers alternate by row.
"""
import argparse, json, re, zipfile
import xml.etree.ElementTree as ET

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

z = zipfile.ZipFile(args.xlsx)
shared = ["".join(t.text or "" for t in si.iter(f"{M}t"))
          for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{M}si")]
sheet = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))[0]


def cell_value(c):
    v = c.find(f"{M}v")
    if c.get("t") == "s":
        return shared[int(v.text)]
    if c.get("t") == "inlineStr":
        return "".join(t.text or "" for t in c.iter(f"{M}t"))
    return v.text if v is not None else ""


dialogs, cur = [], None
for row in ET.fromstring(z.read(sheet)).find(f"{M}sheetData").findall(f"{M}row"):
    cells = {re.match(r"[A-Z]+", c.get("r")).group(): cell_value(c) for c in row.findall(f"{M}c")}
    text = (cells.get("B") or "").strip()
    if not text:
        continue
    if (cells.get("A") or "").strip().upper() == "S" or cur is None:
        cur = []
        dialogs.append(cur)
    cur.append({"text": text, "emotion": (cells.get("C") or "").strip()})
with open(args.out, "w") as fh:
    for i, d in enumerate(dialogs):
        fh.write(json.dumps({"id": f"emodial{i:05d}", "turns": d}, ensure_ascii=False) + "\n")
print(len(dialogs), "dialogs,", sum(map(len, dialogs)), "utterances")
