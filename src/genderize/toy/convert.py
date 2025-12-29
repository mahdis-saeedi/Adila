import json, pandas as pd

df = pd.read_csv('input.csv')  # columns: first, last
def gender_to_isfemale(g):
    if g == 'female': return True
    elif g == 'male': return False
    return None

def read_json_lines(path):
    parsed = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.strip()
            if not raw:
                parsed.append((None, None))
                continue
            clean = raw[:-1] if raw.endswith(',') else raw
            try: parsed.append((json.loads(clean), raw))
            except json.JSONDecodeError: parsed.append((None, raw))
    return parsed

genderize_logs = read_json_lines('genderizeLOG.txt')
genderapi_logs = read_json_lines('genderAPILOG.txt')
assert len(df) == len(genderize_logs) == len(genderapi_logs), 'Row count mismatch between input.csv and log files'

df['isfemale-genderize'] = [gender_to_isfemale(j.get('gender')) if j else None for j, _ in genderize_logs]
df['genderize-acc'] = [j.get('probability') * 100 if j and j.get('probability') is not None else None for j, _ in genderize_logs]
df['genderize-log'] = [raw for _, raw in genderize_logs]
df['isfemale-genderapi'] = [gender_to_isfemale(j.get('gender')) if j else None for j, _ in genderapi_logs]
df['genderapi-acc'] = [j.get('accuracy') if j and j.get('accuracy') is not None else None for j, _ in genderapi_logs]
df['genderapi-log'] = [raw for _, raw in genderapi_logs]

df.to_csv('output.csv', index=False)
