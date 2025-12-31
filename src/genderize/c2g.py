# load the labeled dataset and generate {id_name: (gender, accuracy)}
# this is needed only because in the old code (labelDataset.py -> main.py), we
#   1) extract names from datasets,
#   2) do api call to genderize
#   3) update the dataset with the gender labels

# In yet-to-be-refactor code, for future, we don't need to update/touch the dataset. We simply do
#   1) (same as before) extract id_names from datasets,
#   2) (same as before) do api call to genderize
#   3) generate {id_name: (gender, accuracy)}

# datasets:
# dblp: for all experts (authors): dblp.v12.json.gender.json -> c2g.pkl
# imdb: for missing cast'ncrew like directors. based on actor/actress, we know for some of experts: name.basics.tsv.gender.tsv -> c2g.pkl
# uspt: none. all experts are labeled in original dataset: inventor.tsv -> c2g.pkl

import csv, pickle, json

def imdb_extract_gender_dict(name_basics_tsv_path, output_dir):
    c2g = {}
    with open(name_basics_tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        next(reader, None) # skip the header
        for row in reader:
            id_int = int(row[0].replace('nm', ''))  # nm0000003
            fullname = row[1].lower().replace(' ', '_') # Brigitte Bardot
            gender = row[2].strip() #if row[2] else '' # 'M' or 'F' or ''
            gender_accuracy = int(float(row[3]) * 100) if row[3] else 0 # may be empty
            profession = row[6].lower()

            id_name = f'{id_int}_{fullname}'

            if 'actress' in profession: value = (True, 100)
            elif 'actor' in profession: value = (False,100)
            elif gender: value = (True if gender == 'F' else False, gender_accuracy)
            else: value = (None, 0)
            c2g[id_name] = value
            print(f'{row} --> {id_name} --> {value}')

    with open(f'{output_dir}c2g.pkl', 'wb') as f: pickle.dump(c2g, f)

def dblp_extract_gender_dict(json_path, output_dir):
    c2g = {}
    with open(json_path, 'r', encoding='utf-8') as jf:
        for line in jf:
            try:
                if not line: break
                # Skip lines that are just brackets (for JSON array format files)
                if line.strip() in ['[', ']']: continue
                jsonline = json.loads(line.lower().lstrip(","))
                try: authors = jsonline['authors']
                except: continue  # publication must have authors (members)
                for author in authors:
                    idname = f"{author['id']}_{author['name'].replace(' ', '_').lower()}"
                    value = True if author['gender']['value'] == 'f' else (False if author['gender']['value'] == 'm' else None)
                    acc = int(author['gender']['probability'] * 100) if value is not None else 0
                    if idname not in c2g or (not c2g[idname][0]): c2g[idname] = (value, acc)
            except json.JSONDecodeError as e:  # ideally should happen only for the last line ']'
                print(f'JSONDecodeError: There has been error in loading json line `{line}`!\n{e}')
                continue
            except Exception as e: raise e
    with open(f'{output_dir}c2g.pkl', 'wb') as f: pickle.dump(c2g, f)

def uspt_extract_gender_dict(inventor_tsv_path, output_dir):
    c2g = {}
    with open(inventor_tsv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        next(reader, None) #skip header
        for row in reader:
            inventor_id = row[0].strip()
            name_first = row[1].strip()
            name_last = row[2].strip()
            male_flag = row[3].strip()
            id_name = f'{inventor_id}_{name_first}_{name_last}'
            if male_flag: value = (not bool(float(male_flag)), 100) # we keep isfemale, so not male_flag
            else: value = (None, 0)
            c2g[id_name] = value
            print(f'{row} --> {id_name} --> {value}')

    with open(f'{output_dir}c2g.pkl', 'wb') as f: pickle.dump(c2g, f)

# c2g for the entire dataset is enough for other filtered dataset since it is a superset, including all experts
# imdb_extract_gender_dict('../../output/imdb/toy.title.basics.tsv/name.basics.tsv.gender.tsv', '../../output/imdb/toy.title.basics.tsv/')
# dblp_extract_gender_dict('../../output/dblp/toy.dblp.v12.json/dblp.v12.json.gender.json', '../../output/dblp/toy.dblp.v12.json/')
# uspt_extract_gender_dict('../../output/uspt/toy.patent.tsv/inventor.tsv', '../../output/uspt/toy.patent.tsv/')