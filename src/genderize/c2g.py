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
# dblp: for all experts (authors)
# imdb: for missing cast'ncrew like directors. based on actor/actress, we know for some of experts
# uspt: none. all experts are labeled in original dataset

import csv, pickle

def imdb_extract_gender_dict(tsv_path, output_dir):
    result = {}
    with open(tsv_path, newline='', encoding='utf-8') as f:
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
            result[id_name] = value
            print(f'{row} --> {id_name} --> {value}')

    with open(f'{output_dir}c2g.pkl', 'wb') as f: pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

imdb_extract_gender_dict('../../output/imdb/title.basics.tsv/name.basics.tsv.gender.tsv', '../../output/imdb/title.basics.tsv/')