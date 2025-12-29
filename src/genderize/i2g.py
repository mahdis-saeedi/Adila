import pickle, pandas as pd

def generate_i2g_and_female_csv(c2g_file, c2i_file, output_dir):
    """
    Generate i2g mapping (col_index: (isfemale, acc)) and save csv with column indexes where isfemale==True.
    - c2g_file: pickle file for expert's idname -> gender (idname: (isfemale, acc))
    - c2i_file: pickle file for index.pkl in opentf that has (idname: index)
    """
    with open(c2g_file, 'rb') as f: c2g = pickle.load(f)
    with open(c2i_file, 'rb') as f: c2i = pickle.load(f)['c2i']

    missing_ids = [idname for idname in c2i if idname not in c2g]
    assert not missing_ids, f'The following idnames in c2i are missing in c2g: {missing_ids}'

    i2g = {}
    for idname, col_idx in c2i.items(): i2g[col_idx] = c2g[idname]

    with open(f'{output_dir}i2g.pkl', 'wb') as f: pickle.dump(i2g, f)

    female_columns = sorted(idx for idx, (isfemale, acc) in i2g.items() if isfemale is True)
    pd.DataFrame(female_columns, columns=['teamsvecs-females-col-idx']).to_csv(f'{output_dir}females.csv', index=False)

# generate_i2g_and_female_csv(c2g_file='../../output/dblp/toy.dblp.v12.json/c2g.pkl', c2i_file='../../output/dblp/toy.dblp.v12.json/indexes.pkl', output_dir='../../output/dblp/toy.dblp.v12.json/')
# generate_i2g_and_female_csv(c2g_file='../../output/dblp/dblp.v12.json.mt10.ts2/c2g.pkl', c2i_file='../../output/dblp/dblp.v12.json.mt10.ts2/indexes.pkl', output_dir='../../output/dblp/dblp.v12.json.mt10.ts2/')
# generate_i2g_and_female_csv(c2g_file='../../output/dblp/dblp.v12.json/c2g.pkl', c2i_file='../../output/dblp/dblp.v12.json/indexes.pkl', output_dir='../../output/dblp/dblp.v12.json/')
#
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/toy.title.basics.tsv/c2g.pkl', c2i_file='../../output/imdb/toy.title.basics.tsv/indexes.pkl', output_dir='../../output/imdb/toy.title.basics.tsv/')
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/title.basics.tsv.mt10.ts2/c2g.pkl', c2i_file='../../output/imdb/title.basics.tsv.mt10.ts2/indexes.pkl', output_dir='../../output/imdb/title.basics.tsv.mt10.ts2/')
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/title.basics.tsv/c2g.pkl', c2i_file='../../output/imdb/title.basics.tsv/indexes.pkl', output_dir='../../output/imdb/title.basics.tsv/')
#
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/toy.patent.tsv/c2g.pkl', c2i_file='../../output/imdb/toy.patent.tsv/indexes.pkl', output_dir='../../output/imdb/toy.patent.tsv/')
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/patent.tsv.mt10.ts2/c2g.pkl', c2i_file='../../output/imdb/patent.tsv.mt10.ts2/indexes.pkl', output_dir='../../output/imdb/patent.tsv.mt10.ts2/')
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/patent.tsv/c2g.pkl', c2i_file='../../output/imdb/patent.tsv/indexes.pkl', output_dir='../../output/imdb/patent.tsv/')
