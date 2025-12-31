import pickle, pandas as pd

def generate_i2g_and_female_csv(c2g_file, c2i_file, output_dir):
    """
    Generate i2g mapping (col_index: (isfemale, acc)) and save csv with column indexes where isfemale==True.
    - c2g_file: pickle file for expert's idname -> gender (idname: (isfemale, acc)), ideally the superset and includes for all experts
    - c2i_file: pickle file for index.pkl in opentf that has (idname: index), ideally subset including the ones after some filterings
    """
    with open(c2g_file, 'rb') as f: c2g = pickle.load(f)
    with open(c2i_file, 'rb') as f: c2i = pickle.load(f)['c2i']

    def _(idname):
        try: return f'{int(float(idname.split("_")[0]))}_{"_".join(idname.split("_")[1:])}' # to handle xxx.0 bug in imdb's ids, no change if correct int ids
        except: return idname

    i2g = {}; missing_ids = []
    for idname, col_idx in c2i.items():
        try: i2g[col_idx] = c2g[_(idname)]
        except KeyError: missing_ids.append(_(idname))
    if len(missing_ids) > 0: print(f'The following {len(missing_ids)} idnames in c2i are missing in c2g: {missing_ids}')
    with open(f'{output_dir}i2g.pkl', 'wb') as f: pickle.dump(i2g, f)

    female_columns = sorted(idx for idx, (isfemale, acc) in i2g.items() if isfemale is True)
    pd.DataFrame(female_columns, columns=['teamsvecs-females-col-idx']).to_csv(f'{output_dir}females.csv', index=False)

# generate_i2g_and_female_csv(c2g_file='../../output/dblp/toy.dblp.v12.json/c2g.pkl', c2i_file='../../output/dblp/toy.dblp.v12.json/indexes.pkl', output_dir='../../output/dblp/toy.dblp.v12.json/')
# generate_i2g_and_female_csv(c2g_file='../../output/dblp/dblp.v12.json/c2g.pkl', c2i_file='../../output/dblp/dblp.v12.json.mt10.ts2/indexes.pkl', output_dir='../../output/dblp/dblp.v12.json.mt10.ts2/')
# generate_i2g_and_female_csv(c2g_file='../../output/dblp/dblp.v12.json/c2g.pkl', c2i_file='../../output/dblp/dblp.v12.json/indexes.pkl', output_dir='../../output/dblp/dblp.v12.json/')
#
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/toy.title.basics.tsv/c2g.pkl', c2i_file='../../output/imdb/toy.title.basics.tsv/indexes.pkl', output_dir='../../output/imdb/toy.title.basics.tsv/')
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/title.basics.tsv/c2g.pkl', c2i_file='../../output/imdb/title.basics.tsv.mt10.ts2/indexes.pkl', output_dir='../../output/imdb/title.basics.tsv.mt10.ts2/')
# generate_i2g_and_female_csv(c2g_file='../../output/imdb/title.basics.tsv/c2g.pkl', c2i_file='../../output/imdb/title.basics.tsv/indexes.pkl', output_dir='../../output/imdb/title.basics.tsv/')
#
# generate_i2g_and_female_csv(c2g_file='../../output/uspt/toy.patent.tsv/c2g.pkl', c2i_file='../../output/uspt/toy.patent.tsv/indexes.pkl', output_dir='../../output/uspt/toy.patent.tsv/')
# generate_i2g_and_female_csv(c2g_file='../../output/uspt/patent.tsv/c2g.pkl', c2i_file='../../output/uspt/patent.tsv.mt10.ts2/indexes.pkl', output_dir='../../output/uspt/patent.tsv.mt10.ts2/')
# generate_i2g_and_female_csv(c2g_file='../../output/uspt/patent.tsv/c2g.pkl', c2i_file='../../output/uspt/patent.tsv/indexes.pkl', output_dir='../../output/uspt/patent.tsv/')
