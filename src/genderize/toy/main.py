import pandas as pd, requests, json
from tqdm import tqdm
# This script extracts the data from the two APIs

RED = "\033[91m"; RESET = "\033[0m"

def api(data):
    header = ['first', 'last', 'isfemale-genderize', 'genderize-acc', 'genderize-log', 'isfemale-genderapi', 'genderapi-acc', 'genderapi-log']
    output = []
    for row in tqdm(data.itertuples(index=False), total=len(data)):
        try:# Genderize
            req_ize = requests.get(url := f'https://api.genderize.io?name={row.first}')
            req_ize.raise_for_status()
            j_ize = json.loads(req_ize.text)
            e_ize = None
        except Exception as e: e_ize = e
        finally: print(f'{url}{f" -> {RED}[ERROR:{e_ize}]{RESET}" if e_ize else ""}')

        try:# Gender-API
            req_api = requests.get( url := f'https://gender-api.com/get?split={row.first}%20{row.last}&key=ddpwJPnQqdP3otFMPz8ZppaA4SBsqRRnVlSK')
            req_api.raise_for_status()
            j_api = json.loads(req_api.text)
            if 'errno' in j_api: raise Exception(j_api['errmsg'])
            e_api = None
        except Exception as e: e_api = e
        finally: print(f'{url}{f" -> {RED}[ERROR:{e_api}]{RESET}" if e_ize else ""}')

        o = (row.first, row.last,)
        o += (True if j_ize['gender'] == 'female' else (False if j_ize['gender'] == 'male' else None), j_ize.probability * 100, req_ize.text,) if not e_ize else (None, None, f'ERROR:{e_ize}]')
        o += (True if j_api['gender'] == 'female' else (False if j_api['gender'] == 'male' else None), j_api.accuracy, req_api.text,) if not e_api else (None, None, f'ERROR:{e_ize}]')

        output.append(o)

    (df := pd.DataFrame(output, columns=header)).to_csv('output.csv', index=False)
    return df

def plot(output):
    # plot the difference in accuracy
    import matplotlib.pyplot as plt, numpy as np
    fig, ax = plt.subplots(figsize=(10, 3))

    # ize_acc = np.random.uniform(0, 100, 100)
    # api_acc = np.random.uniform(0, 100, 100)
    # labels = np.arange(100)
    ize_acc = output['genderize-acc'].to_numpy(na_value=0)
    api_acc = output['genderapi-acc'].to_numpy(na_value=0)
    labels = output['first'].tolist()

    sorted_indices = np.argsort(ize_acc - api_acc)[::-1]
    ize_acc_sorted = ize_acc[sorted_indices]
    api_acc_sorted = api_acc[sorted_indices]

    ax.bar(np.arange(len(ize_acc_sorted)), height=ize_acc_sorted, width=1, alpha=0.6, color='blue', label='genderize')
    ax.bar(np.arange(len(api_acc_sorted)), height=api_acc_sorted, width=1, alpha=0.6, color='red', label='genderapi')
    ax.plot(np.arange(len(ize_acc_sorted)), ize_acc_sorted, color='black', linewidth=1.2, linestyle='-', marker='')
    ax.plot(np.arange(len(api_acc_sorted)), api_acc_sorted, color='red', linewidth=1.2, linestyle='-', marker='')
    plt.xticks(ticks=np.arange(len(api_acc_sorted)), labels=labels, rotation=90); plt.xlabel('firstname')

    plt.title('genderize vs. genderapi'); plt.ylabel('accuracy (%)'); plt.ylim(0, 100)
    plt.legend(loc='center'); plt.tight_layout()
    plt.savefig('stats.pdf'); plt.savefig("stats.png")
    plt.show()

if __name__ == '__main__':
    # df = api(data=pd.read_csv('input.csv', nrows=2))
    # plot(output=df)
    plot(output=pd.read_csv('output.csv'))