import os, pickle, logging, multiprocessing, random
log = logging.getLogger(__name__)

import hydra
# from time import time, perf_counter
# from functools import partial
# from random import randrange

#from util.fair_greedy import fairness_greedy
import pkgmgr as opentf
opentf.set_seed(0)
# from plot import area_under_curve

pd = opentf.install_import('pandas')
tqdm = opentf.install_import('tqdm', from_module='tqdm')
csr_matrix = opentf.install_import('scipy', 'scipy.sparse', from_module='csr_matrix')
torch = opentf.install_import('torch')

class Adila:

    def __init__(self, fteamsvecs, fsplits, fpreds, fgender, output, fair_notion='dp', attribute='popularity', is_popular_alg='avg'):
        self.output = f'{output}/adila/{attribute}'
        if not os.path.isdir(self.output): os.makedirs(self.output)
        if not os.path.isdir(f'{self.output}/{fair_notion}'): os.makedirs(f'{self.output}/{fair_notion}')
        with open(fteamsvecs, 'rb') as f: self.teamsvecs = pickle.load(f)
        with open(fsplits, 'rb') as f: self.splits = pickle.load(f)

        self.attribute = attribute
        self.fair_notion = fair_notion
        self.is_popular_alg = is_popular_alg
        self.fpreds = fpreds
        self.fgender = fgender


    def prep(self, coef=1.0) -> tuple: #coefficient to calculate a threshold for popularity (e.g. if 2.0, threshold = 2 * average number of teams per expert)
        try:
            log.info(f'Loading stats, ratios, and ids for minority experts ...')
            with open(f'{self.output}/stats.pkl', 'rb') as f: stats = pickle.load(f)
            minorities = pd.read_csv(f'{self.output}/labels.csv').iloc[:, 0].tolist()
            if self.fair_notion == 'eo':
                with open(f'{self.output}/eo/ratios.pkl', 'rb') as f: ratios = pickle.load(f)
        except (FileNotFoundError, EOFError):
            log.info(f'Loading failed! Generating files at {self.output} ...')
            stats = {}
            stats['*nexperts'] = self.teamsvecs['member'].shape[1]
            col_sums = self.teamsvecs['member'].sum(axis=0)

            stats['nteams_expert-idx'] = {k: v for k, v in enumerate(sorted(col_sums.A1.astype(int), reverse=True))}
            stats['*avg_nteams_expert'] = col_sums.mean()

            x, y = zip(*enumerate(sorted(col_sums.A1.astype(int), reverse=True)))
            # stats['*auc_nteams_expert'] = area_under_curve(x, y, 'expert-idx', 'nteams', show_plot=False)

            threshold = coef * stats[f'*{self.is_popular_alg}_nteams_expert']

            # many nonpopular/male but few popular/female. So, we only keep popular/female idxes as minorities.
            # this should be the same for all baselines, so read once from the file at ./output/{domain}/{dataset}
            if self.attribute == 'popularity': minorities = [expertidx for expertidx, nteam_expert in enumerate(col_sums.getA1()) if threshold <= nteam_expert] #rowid maps to columnid in teamvecs['member']
            elif self.attribute == 'gender': minorities = pd.read_csv(self.fgender).iloc[:, 0].tolist()
            stats['minority_ratio'] = len(minorities) / stats['*nexperts']
            with open(f'{self.output}/stats.pkl', 'wb') as f: pickle.dump(stats, f)
            pd.DataFrame(data=minorities, columns=['teamsvecs-experts-colidx']).to_csv(f'{self.output}/labels.csv', index_label='teamsvecs-experts-colidx')

            ratios = list()
            if self.fair_notion == 'eo': # we need to know per team's ratio of minorities
                skill_member = self.teamsvecs['skill'].transpose() @ self.teamsvecs['member']
                log.info(f'Generating ratios ... ')
                for i in tqdm(self.splits['test']):
                    team_skills = self.teamsvecs['skill'][i].nonzero()[1].tolist()
                    experts = [skill_member[idx].nonzero()[1] for idx in team_skills]
                    skill_holders = set(experts[0]).union(*experts)
                    assert skill_holders, f'{opentf.textcolor["red"]}No expert has team {i}\'s skills {team_skills}!{opentf.textcolor["reset"]}'
                    skill_holders_minorities = set(minorities).intersection(skill_holders)
                    ratios.append(len(skill_holders_minorities) / len(skill_holders))
                    with open(f'{self.output}/eo/ratios.pkl', 'wb') as file: pickle.dump(ratios, file)

        if self.fair_notion == 'dp': ratios = [stats['minority_ratio']]
        return stats, minorities, ratios

    def rerank(self, fpred, minorities, ratios, algorithm='det_greedy', k_max=100, alpha=0.05) -> tuple:
        """
        Args:
            fpred: predictions for test teams |test| * |experts|
            minorities: list of expert-idx who are minorities like females or populars
            ratios: desired ratio of protected experts in the output
            algorithm: ranker algorithm of choice among {'det_greedy', 'det_cons', 'det_relaxed', 'fa-ir'}
            k_max: maximum number of returned team members by reranker
            alpha: significance value for fa*ir algorithm
        Returns:
            tuple (list, list)
        """
        preds = torch.load(fpred)['y_pred']
        log.info(f'{opentf.textcolor["blue"]}Reranking {fpred} using {algorithm} with {k_max} cutoff ...{opentf.textcolor["reset"]}')
        # preds = torch.tensor([[0.1, 0.5, 0.3, 0.4,  0.1, 0.8, 0.3]])
        reranked_file = f'{self.output}/{self.fair_notion}/{os.path.split(fpred)[-1]}.{algorithm}.{self.is_popular_alg + "." if self.attribute=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "") + "." if algorithm=="fa-ir" else ""}{k_max}.rerank.pred'
        try:
            log.info(f'Loading reranked file {reranked_file} for {fpred} if exists ...')
            with open(reranked_file, 'rb') as f: csr_reranked = pickle.load(f)
        except FileNotFoundError:
            log.info(f'No existing rerank version. Reranking {fpred} ...')
            # start_time = perf_counter()
            r = min(max(ratios[0], 0.1), 0.9) #clamps to stay between [0.1,0.9]

            if algorithm == 'fa-ir':
                fsc = opentf.install_import('fairsearchcore')
                fair = fsc.Fair(min(k_max, preds.shape[1]), 1 - r if self.attribute == 'popularity' else r, alpha)
            elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                frr = opentf.install_import('reranking')

            rows, cols, value = list(), list(), list() #for the final reranked probs
            for i, team in enumerate(tqdm(preds)):
                ranked_member_probs = [(m, True if m in minorities else False, float(team[m])) for m in range(len(team))]
                ranked_member_probs.sort(key=lambda x: x[2], reverse=True) #[0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1]

                if self.fair_notion == 'eo': r = min(max(ratios[i], 0.1), 0.9)  # dynamic ratio r, clamps to stay between [0.1,0.9]

                if algorithm == 'fa-ir':
                    # FairScoreDocs needs True label for the members of the protected group.
                    # For gender, our minorities and protected group is the same, i.e., females.
                    # For popularilty, our minorities are populars but the protected group is non-populars. So, 'not' of their minority labels
                    ranked_experts = [fsc.models.FairScoreDoc(m[0], m[2], not m[1] if self.attribute=='popularity' else m[1]) for m in ranked_member_probs]
                    # Reset the Fair obj to dynamic ratio r
                    if self.fair_notion == 'eo': fair = fsc.Fair(min(k_max, preds.shape[1]), 1 - r if self.attribute == 'popularity' else r, alpha)  # fair.p = r; fair._cache = {} #reset the Fair obj but it's buggy

                    # fairsearchcore/fail_prob.py L#177 in __hash__(), cast to int. The value of self.remaining_candidates is of numpy type!
                    # see https://github.com/fair-search/fairsearch-fair-python/issues/4
                    if fair.is_fair(ranked_experts[:k_max]): reranked_idx = ranked_experts[:k_max] #no change
                    else: reranked_idx = fair.re_rank(ranked_experts)[:k_max]
                    reranked_idx = [x.id for x in reranked_idx]
                    # reranked_idx = [2, 0, 1, 5, 4, 3, 6]

                elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                    reranked_idx = frr.rerank([label for _, label, _ in ranked_member_probs], {True: r, False: 1 - r}, None, min(k_max, preds.shape[1]), algorithm, verbose=False) #verbose=True, a dataframe with more info
                    # reranked_idx = [2, 0, 1, 5, 4, 3, 6]

                # elif algorithm == 'fair_greedy':
                #     #TODO refactor and parameterize this algorithm
                #     bias_dict = dict([(member_probs.index(m), {'att': m[1], 'prob': m[2], 'idx': m[0]}) for m in member_probs[:500]])
                #     method = 'move_down'
                #     reranked_idx = fairness_greedy(bias_dict, r, 'att', method)[:k_max]
                #     reranked_probs = [bias_dict[idx]['prob'] for idx in reranked_idx[:k_max]]

                else: raise ValueError('Invalid fair reranking algorithm!')

                for j, reranked_member in enumerate(reranked_idx):
                    rows.append(i)
                    cols.append(reranked_member)
                    value.append(ranked_member_probs[j][2])
                    # we switch the top-rank probs for top-re-ranked experts
                    # this way both lists give correct top experts after final rankings for evaluation
                    # example:
                    # preds: [0.1, 0.5, 0.3, 0.4, 0.1, 0.8, 0.3]
                    # sorted preds: [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1] -> [5, 1, 3, 6, 2, 0, 4]
                    # rerank: [2, 0, 1, 5, 4, 3, 6] -> assign top probs [0.5, 0.4, 0.8, 0.3, 0.3, 0.1, 0.1]
                    # sorted rerank: [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1] -> [2, 0, 1, 5, 4, 3, 6]

            csr_reranked = csr_matrix((value, (rows, cols)), shape=preds.shape) #[0.5, 0.4, 0.8, 0.1, 0.3, 0.3, 0.1]
            with open(reranked_file, 'wb') as f: pickle.dump(csr_reranked, f)
        return preds, csr_reranked

    def eval_fair(self, preds, minorities, reranked_idx, ratios, k_max, alpha, metrics: set = {'skew', 'ndkl'}):
        """
        Args:
            preds: loaded predictions from a .pred file
            minorities: popularity labels
            reranked_idx: indices of re-ranked teams with a pre-defined cut-off
            ratios: desired ratio of popular/non-popular items in the output
            output: address of the output directory
        Returns:
            dict: ndkl metric before and after re-ranking
        """

        # because the mapping between popular/nonpopular and protected/nonprotected is reversed
        # TODO also check if we need more specific file names ( with fairness criteria for example)
        # use argument instead of this line
        # if algorithm == 'fa-ir':
        #     labels = [not value for value in labels]
        dic_before, dic_after = dict(), dict()
        frr = opentf.install_import('reranking')
        for metric in metrics:
            dic_before[metric], dic_after[metric] = list(), list()
            if metric in ['skew', 'exp', 'expu']: dic_before[metric], dic_after[metric] = {'protected': [], 'nonprotected': []}, {'protected': [], 'nonprotected': []}
            for i, team in enumerate(tqdm(preds)):
                # defining the threshold for the times we have or don't have cutoff
                threshold = len(preds) if k_max is None else k_max

                if fairness_notion == 'eo': r = {True: 1 - ratios[i], False: ratios[i]}
                else: r = ratios
                member_probs = [(m, labels[m], float(team[m])) for m in range(len(team))]
                member_probs.sort(key=lambda x: x[2], reverse=True)
                #IMPORTANT: the ratios keys should match the labels!
                if 'ndkl' == metric:
                    dic_before[metric].append(frr.ndkl([label for _, label, _ in member_probs[:threshold]], r))
                    dic_after[metric].append(frr.ndkl([labels[int(m)] for m in reranked_idx[i]], r))

                def calculate_prob(atr: bool, team: list) -> float: return team.count(atr) / len(team)

                if 'skew' == metric:
                    l_before = [label for _, label, _ in member_probs[: threshold]]
                    l_after = [labels[int(m)] for m in reranked_idx[i]]
                    dic_before['skew']['protected'].append(frr.skew(calculate_prob(False, l_before), r[False]))
                    dic_before['skew']['nonprotected'].append(frr.skew(calculate_prob(True, l_before), r[True]))
                    dic_after['skew']['protected'].append(frr.skew(calculate_prob(False, l_after), r[False]))
                    dic_after['skew']['nonprotected'].append(frr.skew(calculate_prob(True, l_after), r[True]))

                if metric in ['exp', 'expu']:
                    #TODO Needs Refactor
                    if metric == 'exp':
                        exp_before, per_group_exp_before = frt.Metrics.EXP(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), 'MinMaxRatio')
                    elif metric == 'expu':
                        exp_before, per_group_exp_before = frt.Metrics.EXPU(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), pd.DataFrame(data=[j[2] for j in member_probs[:k_max]]),'MinMaxRatio')
                    else: raise ValueError('Chosen Metric Is not Valid')

                    try: dic_before[metric]['protected'].append(per_group_exp_before[False])
                    except KeyError:dic_before[metric]['protected'].append(0)
                    try: dic_before[metric]['nonprotected'].append(per_group_exp_before[True])
                    except KeyError: dic_before[metric]['nonprotected'].append(0)
                    dic_before[metric][metric] = exp_before

                    if metric == 'exp':
                        exp_after, per_group_exp_after = frt.Metrics.EXP(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), 'MinMaxRatio')
                        # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                        # dic_after[metric][metric] = exp_after
                    elif metric == 'expu':
                        exp_after, per_group_exp_after = frt.Metrics.EXPU(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), pd.DataFrame(data=[j[2] for i in reranked_idx[i][:k_max] for j in member_probs if j[0] == i]), 'MinMaxRatio')
                        # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                        # dic_after[metric][metric] = exp_after
                    else:raise ValueError('Chosen Metric Is not Valid')
                    try: dic_after[metric]['protected'].append(per_group_exp_after[False])
                    except KeyError: dic_after[metric]['protected'].append(0)
                    try:  dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                    except KeyError:  dic_after[metric]['nonprotected'].append(0)
                    dic_after[metric][metric] = exp_after

            df_before = pd.DataFrame(dic_before[metric]).mean(axis=0).to_frame('mean.before')
            df_after = pd.DataFrame(dic_after[metric]).mean(axis=0).to_frame('mean.after')
            df = pd.concat([df_before, df_after], axis=1)
            df.to_csv(f'{output}.{algorithm}.{popularity_thresholding+"."  if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.{metric}.faireval.csv', index_label='metric')

    @staticmethod
    def eval_utility(teamsvecs_members, reranked_preds, fpred, preds, splits, metrics, output, algorithm, k_max, alpha,  att: str = 'popularity', popularity_thresholding: str ='avg' ) -> None:
        """
        Args:
            teamsvecs_members: teamsvecs pickle file
            reranked_preds: re-ranked teams
            fpred: .pred filename (to see if .pred.eval.mean.csv exists)
            preds: loaded predictions from a .pred file
            splits: indices of test and train samples
            metrics: desired utility metrics
            output: address of the output directory

        Returns:
            None
        """
        y_test = teamsvecs_members[splits['test']]
        try: df_mean_before = pd.read_csv(f'{fpred}.eval.mean.csv', names=['mean'], header=0)#we should already have it at f*.test.pred.eval.mean.csv
        except FileNotFoundError:
            _, df_mean_before, _, _ = calculate_metrics(y_test, preds, False, metrics)
            df_mean_before.to_csv(f'{fpred}.eval.mean.csv', columns=['mean'])
        df_mean_before.rename(columns={'mean': 'mean.before'}, inplace=True)
        _, df_mean_after, _, _ = calculate_metrics(y_test, reranked_preds.toarray(), False, metrics)
        df_mean_after.rename(columns={'mean': 'mean.after'}, inplace=True)
        pd.concat([df_mean_before, df_mean_after], axis=1).to_csv(f'{output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.utileval.csv', index_label='metric')


@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg) -> None:

    adila = Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fpreds, cfg.data.fgender, cfg.data.output, cfg.fair.notion, cfg.fair.attribute, cfg.fair.is_popular_alg)
    stats, minorities, ratios = adila.prep(cfg.fair.is_popular_coef)

    # creating a static ratio in case fairness_notion is 'dp' and hard ratio is set
    if cfg.fair.notion == 'dp' and cfg.fair.dp_ratio: ratios = [1 - cfg.fair.ratio if cfg.fair.attribute == 'popularity' else cfg.fair.ratio]


    if os.path.isfile(cfg.data.fpreds): reranked_preds = adila.rerank(cfg.data.fpreds, minorities, ratios, cfg.fair.algorithm, cfg.fair.k_max, cfg.fair.alpha)


    # for algorithm in ['det_greedy', 'det_relaxed', 'det_const_sort', 'fa-ir', 'det_cons']:
    #     for notion in ['eo', 'dp']:
    #         for attribute in ['popularity', 'gender']:
    #             for is_popular_alg in ['avg', 'auc']:
    #                 adila = Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fpreds, cfg.data.fgender,
    #                               cfg.data.output, notion, attribute, is_popular_alg)
    #                 stats, minorities, ratios = adila.prep(cfg.fair.is_popular_coef)
    #                 if os.path.isfile(cfg.data.fpreds):
    #                     try:
    #                         reranked_preds = adila.rerank(cfg.data.fpreds, minorities, ratios, algorithm,
    #                                                   cfg.fair.k_max, cfg.fair.alpha)
    #                     except Exception as e:
    #                         print(e)

    # if os.path.isdir(cfg.data.fpreds):
    #     # given a root folder, we can crawl the folder to find *.pred files and run the pipeline for all
    #     files = list()
    #     for dirpath, dirnames, filenames in os.walk(cfg.data.fpreds): files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith("pred") and 'rerank' not in file]
    #
    #     files = pd.DataFrame(files, columns=['.', '..', 'domain', 'baseline', 'setting', 'rfile'])
    #     address_list = list()
    #
    #     pairs = []
    #     for i, row in files.iterrows():
    #         output = f"{row['.']}/{row['..']}/{row['domain']}/{row['baseline']}/{row['setting']}/"
    #         pairs.append((f'{output}{row["rfile"]}', f'{output}rerank/'))
    #
    #     if params.settings['parallel']:
    #         print(f'Parallel run started ...')
    #         with multiprocessing.Pool(multiprocessing.cpu_count() if params.settings['core'] < 0 else params.settings['core']) as executor:
    #             executor.starmap(partial(Reranking.run,
    #                                      fsplits=args.fsplits,
    #                                      fairness_notion=args.fairness_notion,
    #                                      att=args.att,
    #                                      fgender=args.fgender,
    #                                      algorithm=args.algorithm,
    #                                      k_max=params.settings['fair']['k_max'],
    #                                      alpha=params.settings['fair']['alpha'],
    #                                      np_ratio=params.settings['fair']['np_ratio'],
    #                                      popularity_thresholding=params.settings['fair']['popularity_thresholding'],
    #                                      fairness_metrics=params.settings['fair']['metrics'],
    #                                      fteamsvecs=args.fteamsvecs,
    #                                      utility_metrics=params.settings['utility_metrics']), pairs)
    #

    #
    # try:
    #     print('Loading fairness evaluation results before and after reranking ...')
    #     for metric in fairness_metrics:
    #         fairness_eval = pd.read_csv(f'{new_output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.{metric}.faireval.csv')
    # except FileNotFoundError:
    #     print(f'Loading fairness results failed! Evaluating fairness metric {fairness_metrics} ...')
    #     Reranking.eval_fairness(preds, labels, reranked_idx, ratios, new_output, algorithm, k_max, alpha, fairness_notion, fairness_metrics, att, popularity_thresholding)
    #
    # try:
    #     print('Loading utility metric evaluation results before and after reranking ...')
    #     utility_before = pd.read_csv(f'{new_output}.{algorithm}.{popularity_thresholding+"." if att=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "")+"." if algorithm=="fa-ir" else ""}{k_max}.utileval.csv')
    # except:
    #     print(f' Loading utility metric results failed! Evaluating utility metric {utility_metrics} ...')
    #     Reranking.eval_utility(teamsvecs['member'], reranked_preds, fpred, preds, splits, utility_metrics, new_output, algorithm, k_max, alpha, att, popularity_thresholding)
    #
    # print(f'Pipeline for the baseline {fpred} completed by {multiprocessing.current_process()}! {time() - st}')
    # print('#'*100)


if __name__ == '__main__': run()

