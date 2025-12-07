import os, pickle, logging, multiprocessing, random
log = logging.getLogger(__name__)

import hydra
# from time import time, perf_counter
# from functools import partial
# from random import randrange

#from util.fair_greedy import fairness_greedy
import pkgmgr as opentf, plot

pd = opentf.install_import('pandas')
tqdm = opentf.install_import('tqdm', from_module='tqdm')
csr_matrix = opentf.install_import('scipy', 'scipy.sparse', from_module='csr_matrix')
torch = opentf.install_import('torch')

class Adila:

    def __init__(self, fteamsvecs, fsplits, fgender, output, fair_notion='dp', attribute='popularity', is_popular_alg='avg'):
        self.output = f'{output}/adila/{attribute}'
        if not os.path.isdir(self.output): os.makedirs(self.output)
        if not os.path.isdir(f'{self.output}/{fair_notion}'): os.makedirs(f'{self.output}/{fair_notion}')
        with open(fteamsvecs, 'rb') as f: self.teamsvecs = pickle.load(f)
        with open(fsplits, 'rb') as f: self.splits = pickle.load(f)

        self.attribute = attribute
        self.fair_notion = fair_notion
        self.is_popular_alg = is_popular_alg
        self.fgender = fgender
        self.minorities = []

    def _get_labeled_sorted_preds(self, preds, minorities):
        sorted_probs, sorted_indices = preds.sort(dim=1, descending=True)  # |Test| * |Experts|
        sorted_labels = (sorted_indices[..., None] == torch.tensor(minorities)).any(dim=-1)
        ## if |experts| are small/mid scale >> dense vector of boolean labels
        # labels = torch.zeros(preds.shape[1], dtype=torch.bool, device=preds.device)
        # labels[minorities] = True
        # sorted_labels = labels[sorted_indices]  # torch uses advanced indexing, not broadcasting! still |Test| * |Experts|
        return torch.stack([sorted_indices, sorted_labels.to(sorted_indices.dtype), sorted_probs], dim=-1)
        # [[expertid, minority label, ranked prob], ...]

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
            stats['*auc_nteams_expert'] = plot.area_under_curve(x, y, 'expert-idx', 'nteams', show_plot=False)

            threshold = coef * stats[f'*{self.is_popular_alg}_nteams_expert']

            # many nonpopular/male but few popular/female. So, we only keep popular/female idxes as minorities.
            # this should be the same for all baselines, so read once from the file at ./output/{domain}/{dataset}
            if self.attribute == 'popularity': minorities = [expertidx for expertidx, nteam_expert in enumerate(col_sums.getA1()) if threshold <= nteam_expert] #rowid maps to columnid in teamvecs['member']
            elif self.attribute == 'gender': minorities = pd.read_csv(self.fgender).iloc[:, 0].tolist()
            stats['minority_ratio'] = len(minorities) / stats['*nexperts']
            with open(f'{self.output}/stats.pkl', 'wb') as f: pickle.dump(stats, f)
            pd.DataFrame(data=minorities, columns=['teamsvecs-experts-colidx']).to_csv(f'{self.output}/labels.csv', index=False)

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
            fpred: the filename for predictions for test teams |test| * |experts|
            minorities: list of expert-idx who are minorities like females or populars
            ratios: desired ratio of protected experts in the output
            algorithm: ranker algorithm of choice among {'det_greedy', 'det_cons', 'det_relaxed', 'fa-ir'}
            k_max: maximum number of returned team members by reranker
            alpha: significance value for fa*ir algorithm
        Returns:
            preds: loaded predictions (probs) |test| * |experts|
            preds_: adjusted predictions (probs) after reranking |test| * |experts|
            fpred_: the filename for the saved reranked_preds
        """
        preds = torch.load(fpred, map_location='cpu')['y_pred']
        log.info(f'{opentf.textcolor["blue"]}Reranking {fpred} using {algorithm} with {k_max} cutoff ...{opentf.textcolor["reset"]}')
        # preds = torch.tensor([[0.1, 0.5, 0.3, 0.4,  0.1, 0.8, 0.3]])
        fpred_ = f'{self.output}/{self.fair_notion}/{os.path.split(fpred)[-1]}.{algorithm}.{self.is_popular_alg + "." if self.attribute=="popularity" else ""}{f"{alpha:.2f}".replace("0.", "") + "." if algorithm=="fa-ir" else ""}{k_max}.rerank.pred'
        try:
            log.info(f'Loading reranked file {fpred_} for {fpred} if exists ...')
            with open(fpred_, 'rb') as f: preds_ = pickle.load(f)
        except FileNotFoundError:
            log.info(f'No existing rerank version. Reranking {fpred} ...')
            # start_time = perf_counter()
            r = min(max(ratios[0], 0.1), 0.9) #clamps to stay between [0.1,0.9]

            if algorithm == 'fa-ir':
                fsc = opentf.install_import('fairsearchcore')
                fair = fsc.Fair(min(k_max, preds.shape[1]), 1 - r if self.attribute == 'popularity' else r, alpha)
            elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                frr = opentf.install_import('reranking')

            preds_ = preds.detach().clone() #for the final reranked probs
            teams_ = self._get_labeled_sorted_preds(preds, minorities)
            # [[expertid, minority label, ranked prob], ...]

            for i, team_ in enumerate(tqdm(teams_)):
                if self.fair_notion == 'eo': r = min(max(ratios[i], 0.1), 0.9)  # dynamic ratio r, clamps to stay between [0.1,0.9]

                if algorithm == 'fa-ir':
                    # FairScoreDocs needs True label for the members of the protected group.
                    # For gender, our minorities and protected group is the same, i.e., females.
                    # For popularilty, our minorities are populars but the protected group is non-populars. So, 'not' of their minority labels
                    experts = [fsc.models.FairScoreDoc(int(m[0]), float(m[2]), not bool(m[1]) if self.attribute=='popularity' else bool(m[1])) for m in team_]
                    # Reset the Fair obj to dynamic ratio r
                    if self.fair_notion == 'eo': fair = fsc.Fair(min(k_max, preds.shape[1]), 1 - r if self.attribute == 'popularity' else r, alpha)  # fair.p = r; fair._cache = {} #reset the Fair obj but it's buggy

                    # fairsearchcore/fail_prob.py L#177 in __hash__(), cast to int. The value of self.remaining_candidates is of numpy type!
                    # see https://github.com/fair-search/fairsearch-fair-python/issues/4
                    if fair.is_fair(experts[:k_max]): experts_ = experts[:k_max] #no change
                    else: experts_ = fair.re_rank(experts)[:k_max]
                    experts_ = [x.id for x in experts_]
                    # reranked_idx = [2, 0, 1, 5, 4, 3, 6]

                elif algorithm in ['det_greedy', 'det_relaxed', 'det_cons', 'det_const_sort']:
                    experts_ = frr.rerank([bool(label) for _, label, _ in team_], {True: r, False: 1 - r}, None, min(k_max, preds.shape[1]), algorithm, verbose=False) #verbose=True, a dataframe with more info
                    # reranked_idx = [2, 0, 1, 5, 4, 3, 6]

                # elif algorithm == 'fair_greedy':
                #     #TODO refactor and parameterize this algorithm
                #     bias_dict = dict([(member_probs.index(m), {'att': m[1], 'prob': m[2], 'idx': m[0]}) for m in member_probs[:500]])
                #     method = 'move_down'
                #     reranked_idx = fairness_greedy(bias_dict, r, 'att', method)[:k_max]
                #     reranked_probs = [bias_dict[idx]['prob'] for idx in reranked_idx[:k_max]]

                else: raise ValueError('Invalid fair reranking algorithm!')

                for j, expert_ in enumerate(experts_): preds_[i][expert_] = team_[j][2]
                # we switch the top-rank probs for top-re-ranked experts
                # this way both lists give correct top experts after final rankings for evaluation
                # example:
                # preds: [0.1, 0.5, 0.3, 0.4, 0.1, 0.8, 0.3]
                # sorted preds: [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1] -> [5, 1, 3, 6, 2, 0, 4]
                # rerank: [2, 0, 1, 5, 4, 3, 6] -> assign top probs [0.5, 0.4, 0.8, 0.3, 0.3, 0.1, 0.1]
                # sorted rerank: [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1] -> [2, 0, 1, 5, 4, 3, 6]

            with open(fpred_, 'wb') as f: pickle.dump(preds_, f)
        return preds, preds_, fpred_

    def eval_fair(self, preds, minorities, preds_, fpred_, ratios, topK, metrics=['skew', 'ndkl'], per_instance=False):
        """
        Args:
            preds: loaded predictions from a .pred file
            minorities: list of popular or female labels (true labels)
            preds_, fpred_: re-ranked probs considering a cut-off min(k_max, preds.shape[1]) and the stored filename
            ratios: inferred or a desired ratio of minorities
            topK: cutoff for fair reranking methods, ideally should be equal to k_max in reranking
            metrics: fairness evaluation metrics
            per_instance: evaluation metric value for each test team instance
        Returns:
            None but the results are stored in *.csv files
        """
        log.info(f'{opentf.textcolor["green"]}Fairness evaluation for {fpred_} using {metrics} with {topK} cutoff ...{opentf.textcolor["reset"]}')
        frr = opentf.install_import('reranking') # for ndkl and skew
        teams = self._get_labeled_sorted_preds(preds, minorities)  # [5, 1, 3, 6, 2, 0, 4] -> [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1]
        teams_ = self._get_labeled_sorted_preds(preds_, minorities)  # [2, 0, 1, 5, 4, 3, 6] -> [0.8, 0.5, 0.4, 0.3, 0.3, 0.1, 0.1]

        results = []
        topK = min(topK, preds.shape[1])
        for i, (team, team_) in enumerate(tqdm(zip(teams, teams_))):
            lsteam, lsteam_ = team[:, 1][:topK].bool().tolist(), team_[:, 1][:topK].bool().tolist()
            if self.fair_notion == 'eo': r = min(max(ratios[i], 0.1), 0.9)  # dynamic ratio r, clamps to stay between [0.1,0.9]
            else: r = ratios[0]

            result = {}
            for metric in metrics:
                if 'ndkl' == metric:
                    result[f'before.{metric}'] = frr.ndkl(lsteam, {True: r, False: 1 - r})
                    result[f'after.{metric}'] = frr.ndkl(lsteam_, {True: r, False: 1 - r})

                if 'skew' == metric:
                    result[f'before.{metric}.minority'] = frr.skew(lsteam.count(True)/topK, r)
                    result[f'before.{metric}.majority'] = frr.skew(lsteam.count(False)/topK, 1 - r)
                    result[f'after.{metric}.minority'] = frr.skew(lsteam_.count(True)/topK, r)
                    result[f'after.{metric}.majority'] = frr.skew(lsteam_.count(False)/topK, 1 - r)

                # if metric in ['exp', 'expu']:
                #     frt = opentf.install_import('FairRankTune') #python 3.9+
                #     if metric == 'exp': exp_before, per_group_exp_before = frt.Metrics.EXP(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), 'MinMaxRatio')
                #     elif metric == 'expu': exp_before, per_group_exp_before = frt.Metrics.EXPU(pd.DataFrame(data=[j[0] for j in member_probs[:k_max]]), dict([(j[0], j[1]) for j in member_probs[:k_max]]), pd.DataFrame(data=[j[2] for j in member_probs[:k_max]]),'MinMaxRatio')
                #
                #     try: before[metric]['protected'].append(per_group_exp_before[False])
                #     except KeyError: before[metric]['protected'].append(0)
                #     try: before[metric]['nonprotected'].append(per_group_exp_before[True])
                #     except KeyError: before[metric]['nonprotected'].append(0)
                #     before[metric][metric] = exp_before
                #
                #     if metric == 'exp': exp_after, per_group_exp_after = frt.Metrics.EXP(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), 'MinMaxRatio')
                #         # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                #         # dic_after[metric][metric] = exp_after
                #     elif metric == 'expu': exp_after, per_group_exp_after = frt.Metrics.EXPU(pd.DataFrame(data=reranked_idx[i][:k_max]), dict([(j, labels[j]) for j in reranked_idx[i][:k_max]]), pd.DataFrame(data=[j[2] for i in reranked_idx[i][:k_max] for j in member_probs if j[0] == i]), 'MinMaxRatio')
                #         # dic_after[metric]['protected'].append(per_group_exp_after[False]), dic_after[metric]['nonprotected'].append(per_group_exp_after[True])
                #         # dic_after[metric][metric] = exp_after
                #
                #     try: after[metric]['protected'].append(per_group_exp_after[False])
                #     except KeyError: after[metric]['protected'].append(0)
                #     try:  after[metric]['nonprotected'].append(per_group_exp_after[True])
                #     except KeyError:  after[metric]['nonprotected'].append(0)
                #     after[metric][metric] = exp_after
            results.append(result)
        df = pd.DataFrame(results)
        if per_instance: df.to_csv(f'{fpred_}.eval.fair.instance.csv', index=False)
        df.mean(axis=0).to_frame(name='mean').rename_axis('metrics').to_csv(f'{fpred_}.eval.fair.mean.csv')
        log.info(f'Saved at {fpred_}.eval.fair.mean{"/instance" if per_instance else ""}.csv.')

    def eval_utility(self, preds, fpred, preds_, fpred_, topK, metrics, per_instance=False) -> None:
        """
        Args:
            preds: the file for the predictions, *.pred file
            preds_: the file for the re-ranked probs considering a cut-off min(k_max, preds.shape[1]) and the stored filename
            topK: first stage retrieval for efficiency
            metrics: utility evaluation metrics
            per_instance: evaluation metric value for each test team instance
        Returns:
            None but the results are stored in *.csv files
        """

        def _evaluate(Y_, metrics, per_instance, preds, topK):
            import metric as evl
            # evl.metric works on numpy or scipy.sparse. so, we need to convert Y_ which is torch.tensor, either sparse or not
            Y_ = Y_.to_dense().numpy()
            df, df_mean = pd.DataFrame(), pd.DataFrame()
            if metrics.trec: df, df_mean = evl.calculate_metrics(Y, Y_, topK, per_instance, metrics.trec)
            return df, df_mean

        Y = self.teamsvecs['member'][self.splits['test']]
        for key in metrics:
            if key != 'topk': metrics[key] = [m.replace('topk', metrics.topk) for m in metrics[key]]
        log.info(f'{opentf.textcolor["magenta"]}Utility evaluation for {fpred_} using {metrics} ... {opentf.textcolor["reset"]}')
        try:
            log.info(f'Before: Loading {fpred}.eval.mean.csv ...')
            df_before_mean = pd.read_csv(f'{fpred}.eval.mean.csv', names=['mean'], header=0)#we should already have it at f*.test.pred.eval.mean.csv
            if per_instance: df_before = pd.read_csv(f'{fpred}.eval.instance.csv', header=0)
        except FileNotFoundError:
            log.info(f'Before: Loading {fpred}.eval.mean.csv failed! Evaluating from scratch ...')
            df_before, df_before_mean = _evaluate(preds, metrics, per_instance, preds, topK)
            if per_instance: df_before.to_csv(f'{fpred}.eval.instance.csv', float_format='%.5f', index=False)
            log.info(f'Before: Saving {fpred}.eval.mean.csv ...')
            df_before_mean.to_csv(f'{fpred}.eval.mean.csv')

        if per_instance: df_before.rename(columns={c: f'{c}.before' for c in df_before.columns}, inplace=True)
        df_before_mean.rename(columns={'mean': 'mean.before'}, inplace=True)

        log.info(f'After: Evaluating {fpred_} ...')
        df_after, df_after_mean = _evaluate(preds_, metrics, per_instance, preds, topK)
        if per_instance: df_after.rename(columns={c: f'{c}.after' for c in df_after.columns}, inplace=True)
        df_after_mean.rename(columns={'mean': 'mean.after'}, inplace=True)
        if per_instance: pd.concat([df_before.reset_index(drop=True), df_after.reset_index(drop=True)], axis=1).to_csv(f'{fpred_}.eval.utility.instance.csv', float_format='%.5f', index=False)
        pd.concat([df_before_mean, df_after_mean], axis=1).to_csv(f'{fpred_}.eval.utility.mean.csv', index_label='metric')
        log.info(f'After: Saved at {fpred_}.eval.utility.mean.csv.')

@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg) -> None:
    opentf.set_seed(cfg.seed)
    adila = Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fgender, cfg.data.output, cfg.fair.notion, cfg.fair.attribute, cfg.fair.is_popular_alg)
    stats, minorities, ratios = adila.prep(cfg.fair.is_popular_coef)
    # creating a static ratio in case fairness_notion is 'dp' and hard ratio is set
    if cfg.fair.notion == 'dp' and cfg.fair.dp_ratio: ratios = [1 - cfg.fair.ratio if cfg.fair.attribute == 'popularity' else cfg.fair.ratio]

    if os.path.isfile(cfg.data.fpred):
        preds, preds_, fpred_ = adila.rerank(cfg.data.fpred, minorities, ratios, cfg.fair.algorithm, cfg.fair.k_max, cfg.fair.alpha)
        adila.eval_fair(preds, minorities, preds_, fpred_, ratios, cfg.eval.topK, cfg.eval.fair_metrics, cfg.eval.per_instance)
        adila.eval_utility(preds, cfg.data.fpred, preds_, fpred_, cfg.eval.topK, cfg.eval.utility_metrics, cfg.eval.per_instance)

    # ## bruteforce
    # for algorithm in ['fa-ir', 'det_greedy', 'det_relaxed', 'det_const_sort', 'det_cons']:
    #     for notion in ['eo', 'dp']:
    #         for attribute in ['popularity', 'gender']:
    #             for is_popular_alg in ['avg', 'auc']:
    #                 adila = Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fgender, cfg.data.output, notion, attribute, is_popular_alg)
    #                 stats, minorities, ratios = adila.prep(cfg.fair.is_popular_coef)
    #                 if os.path.isfile(cfg.data.fpred):
    #                     try:
    #                         preds, preds_, fpred_ = adila.rerank(cfg.data.fpred, minorities, ratios, algorithm, cfg.fair.k_max, cfg.fair.alpha)
    #                         adila.eval_fair(preds, minorities, preds_, fpred_, ratios, cfg.eval.topK, cfg.eval.fair_metrics, cfg.eval.per_instance)
    #                         adila.eval_utility(preds, cfg.data.fpred, preds_, fpred_, cfg.eval.topK, cfg.eval.utility_metrics, cfg.eval.per_instance)
    #                     except Exception as e: log.info(f'{opentf.textcolor["red"]}{e}{opentf.textcolor["reset"]}')

    # if os.path.isdir(cfg.data.fpred):
    #     # given a root folder, we can crawl the folder to find *.pred files and run the pipeline for all
    #     files = list()
    #     for dirpath, dirnames, filenames in os.walk(cfg.data.fpred): files += [os.path.join(os.path.normpath(dirpath), file).split(os.sep) for file in filenames if file.endswith("pred") and 'rerank' not in file]
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
    #                                      ), pairs)
if __name__ == '__main__': run()

    #

