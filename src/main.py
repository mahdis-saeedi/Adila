import logging, multiprocessing
log = logging.getLogger(__name__)
import hydra

from adila import *

def init_process(): logging.basicConfig(level=logging.INFO)

def __(fpred, adila, minorities, ratios, algorithm, k_max, alpha, evalcfg):
    preds, preds_, fpred_ = adila.rerank(fpred, minorities, ratios, algorithm, k_max, alpha)
    adila.eval_fair(preds, minorities, preds_, fpred_, ratios, evalcfg.topK, evalcfg.fair_metrics, evalcfg.per_instance)
    adila.eval_utility(preds, fpred, preds_, fpred_, evalcfg.topK, evalcfg.utility_metrics, evalcfg.per_instance)

def _(adila, fpred, minorities, ratios, algorithm, k_max, alpha, acceleration, evalcfg):
    if os.path.isfile(fpred): __(fpred, adila, minorities, ratios, algorithm, k_max, alpha, evalcfg)
    elif os.path.isdir(fpred):
        import glob; from functools import partial
        fpreds = glob.glob(f'{fpred}*.pred')
        n_processes = multiprocessing.cpu_count() - 1 if acceleration == 'cpu' else int(acceleration.split(':')[1])
        if n_processes < 2:
            for fpred in fpreds: __(fpred, adila, minorities, ratios, algorithm, k_max, alpha, evalcfg)
        else:
            with multiprocessing.Pool(initializer=init_process, processes=n_processes) as p:
                p.map(partial(__, adila=adila, minorities=minorities, ratios=ratios, algorithm=algorithm, k_max=k_max, alpha=alpha, evalcfg=evalcfg), fpreds)

@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg) -> None:
    for algorithm in cfg.fair.algorithm:
        for notion in cfg.fair.notion:
            for attribute in cfg.fair.attribute:
                for is_popular_alg in cfg.fair.is_popular_alg:
                    adila = Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fgender, cfg.data.output, notion, attribute, is_popular_alg)
                    stats, minorities, ratios = adila.prep(cfg.fair.is_popular_coef)
                    if notion == 'dp' and cfg.fair.dp_ratio: ratios = [1 - cfg.fair.ratio if attribute == 'popularity' else cfg.fair.ratio]
                    try: _(adila, cfg.data.fpred, minorities, ratios, algorithm, cfg.fair.k_max, cfg.fair.alpha, cfg.acceleration, cfg.eval)
                    except Exception as e: log.info(f'{opentf.textcolor["red"]}{e}{opentf.textcolor["reset"]}')

if __name__ == '__main__': run()