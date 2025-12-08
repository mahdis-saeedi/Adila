import os, pickle, logging, multiprocessing, random
log = logging.getLogger(__name__)

import hydra

#from util.fair_greedy import fairness_greedy
import pkgmgr as opentf, main

@hydra.main(version_base=None, config_path='.', config_name='__config__')
def run(cfg) -> None:
    ## bruteforce
    for algorithm in ['fa-ir', 'det_greedy', 'det_relaxed', 'det_const_sort', 'det_cons']:
        for notion in ['eo', 'dp']:
            for attribute in ['popularity', 'gender']:
                for is_popular_alg in ['avg', 'auc']:
                    adila = main.Adila(cfg.data.fteamsvecs, cfg.data.fsplits, cfg.data.fgender, cfg.data.output, notion, attribute, is_popular_alg)
                    stats, minorities, ratios = adila.prep(cfg.fair.is_popular_coef)
                    cfg.fair.algorithm = algorithm
                    try: main._(adila, minorities, ratios, cfg)
                    except Exception as e: log.info(f'{opentf.textcolor["red"]}{e}{opentf.textcolor["reset"]}')

if __name__ == '__main__': run()
