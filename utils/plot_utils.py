import matplotlib.pyplot as plt
import numpy as np
import os

def draw(top_ks, accs, baselines, num_samples, model_name, subset, result_save_path, verbose=False, checkpoints = [0.2,0.5,0.8]):
    """
    verbose: whether to draw the details on the graph
    checkpoints: the accs that need to be checked
    """
    plt.figure(figsize=(num_samples//2, 4))
    plt.title(f'acc vs. top_k, subset={subset}')
    plt.xlabel('top_k')
    plt.ylabel('acc')
    plt.xlim(0, len(top_ks))
    plt.ylim(0, 1)
    plt.xticks(top_ks)
    plt.yticks([i / 10 for i in range(11)])
    plt.plot(top_ks, baselines)
    plt.plot(top_ks, accs)
    plt.legend(['baseline', model_name])

    verbose_result = []
    if verbose:
        for y0 in checkpoints:
            index = np.argmin(np.abs(np.array(accs) - y0))
            verbose_result.append((index, top_ks[index], accs[index]))
            # print(index, top_ks[index], accs[index])
            plt.scatter(top_ks[index], accs[index], s=50, color='b')
            plt.text(top_ks[index]-10, accs[index]+0.05, rf'$acc \approx {y0}$', fontdict={'size':15, 'color':'b'})
            plt.plot([top_ks[index], top_ks[index]], [0, accs[index]], 'r--')
        
        # calculate the score
        # use y=1/np.sqrt(x) as weight
        weights = np.array([1/np.sqrt(x) for x in top_ks])
        baseline_score = np.sum(weights * baselines)
        full_score = np.sum(weights * np.array([1 for _ in top_ks]))    # if all top_k's acc are 1
        eval_score = np.sum(weights * accs)
        # project the score's range from [baseline_score,full_score] to [0,1]
        baseline_score_rel = baseline_score / full_score    # ~0
        range_orig = 1 - baseline_score_rel # scaling_factor = 1/range_orig
        eval_score_rel = eval_score / full_score    # ~[0,1]
        score = (1 - (1 - eval_score_rel) / range_orig)

        # save a report
        eval_report_path = os.path.join(result_save_path, 'eval_report.md')
        with open(eval_report_path, 'w') as f:
            f.write('# Evaluation report\n')
            # save the acc of the sample checkpoints 
            for i in range(len(checkpoints)):
                f.write(f'+ top_k={verbose_result[i][1]} ~ acc={checkpoints[i]}, true_acc={verbose_result[i][2]:.3f}\n')
            f.write(f'Score: {score:.3f}')
            
            
    save_name = 'eval_results.jpg' if not verbose else 'eval_results_verbose.jpg'
    plt.savefig(os.path.join(result_save_path, save_name))