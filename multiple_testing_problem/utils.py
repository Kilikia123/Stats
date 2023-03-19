import numpy as np
import scipy.stats as sps

from mht import AdjustmentMethodABC


def compute_fwer(reject, alt_mask):
    """
    Функция подсчета FWER.
    
    :param reject: булева матрица результатов проверки гипотез,
                   имеет размер (число_экспериментов, количество_гипотез).
                   В позиции (k, j) записано True, если в эксперименте k 
                   гипотеза H_j отвергается.
    :param alt_mask: булев массив верности гипотез размера (количество_гипотез,).
                     В позиции j записано True, если гипотеза H_j не верна.
        
    :return: оценка fwer по данным экспериментам.
    """

    vps = np.logical_and(reject, ~np.array(alt_mask)).sum(axis = 1)
    
    fwer = (vps > 0).sum() / len(vps)

    assert np.isscalar(fwer)
    assert 0 <= fwer <= 1, "Посчитанное значение не в промежутке [0, 1]"
    return fwer


def compute_fdr(reject, alt_mask):
    """
    Функция подсчета FDR.
    
    :param reject: булева матрица результатов проверки гипотез,
                   имеет размер (число_экспериментов, количество_гипотез).
                   В позиции (k, j) записано True, если в эксперименте k 
                   гипотеза H_j отвергается.
    :param alt_mask: булев массив верности гипотез размера (количество_гипотез,).
                     В позиции j записано True, если гипотеза H_j не верна.
        
    :return: оценка fdr по данным экспериментам.
    """
    vps = np.logical_and(reject, ~np.array(alt_mask)).sum(axis = 1)
    rs = reject.sum(axis = 1)
    
    fdr = np.mean(vps / np.maximum(rs, 1))


    assert np.isscalar(fdr)
    assert 0 <= fdr <= 1, "Посчитанное значение не в промежутке [0, 1]"
    return np.nan_to_num(fdr)


def compute_power(reject, alt_mask):
    """
    Функция подсчета мощности.
    
    :param reject: булева матрица результатов проверки гипотез,
                   имеет размер (число_экспериментов, количество_гипотез).
                   В позиции (k, j) записано True, если в эксперименте k 
                   гипотеза H_j отвергается.
    :param alt_mask: булев массив верности гипотез размера (количество_гипотез,).
                     В позиции j записано True, если гипотеза H_j не верна.
        
    :return: оценка мощности по данным экспериментам.
    """

    # Если данные полностью согласуются с основными гипотезами, то
    # мощность не определена, потому вместо неё возвращаем np.nan
    # (так полезно сделать, чтобы heatmap-ы нормально отрисовались)
    if ~np.any(alt_mask):
        return np.nan

    vps = np.logical_and(reject, ~np.array(alt_mask)).sum(axis = 1)
    rs = reject.sum(axis = 1)
    delta_m = np.sum(alt_mask)
    
    power = ((rs - vps) / np.maximum(delta_m, 1)).mean()

    assert np.isscalar(power)
    assert 0 <= power <= 1, "Посчитанное значение не в промежутке [0, 1]"
    return power


class BonferroniAdjustment(AdjustmentMethodABC):
    def __init__(self, alpha):
        super().__init__(
            name="Метод Бонферрони",
            alpha=alpha,
            controls_for="FWER"
        )

    def adjust(self, pvalues):
        """
        Функция МПГ-коррекции.
        
        :param pvalues: Матрица p-value, имеет размер 
                        (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано p-value критерия S_j для проверки 
                        гипотезы H_j в эксперименте k.
            
        :return reject: булева матрица результатов проверки гипотез,
                        имеет размер (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано True, если в эксперименте k 
                        гипотеза H_j отвергается.
        :return adjusted: Матрица скорректированных p-value, имеет размер 
                          (число_экспериментов, количество_гипотез).
        """

        assert pvalues.ndim == 2,\
            "По условию задачи, adjust принимает матрицу "\
            "размерностей (число_экспериментов, количество_гипотез)"

        reject = np.minimum(pvalues * pvalues.shape[1], 1) <= self.alpha
        adjusted = np.minimum(pvalues * pvalues.shape[1], 1)

        assert (reject.shape == pvalues.shape
                and adjusted.shape == pvalues.shape),\
            "Размерности матриц reject и adjusted "\
            "отличаются от размерностей входных данных"
        assert np.all((0 <= adjusted) & (adjusted <= 1)),\
            "Некоторые из скорректированных p-значений не лежат в [0, 1]"
        return reject, adjusted


class HolmAdjustment(AdjustmentMethodABC):
    def __init__(self, alpha):
        super().__init__(
            name="Метод Холма",
            alpha=alpha,
            controls_for="FWER"
        )

    def adjust(self, pvalues):
        """
        Функция МПГ-коррекции.
        
        :param pvalues: Матрица p-value, имеет размер 
                        (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано p-value критерия S_j для проверки 
                        гипотезы H_j в эксперименте k.
            
        :return reject: булева матрица результатов проверки гипотез,
                        имеет размер (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано True, если в эксперименте k 
                        гипотеза H_j отвергается.
        :return adjusted: Матрица скорректированных p-value, имеет размер 
                          (число_экспериментов, количество_гипотез).
        """

        assert pvalues.ndim == 2,\
            "По условию задачи, adjust принимает матрицу "\
            "размерностей (число_экспериментов, количество_гипотез)"

        adjusted, reject = np.zeros((pvalues.shape[0], pvalues.shape[1])), np.zeros((pvalues.shape[0], pvalues.shape[1]))
        
        pvalues_s = np.sort(pvalues, axis=1)
        pval_ = (pvalues.shape[1] - np.arange(0, pvalues.shape[1])) * pvalues_s
        final_pval = []
        
        for i in range(pvalues.shape[1]):
            if i == 0:
                final_pval = np.minimum(1, pval_[:, i]).reshape(-1, 1)
                continue
            final_pval = np.hstack((final_pval, np.minimum(1, np.maximum(pval_[:, i], final_pval[:, i-1])).reshape(-1, 1)))
        
        pos = np.argsort(pvalues, axis=1)
        k = final_pval < self.alpha
        
        for i in range(pvalues.shape[0]):
            for j in range(pvalues.shape[1]):
                adjusted[i][pos[i][j]] = final_pval[i][j]
                reject[i][pos[i][j]] = k[i][j]
                
        
        assert (reject.shape == pvalues.shape
                and adjusted.shape == pvalues.shape),\
            "Размерности матриц reject и adjusted "\
            "отличаются от размерностей входных данных"
        assert np.all((0 <= adjusted) & (adjusted <= 1)),\
            "Некоторые из скорректированных p-значений не лежат в [0, 1]"
        return reject, adjusted


class BenjaminiYekutieliAdjustment(AdjustmentMethodABC):
    def __init__(self, alpha):
        super().__init__(
            name="Метод Бенджамини-Иекутиели",
            alpha=alpha,
            controls_for="FDR"
        )

    def adjust(self, pvalues):
        """
        Функция МПГ-коррекции.
        
        :param pvalues: Матрица p-value, имеет размер 
                        (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано p-value критерия S_j для проверки 
                        гипотезы H_j в эксперименте k.
            
        :return reject: булева матрица результатов проверки гипотез,
                        имеет размер (число_экспериментов, количество_гипотез).
                        В позиции (k, j) записано True, если в эксперименте k 
                        гипотеза H_j отвергается.
        :return adjusted: Матрица скорректированных p-value, имеет размер 
                          (число_экспериментов, количество_гипотез).
        """

        assert pvalues.ndim == 2,\
            "По условию задачи, adjust принимает матрицу "\
            "размерностей (число_экспериментов, количество_гипотез)"

        adjusted, reject = np.zeros((pvalues.shape[0], pvalues.shape[1])), np.zeros((pvalues.shape[0], pvalues.shape[1]))
        
        pvalues_s = np.sort(pvalues, axis=1)[:, -1::-1]
        pval_ = np.sum(1/np.arange(1, pvalues.shape[1]+1)) * pvalues_s.shape[1] / np.arange(pvalues.shape[1], 0,-1) * pvalues_s
        
        final_pval = []
        
        for i in range(pvalues_s.shape[1]):
            if i == 0:
                final_pval = np.minimum(1, pval_[:, i]).reshape(-1, 1)
                continue
            final_pval = np.hstack((final_pval, np.minimum(1, np.minimum(pval_[:, i], final_pval[:, i - 1])).reshape(-1, 1)))
        
        pos = np.argsort(pvalues, axis=1)[:, -1::-1]
        k = final_pval < self.alpha
        
        for i in range(pvalues.shape[0]):
            for j in range(pvalues.shape[1]):
                adjusted[i][pos[i][j]] = final_pval[i][j]
                reject[i][pos[i][j]] = k[i][j]
               
            
        assert (reject.shape == pvalues.shape
                and adjusted.shape == pvalues.shape),\
            "Размерности матриц reject и adjusted "\
            "отличаются от размерностей входных данных"
        assert np.all((0 <= adjusted) & (adjusted <= 1)),\
            "Некоторые из скорректированных p-значений не лежат в [0, 1]"
        return reject, adjusted


def criterion(samples, theta_0, Sigma):
    """
    Векторная реализация равномерно наиболее мощного критерия 
    для правосторонней гипотезы из условия.
    
    :param samples: Матрица выборок размера 
                    (число_экспериментов, размер_выборок, размерность_пространства)
    :param theta_0: Вектор средних, соответствующий основным гипотезам. 
                    Размер (размерность_пространства,)
    :param Sigma: Матрица ковариаций компонент вектора. 
                  Размер (размерность_пространства, размерность_пространства)
    
    :return pvalues: Матрица p-value, имеет размер 
                    (число_экспериментов, количество_гипотез).
                    В позиции (k, j) записано p-value критерия S_j для проверки 
                    гипотезы H_j в эксперименте k.
    """
    assert samples.ndim == 3, "На вход должен подаваться 3-мерный тензор"
    n_runs, sample_size, n_hypotheses = samples.shape

    pvalues = 1 - sps.norm.cdf(np.sqrt(sample_size) * (np.mean(samples, axis=1) - theta_0) / (np.diagonal(Sigma).reshape(Sigma.shape[0],)))

    assert pvalues.shape == (n_runs, n_hypotheses),\
        "Некорректная форма матрицы p-значений."\
        f"Должно быть {(n_runs, n_hypotheses)}, a вместо этого {pvalues.shape}"
    return pvalues
